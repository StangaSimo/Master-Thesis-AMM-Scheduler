#ifndef SCHEDULER_H
#define SCHEDULER_H

#ifdef ENABLE_OPENVINO
#include "ov_wrapper.h"
#include <openvino/op/util/attr_types.hpp>
#endif

#ifdef ENABLE_CUDA
#include "cuda_wrapper.h"
#endif

#ifdef ENABLE_SYCL
#include <sycl/vector.hpp>
#include "sycl_wrapper.h"
#endif

#ifdef ENABLE_OPENBLAS
#include "cpu.hpp"
#endif

//#include "cuda_wrapper.h"
//#include "ov_wrapper.h"
//#include "sycl_wrapper.h"
//#include "cpu.hpp"

#include "sharedbuffer.hpp"
#include "performancemap.hpp"
#include "profiler.hpp"
#include "tasks.hpp"
#include "config.hpp"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <atomic>
#include <stdlib.h>
#include <thread>
#include <array>
#include <memory>
#include <unistd.h>
#include <vector>
#include <fstream>

#ifdef DEBUG 
    #define PRINT(x) std::cout << "DEBUG: " << x << std::endl
#else
    #define PRINT(x)
#endif

#ifdef ENABLE_PROFILING
    #define PROF(x) x 
#else
    #define PROF(x) 
#endif

using namespace std;

/* Scheduler Logic Type */
enum Logic : size_t { 
    ROUND_ROBIN,
    CUDA_ONLY,
    STATIC_PARTITIONING,
    LARGE_MATRIX_SPLIT,
    LARGE_MATRIX_SPLIT_CUDA,
    STATIC_HETERO_PARTITIONING,
    DYNAMIC,
};

/****************************** Prototypes ************************************/
inline void handle_task(BT backend_type, task *task);
inline string get_benchmark_filename(BT bt, int type); 
inline void init_acc(BT backend_type); 
inline void free_acc(BT backend_type); 
inline void benchmark_acc(BT bt, Type type, string filename, int M, int N, int K);
inline void benchmark(BT bt, Type type, string filename);
inline string get_logic_string(Logic l); 
inline string get_acc_string(BT backend_type); 

/* for simplicity this is the buffer pointer */
using SharedBuffer_T = array<unique_ptr<SharedBuffer>, BT::COUNT>;

/****************************** Scheduler Class ************************************/

class AMScheduler {
    private:

        /* atomic for shutting down threads */
        atomic<bool> threads_keep_running;
        atomic<int> pending_tasks = 0;
        atomic<bool>* busy[BT::COUNT];

        /* shared buffers beetween threads */
        SharedBuffer_T shared_buffers;

        /* array of threads, one for each accellerator */
        array<unique_ptr<thread>, BT::COUNT> threads; 

        /* type of scheduler strategy to use */
        Logic strategy;

        /* backend_type vector for handling the accellerators */
        vector<BT> bts; 

        /* Perfomance Map for each accellerator */
        array<unique_ptr<PerformanceMap>, BT::COUNT> bts_map; 

        /* metrics profiler */
        Profiler profiler = Profiler();

        /* pointer for split_matrix logic*/ 
        vector<task*> sub_task_array;

/**********************************  Worker  ***********************************/

        void worker_thread(BT bt, SharedBuffer *buffer) { 
            task* task = nullptr;
            PROF(profiler.init_last_work(bt));

            if (strategy == Logic::DYNAMIC) {

                /* Dynamic execution */
                while (threads_keep_running) { 
                    task = buffer->get();
                    if (!task) {continue;}

                    PROF(profiler.start_worker(bt));                 
                    handle_task(bt, task);
                    PROF(profiler.end_worker(bt));                 

                    *busy[bt] = false;
                    pending_tasks--;
                }

            } else {

                /* Static execution */
                while (threads_keep_running) { 
                    /* blocking if SLEEP on */
                    task = buffer->get();

                    /* if there isn't a task we check if we have to shutdown */
                    if (!task) {continue;}

                    PROF(profiler.start_worker(bt));                 
                    handle_task(bt, task);
                    PROF(profiler.end_worker(bt));                 
                    pending_tasks--;
                }

            }
        } 

        /**********************************  Coordinator  ***********************************/

        /* if there is no task, we check if we have to shutdown */
        void coordinator_thread(SharedBuffer_T &buffers) { 
            int i=0;
            int c=0;
            int bts_len = bts.size(); /* number of accellerators */

            double acc_speed[bts_len];
            double total_speed=0.0;
            int n_acc_task[BATCH_SIZE];
            int remaining_c=0;
            int t=0;

            SharedBuffer *buffer = buffers[BT::CORDINATOR].get();
            task* single_task = nullptr;
            task* tasks[BATCH_SIZE];
            task* tasks_hetero[BATCH_SIZE_HETERO];

            bool dispached = true;

            double bt_ms[bts_len];
            for (int i=0; i<bts_len; i++)
                bt_ms[i]=0.0;

            switch (strategy) {
            case Logic::CUDA_ONLY:

                PROF(profiler.start_power_monitor());

                if (buffers[BT::CUDA] == nullptr) {
                    cerr << "[SCHEDULER] Error, cuda logic but no cuda card";
                    exit(EXIT_FAILURE);
                }

                while (threads_keep_running) { 
                    single_task = buffer->get();
                    if (!single_task) {continue;} 
                    buffers[BT::CUDA]->put(single_task);
                } 

                PROF(profiler.stop_power_monitor());

                break;

            /* basic logic, just send all the tasks to all the bt */
            case Logic::ROUND_ROBIN: 

                while (threads_keep_running) { 
                    single_task = buffer->get();
                    if (!single_task) {continue;} 

                    buffers[bts[i]]->put(single_task);

                    i = (i+1) % bts_len;
                } 
                break;

                /* batch style partitioning */
            case Logic::STATIC_PARTITIONING: 

                PROF(profiler.start_power_monitor());

                while (threads_keep_running) { 

                    PROF(profiler.start_fetch());

                    while (c < BATCH_SIZE) {
                        tasks[c] = buffer->get();
                        if (!tasks[c]) {break;} 
                        c++;
                    }

                    PROF(profiler.stop_fetch());

                    PROF(profiler.start_logic());

                    if (c == 0) {continue;}

                    /* if tasks less number of bts, send all to the fastest */
                    if (bts_len >= c) { 
                        for (int j=0; j<c; j++)
                            buffers[bts[0]]->put(tasks[j]);
                        c = 0;
                        continue;
                    }

                    /* speeds */
                    total_speed = 0.0;
                    for (int j=0; j<bts_len; j++){
                        double single_matrix_ms = bts_map[bts[j]]->query(tasks[0]->M, tasks[0]->N, tasks[0]->K, tasks[0]->type, true);
                        acc_speed[j] = 1.0 / (single_matrix_ms + 1e-9); /* 1e-9 for not dividing for 0 */       
                        total_speed += acc_speed[j];
                    }
                    
                    /* exact task per acc (double) */
                    double shares[bts_len];
                    int total_assigned = 0;
                    
                    for (int j=0; j<bts_len; j++) {
                        double ratio = acc_speed[j] / total_speed;
                        shares[j] = ratio * (double)c;
                        n_acc_task[j] = (int)shares[j]; /* integer part */
                        total_assigned += n_acc_task[j];
                    }

                    int missing_c = c - total_assigned;
                    
                    /* assign the missing task to the accellerator 
                     * mitigate slowness between accellerators */
                    while (missing_c > 0) {
                        int best_j = -1;
                        double max_decimal = -1.0;

                        /* find the accellerator with the highest decimal part */
                        for (int j=0; j<bts_len; j++) {
                            double decimal_part = shares[j] - (double)n_acc_task[j];
                            if (decimal_part > max_decimal) {
                                max_decimal = decimal_part;
                                best_j = j;
                            }
                        }

                        /* assign one more task to the winner */
                        if (best_j != -1) {
                            n_acc_task[best_j]++;
                            /* set to -1 so it won't be picked again */
                            shares[best_j] = -1.0; 
                            missing_c--;
                        } else {
                            /* fallback */
                            n_acc_task[0]++;
                            missing_c--;
                        }
                    }

                    PROF(profiler.stop_logic());

                    PROF(profiler.start_dispatch());

                    /* send task to the accellerators */
                    t = 0;
                    for (int j=0; j<bts_len; j++) {
                        for (int w=0; w<n_acc_task[j]; w++){
                            if (t >= c) break;
                            
                            buffers[bts[j]]->put(tasks[t]);
                            t++;
                        }
                    }
                    
                    c = 0; 
                    PROF(profiler.stop_dispatch());
                    PROF(profiler.record_sample());
                } 

                PROF(profiler.stop_power_monitor());
                break;
            
            /* split a large matrix row wise with Iterative Refinement Partitioning*/
            case Logic::LARGE_MATRIX_SPLIT:

                PROF(profiler.start_power_monitor());

                while (threads_keep_running) { 
                    PROF(profiler.start_fetch());

                    single_task = buffer->get();
                    if (!single_task) {continue;} 

                    PROF(profiler.stop_fetch());
                    PROF(profiler.start_logic());

                    double curr_speeds[bts_len];
                    int rows[bts_len];

                    int rest = single_task->M % bts_len;
                    int base = single_task->M / bts_len;

                    /* equal row initially */
                    for (int j=0; j<bts_len; j++) 
                        rows[j] = base + (j < rest ? 1 : 0);

                    /* loop for adjusting the ratios */
                    for (int i = 0; i < SPLIT_MATRIX_ITERATION; i++) {
                        double total_speed = 0.0;

                        /* current speed for 1 row */
                        for (int j=0; j<bts_len; j++) {
                            int r = (rows[j] > 0) ? rows[j] : 1;

                            double ms = 0.0;

                            ms = bts_map[bts[j]]->query(r, single_task->N, single_task->K, single_task->type, false);

                            /* update time if r exceed max size */
                            if (r > MAX_SIZE){
                                double max_ms = bts_map[bts[j]]->query(MAX_SIZE, single_task->N, single_task->K, single_task->type, false);
                                ms = max_ms * ((double)r / (double)MAX_SIZE);
                            }


                            cout << "acc " << j << " ms " <<  ms << " for M: " << r << " N: " << single_task->N << " K: " << single_task->K << " \n";

                            /* single row speed, number of rows / ms */
                            double speed = (double)r / (ms + 1e-9);

                            curr_speeds[j] = speed;
                            total_speed += speed;
                        }

                        /* adjust the row with the ratios */
                        int rows_distributed = 0;
                        for (int j=0; j<bts_len; j++) {
                            double ratio = curr_speeds[j] / total_speed;

                            /* new rows from the ratio */
                            int target_rows = (int)(ratio * single_task->M);

                            rows[j] = target_rows;
                            rows_distributed += rows[j];
                        }
                    }

                    PROF(profiler.stop_logic());
                    PROF(profiler.start_dispatch());

                    size_t elem_size = (single_task->type == Type::FLOAT) ? 4 : 2;
                    size_t offset_rows_acc = 0;

                    /* dispatch tasks */
                    for (int j=0; j<bts_len; j++) {
                        int h = rows[j];
                        cout << get_acc_string(bts[j]) << " get " << h << " rows\n";
                        if (h <= 0) continue; 

                        /* divide rows into tasks if exceed MAX_SIZE */
                        while (h > 0) {
                            int h_limit = h;
                            
                            if (h_limit > MAX_SIZE) 
                                h_limit = MAX_SIZE;

                            task* sub_task = new ::task;
                            *sub_task = *single_task;

                            /* save the pointer for later freeup */
                            sub_task_array.push_back(sub_task);

                            /* A row */
                            sub_task->M = h_limit; 

                            /* A offset */
                            size_t offset_A = offset_rows_acc * single_task->K * elem_size;
                            sub_task->A = (char*)single_task->A + offset_A;

                            /* B offset still all in memory */
                            sub_task->B = single_task->B;

                            /* C offset */
                            size_t offset_C = offset_rows_acc * single_task->N * elem_size;
                            sub_task->C = (char*)single_task->C + offset_C;

                            /* add task to the pending */
                            pending_tasks++;

                            /* submit to the acc */
                            buffers[bts[j]]->put(sub_task);

                            /* for each accellerator we add more offset */
                            offset_rows_acc += h_limit; 
                            
                            h -= h_limit;
                        }
                    }

                    /* remove task */
                    pending_tasks--;
                    PROF(profiler.stop_dispatch());
                    PROF(profiler.record_sample());    
                } 

                PROF(profiler.stop_power_monitor());
                break;

            case Logic::STATIC_HETERO_PARTITIONING:  

                PROF(profiler.start_power_monitor());

                while (threads_keep_running) { 

                    PROF(profiler.start_fetch());

                    while (c < BATCH_SIZE_HETERO) {
                        tasks_hetero[c] = buffer->get();
                        if (!tasks_hetero[c]) {break;} 
                        c++;
                    }

                    PROF(profiler.stop_fetch());

                    PROF(profiler.start_logic());
                    if (c == 0) {continue;}


                    /* iterator for each bt */ 
                    int i_bt[bts_len];

                    for (int i=0; i<bts_len; i++)
                        i_bt[i] = 0;
                    
                    task* dispatch_tasks[bts_len][BATCH_SIZE_HETERO];
                    
                    /* for every task */
                    for (int i = 0; i < c; i++) {
                        int best_bt = -1;
                        double min_ms = 1e15;

                        /* for every bt */
                        for (int j = 0; j < bts_len; j++) {
                            double query_ms = bts_map[bts[j]]->query(tasks_hetero[i]->M, tasks_hetero[i]->N, tasks_hetero[i]->K, tasks_hetero[i]->type, false);

                            double finish_ms = bt_ms[j] + query_ms;

                            /* we assign the task to the least "filled" */
                            if (finish_ms < min_ms) {
                                min_ms = finish_ms;
                                best_bt = j;
                            }
                        }

                        /* openvino and sycl jit updates */
                        if (bts[best_bt] == BT::OPENVINO || bts[best_bt] == BT::SYCL)
                            bts_map[bts[best_bt]]->query(tasks_hetero[i]->M, tasks_hetero[i]->N, tasks_hetero[i]->K, tasks_hetero[i]->type, true);

                        dispatch_tasks[best_bt][i_bt[best_bt]] = tasks_hetero[i];
                        i_bt[best_bt]++;
                        bt_ms[best_bt] = min_ms;
                    }

                    PROF(profiler.stop_logic());

                    PROF(profiler.start_dispatch());
                    for (int i = 0; i < bts_len; i++) {
                       cout << "acc" << get_acc_string(bts[i]) << " contiene : " << i_bt[i] <<  " tasks con " << bt_ms[i] << " ms di carico \n";

                        for (int j=0; j<i_bt[i]; j++)
                            buffers[bts[i]]->put(dispatch_tasks[i][j]);

                    }
                    PROF(profiler.stop_dispatch());
                    PROF(profiler.record_sample());    
                    c = 0;
                }

                PROF(profiler.stop_power_monitor());
                break;

            case Logic::DYNAMIC:  

                PROF(profiler.start_power_monitor());

                i = 0;
                while (threads_keep_running) { 
                    PROF(profiler.start_logic());

                    if (dispached) {
                        PROF(profiler.start_fetch());
                        single_task = buffer->get();
                        PROF(profiler.stop_fetch());
                        if (!single_task) {continue;} 
                        dispached = false;
                    }

                    if (*busy[bts[i]] == false) {
                        *busy[bts[i]] = true; 
                        dispached = true;
                        PROF(profiler.start_dispatch());
                        buffers[bts[i]]->put(single_task);
                        PROF(profiler.stop_dispatch());
                    }

                    i++;
                    i = i % bts_len;

                    PROF(profiler.stop_logic());

                    if (dispached)
                        PROF(profiler.record_sample());    
                }


                PROF(profiler.stop_power_monitor());
                break;

            case Logic::LARGE_MATRIX_SPLIT_CUDA:

                PROF(profiler.start_power_monitor());

                if (buffers[BT::CUDA] == nullptr) {
                    cout << "[SCHEDULER] Error: no CUDA backend available.\n";
                    exit(EXIT_FAILURE);
                }

                while (threads_keep_running) { 
                    PROF(profiler.start_fetch());

                    single_task = buffer->get();
                    if (!single_task) {continue;} 

                    PROF(profiler.stop_fetch());
                    
                    PROF(profiler.start_logic());
                    int rows_left = single_task->M;
                    int offset_rows = 0;
                    size_t elem_size = (single_task->type == Type::FLOAT) ? 4 : 2;
                    PROF(profiler.stop_logic());

                    PROF(profiler.start_dispatch());

                    while (rows_left > 0) {
                        
                        int chunk_h = rows_left;
                        if (chunk_h > MAX_SIZE) chunk_h = MAX_SIZE;

                        task* sub_task = new ::task;
                        *sub_task = *single_task; 

                        sub_task_array.push_back(sub_task);

                        sub_task->M = chunk_h;

                        size_t offset_A = (size_t)offset_rows * single_task->K * elem_size;
                        size_t offset_C = (size_t)offset_rows * single_task->N * elem_size;

                        sub_task->A = (char*)single_task->A + offset_A;
                        sub_task->C = (char*)single_task->C + offset_C;

                        pending_tasks++;

                        buffers[BT::CUDA]->put(sub_task);

                        rows_left -= chunk_h;
                        offset_rows += chunk_h;
                    }

                    pending_tasks--;
                    
                    PROF(profiler.stop_dispatch());
                    PROF(profiler.record_sample());
                }

                PROF(profiler.stop_power_monitor());
                break;
            default:
                printf("[SCHEDULER] Error in chosing the logic.\n");
                exit(EXIT_FAILURE);
            }
        }

/**********************************  Init and Stop  ***********************************/

        void init_threads() {

            /* from the fastest to the slowest */
            /* init here and not in the threads */
#ifdef ENABLE_CUDA
            init_acc(BT::CUDA);
            bts.push_back(BT::CUDA);
#endif
#ifdef ENABLE_SYCL
            init_acc(BT::SYCL);
            bts.push_back(BT::SYCL);
#endif
#ifdef ENABLE_OPENVINO
            init_acc(BT::OPENVINO); 
            bts.push_back(BT::OPENVINO);
#endif
#ifdef ENABLE_OPENBLAS
            init_acc(BT::OPENBLAS); 
            bts.push_back(BT::OPENBLAS);
#endif

            /* initialize atomics for dynamic strategy */
            if(strategy == Logic::DYNAMIC) {
                #ifdef SLEEP 
                cout << "[SCHEDULER] ERROR remove "#define SLEEP" with Dynamic logic\n";
                exit(EXIT_FAILURE);
                #endif
                for (int i=0; i<bts.size(); i++) {
                    busy[bts[i]] = new atomic<bool>;
                    *busy[bts[i]] = false;
                }
            }

            for (BT i : bts) {
                shared_buffers[i] = make_unique<SharedBuffer>(BUFFER_LENGHT);
                threads[i] = make_unique<thread>(&AMScheduler::worker_thread, this, 
                                                    i, shared_buffers[i].get());
            }

            if (bts.size() == 0) {PRINT("[SCHEDULER] ATTENTION, no accellerator\n"); exit(EXIT_FAILURE);}

            /* coordinator thread */
            shared_buffers[BT::CORDINATOR] = make_unique<SharedBuffer>(BUFFER_LENGHT);
            threads[BT::CORDINATOR] = make_unique<thread>(&AMScheduler::coordinator_thread,this, 
                                        ref(shared_buffers)); /* ref because the thread function try to copy all the parameters */
        }

        /* shut down threads */
        void stop_threads() {
            threads_keep_running = false;

            for (auto& thread : threads) 
                if (thread) /* unique pointer are null if not initializated */
                    if(thread->joinable())
                        thread->join();

            /* shut down backends */
            for (auto i : bts) 
                free_acc(i);
            
            /* cleanups sub_task array */
            if (strategy == Logic::LARGE_MATRIX_SPLIT)
                for (int i=0; i<sub_task_array.size(); i++)
                    delete sub_task_array[i];

            /* cleanups atomics */
            if(strategy == Logic::DYNAMIC)
                for (int i=0; i<bts.size(); i++)
                    delete busy[bts[i]];

            PRINT("[SCHEDULER] Threads stopped.\n");
        }

        /**********************************  Init Benchmarks and Map ***********************************/
         
        /* upload the banchmark files and generate them if not presents */
        void init_benchmarks() {
            for (BT i : bts) {

                string filename_f; 
                string filename_h; 

                filename_f = get_benchmark_filename(i, Type::FLOAT); 
                filename_h = get_benchmark_filename(i, Type::HALF); 

                if (filesystem::exists(filename_f)) {
                     PRINT("[SCHEDULER] File: " << filename_f <<  " opened.\n"); 
                } else {
                     PRINT("[SCHEDULER] File: " << filename_f <<  " not present\n"); 
                     benchmark(i, Type::FLOAT ,filename_f);
                     PRINT("[SCHEDULER] Benchmark file created\n"); 
                }

                if (filesystem::exists(filename_h)){
                     PRINT("[SCHEDULER] File: " << filename_h <<  " opened.\n"); 
                } else {
                     PRINT("[SCHEDULER] File: " << filename_h <<  " not present\n"); 
                     benchmark(i, Type::HALF ,filename_h);
                     PRINT("[SCHEDULER] Benchmark file created\n"); 
                }
            }
        }

        /* create the map for each accellerator */
        void init_maps() {
            for (BT i : bts) {
                string filename_f; 
                string filename_h; 
                filename_f = get_benchmark_filename(i, Type::FLOAT); 
                filename_h = get_benchmark_filename(i, Type::HALF); 

                if (filesystem::exists(filename_f) && filesystem::exists(filename_h)) {
                    bts_map[i] = make_unique<PerformanceMap>(STEP_SIZE, i, filename_f, filename_h);
                } else {
                     cerr << "[SCHEDULER] File: " << filename_f <<  " or " << filename_h << " not found for map init\n"; 
                     exit(EXIT_FAILURE);
                }
           }
        }

        /**********************************  Public Interface ***********************************/

    public: 
        /* constructur init threads, benchmark files, perfomanceMaps */
        AMScheduler(Logic logic) : threads_keep_running(true), strategy(logic) {
            get_logic_string(logic);
            init_threads();            
            PRINT("[SCHEDULER] Threads started.\n");
            init_benchmarks();            
            PRINT("[SCHEDULER] Benchmarks done.\n");
            init_maps();
            PRINT("[SCHEDULER] PerformanceMap done.\n");
        }

        /* destructur, stop the threads*/
        ~AMScheduler() {
            stop_threads();
        }

        /* send the tasks to the coordinator */
        void do_tasks(task* tasks, size_t n) {
            pending_tasks += n;
            for (int i=0; i<n; i++) {
                tasks[i].start_time = chrono::high_resolution_clock::now();                
                shared_buffers[BT::CORDINATOR]->put(&tasks[i]);
            }
            PRINT("[SCHEDULER] Task dispached.\n");
        }

        /* check if all the buffers are empty */
        void wait() {
            PRINT("[SCHEDULER] Waiting.. \n");

            while (pending_tasks > 0) {
                usleep(1000);
            }

            PRINT("[SCHEDULER] All tasks done. \n");
        }

        /* return the profiler for printing the stats */
        void print_stats(task* tasks, int num_tasks) {
            if (strategy == Logic::LARGE_MATRIX_SPLIT) {
                profiler.print_stats(bts, sub_task_array, get_logic_string(strategy));
            } else {
                profiler.print_stats(bts, tasks, num_tasks, get_logic_string(strategy));
            }
        }
};

/**********************************  Helper Functions ***********************************/

/* benchmark the accellerator and write in the csv*/
/* RECALL: NON USARE LA SOLITA TASK PER IL BENCHMARK per la cache */
inline void benchmark_acc(BT bt, Type type, string filename, int M, int N, int K) {
    chrono::high_resolution_clock::time_point start_time;
    chrono::high_resolution_clock::time_point end_time;
    chrono::duration<double, std::milli> duration;

    const int N_RUNS = 4;
    const int N_TASKS = N_RUNS + 2;

    double total_ms = 0.0;
    double jit_ms = 0.0;
    double no_jit_ms = 0.0;

    /* we simulate a thread */
    task* tasks = init_tasks(N_TASKS, M, N, K, type);

    /* warmup | jit detection */
    start_time = chrono::high_resolution_clock::now();
    handle_task(bt, &tasks[0]);
    end_time = chrono::high_resolution_clock::now();
    duration = end_time - start_time;
    jit_ms = duration.count();

    start_time = chrono::high_resolution_clock::now();
    handle_task(bt, &tasks[1]);
    end_time = chrono::high_resolution_clock::now();
    duration = end_time - start_time;
    jit_ms = jit_ms - duration.count();

    /* runs with */
    for (int i = 2; i < N_TASKS; i++) {
        start_time = chrono::high_resolution_clock::now();
        handle_task(bt, &tasks[i]);
        end_time = chrono::high_resolution_clock::now();
        duration = end_time - start_time;
        total_ms += duration.count();
    }

    clean_tasks(tasks, N_TASKS);
    
    double avg_ms = total_ms/N_RUNS;

    ofstream file(filename, ios::app);

    if (file.is_open()) {
        if (filesystem::file_size(filename) == 0) 
            file << "Accelerator,DataType,M,N,K,Avg_Time_ms,Jit_Time_ms\n";
        
        string acc_str = get_acc_string(bt);
        file << acc_str << "," << type << "," << M << "," << N << "," << K << "," << avg_ms << "," << jit_ms << "\n";
        
        PRINT("result: " << acc_str << "," << type << "," << M << "," << N << "," << K << "," << avg_ms << "," << jit_ms << "\n");
        file.close();
    } else {
        cerr << "[SCEDULER] ERROR Cannot open file: " << filename << "\n";
    }
}

/* call benchmark_acc for each test matrix*/
inline void benchmark(BT bt, Type type, string filename) {
    vector<int> dims;

    for (int d = STEP_SIZE; d <= STEP_SIZE*STEP_TOTAL; d += STEP_SIZE) 
        dims.push_back(d);

    for (int m : dims) 
        for (int n : dims) 
            for (int k : dims) 
                benchmark_acc(bt, type, filename, m, n, k);

    PRINT("[SCHEDULER] Benchmark ===\n");
    PRINT("Results saved in bin/csv/" << filename << "\n");
}

/* task handler for the workers, choose the right backend_type */
inline void handle_task(BT backend_type, task *task) {
    switch (backend_type) {
    case BT::CUDA:
#ifdef ENABLE_CUDA
        if (task->type == Type::FLOAT)
            cuda_gemm_32bit(task->A,task->B,task->C, task->M, task->N, task->K);
        if (task->type == Type::HALF)
            cuda_gemm_16bit(task->A,task->B,task->C, task->M, task->N, task->K);
#endif 
        break;    

    case BT::SYCL:
#ifdef ENABLE_SYCL
        if (task->type == Type::FLOAT)
            sycl_gemm_32bit(task->A,task->B,task->C, task->M, task->N, task->K);
        if (task->type == Type::HALF)
            sycl_gemm_16bit(task->A,task->B,task->C, task->M, task->N, task->K);
#endif 
        break;    

    case BT::OPENVINO:
#ifdef ENABLE_OPENVINO
        if (task->type == Type::FLOAT)
            ov_gemm_32bit(task->A,task->B,task->C, task->M, task->N, task->K);
        if (task->type == Type::HALF)
            ov_gemm_16bit(task->A,task->B,task->C, task->M, task->N, task->K);
#endif 
        break;    

    case BT::OPENBLAS:
#ifdef ENABLE_OPENBLAS
        if (task->type == Type::FLOAT)
            cpu_gemm_32bit(task->A,task->B,task->C, task->M, task->N, task->K);
        if (task->type == Type::HALF)
            cpu_gemm_16bit(task->A,task->B,task->C, task->M, task->N, task->K);
#endif 
        break;    

    default: 
        cerr << "[SCHEDULER] ERROR handle task\n";
        exit(EXIT_FAILURE);
    }

    task->end_time = chrono::high_resolution_clock::now();
}

/* call the init func of the accellerator */
inline void init_acc(BT backend_type) {
    switch (backend_type) {
    case BT::CUDA:
#ifdef ENABLE_CUDA
        cuda_init();
#endif 
        break;
    case BT::SYCL:
#ifdef ENABLE_SYCL
        sycl_init();
#endif 
        break;    
    case BT::OPENVINO:
#ifdef ENABLE_OPENVINO
        ov_init();
#endif 
        break;    
    case BT::OPENBLAS:
#ifdef ENABLE_OPENBLAS
        cpu_init();
#endif 
        break;    
    default: 
        cerr << "[SCHEDULER] ERROR Change Status\n";
        exit(EXIT_FAILURE);
    }
}

/* call the init func of the accellerator */
inline void free_acc(BT backend_type) {
    switch (backend_type) {
    case BT::CUDA:
#ifdef ENABLE_CUDA
        cuda_free();
#endif 
    case BT::SYCL:
#ifdef ENABLE_SYCL
        sycl_free();
#endif 
        break;    
    case BT::OPENVINO:
#ifdef ENABLE_OPENVINO
        ov_free();
#endif 
        break;    
    case BT::OPENBLAS:
#ifdef ENABLE_OPENBLAS
#endif 
        break;    
    default: 
        cerr << "[SCHEDULER] ERROR Change Status\n";
        exit(EXIT_FAILURE);
    }
}

inline string get_logic_string(Logic l) {
    string logic;
    switch (l) {
        case Logic::ROUND_ROBIN:
            logic = "ROUND_ROBIN"; break;
        case Logic::CUDA_ONLY:
            logic = "CUDA_ONLY"; break;
        case Logic::STATIC_PARTITIONING:
            logic = "STATIC_PARTITIONING"; break;
        case Logic::LARGE_MATRIX_SPLIT:
            logic = "LARGE_MATRIX_SPLIT"; break;
        case Logic::LARGE_MATRIX_SPLIT_CUDA:
            logic = "LARGE_MATRIX_SPLIT_CUDA"; break;
        case Logic::STATIC_HETERO_PARTITIONING:
            logic = "STATIC_HETERO_PARTITIONING"; break;
        case Logic::DYNAMIC:
            logic = "DYNAMIC"; break;
        default:
            cerr << "[SCHEDULER] ERROR print_logic \n";
            exit(EXIT_FAILURE);
    }
    PRINT("[SCHEDULER] Started with " << logic << " logic \n");
    return logic;
}

/* get the right csv filename for each accellerators */
inline string get_benchmark_filename(BT bt, int type) {
    string base_path = "bin/csv/";
    if (!filesystem::exists(base_path)) {
        filesystem::create_directories(base_path);
    }

    string acc_name = get_acc_string(bt) ;

    string type_name;
    switch (type) {
        case Type::FLOAT: type_name = "FLOAT"; break;
        case Type::HALF: type_name = "HALF"; break;
        case Type::UINT8: type_name = "UINT8"; break;
        default:
            cerr << "[SCHEDULER] ERROR type get_benchmark_filename \n";
            exit(EXIT_FAILURE);
    }

    return base_path + acc_name + "_" + type_name + ".csv";
}
#endif
