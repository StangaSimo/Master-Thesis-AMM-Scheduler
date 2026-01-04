#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "cuda_wrapper.h"
#include "ov_wrapper.h"
#include "sycl_wrapper.h"

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

enum Logic : size_t { /* Scheduler Logic Type */
    ROUND_ROBIN,
    CUDA_ONLY,
    STATIC_PARTITIONING,
};

/******* Prototypes ********/
inline void handle_task(BT backend_type, task *task);
inline string get_benchmark_filename(BT bt, int type); 
inline void init_acc(BT backend_type); 
inline void free_acc(BT backend_type); 
inline void benchmark_acc(BT bt, Type type, string filename, int M, int N, int K);
inline void benchmark(BT bt, Type type, string filename);
inline void print_logic(Logic l); 
inline string get_acc_string(BT backend_type); 

/* for simplicity this is the buffer pointer */
using SharedBuffer_T = array<unique_ptr<SharedBuffer>, BT::COUNT>;

/****************************** Scheduler Class ************************************/

class AMScheduler {
    private:

        /* atomic for shutting down threads */
        atomic<bool> threads_keep_running;

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

/**********************************  Worker  ***********************************/

        void worker_thread(BT bt, SharedBuffer *buffer) { 
            task* task = nullptr;
            
            PROF(profiler.init_last_work(bt));
            while(threads_keep_running) { 

                /* blocking get for max 50ms if SLEEP on */
                task = buffer->get();

                /* if there isn't a task we check if we have to shutdown */
                if (!task) {continue;}

                PROF(profiler.start_worker(bt));                 
                handle_task(bt, task);
                PROF(profiler.end_worker(bt));                 
            }
        } 

/**********************************  Coordinator  ***********************************/

        /* if there is no task, we check if we have to shutdown */
        void coordinator_thread(SharedBuffer_T &buffers) { 
            int i=0;
            int c=0;
            int bts_len = bts.size();

            double acc_speed[bts_len];
            double total_speed = 0.0;
            int n_acc_task[BATCH_SIZE];
            int remaining_c=0;
            int t=0;
            

            SharedBuffer *buffer = buffers[BT::CORDINATOR].get();
            task* single_task = nullptr;
            task* tasks[BATCH_SIZE];

            switch (strategy) {
            case Logic::CUDA_ONLY:

                if (buffers[BT::CUDA] == nullptr) {
                    cerr << "[SCHEDULER] Error, cuda logic but no cuda card";
                    exit(EXIT_FAILURE);
                }

                while(threads_keep_running) { 
                    single_task = buffer->get();
                    if (!single_task) {continue;} 
                    buffers[BT::CUDA]->put(single_task);
                } 

                break;

            /* basic logic, if there is one that don't do nothing just send them request */
            case Logic::ROUND_ROBIN: 

                while(threads_keep_running) { 
                    single_task = buffer->get();
                    if (!single_task) {continue;} 

                    buffers[bts[i]]->put(single_task);

                    i = (i+1) % bts_len;
                } 
                break;

            /* batch style partitioning with time decision assuming same matrix */
            case Logic::STATIC_PARTITIONING: 

                while(threads_keep_running) { 
                    
                    PROF(profiler.start_fetch());

                    while (c < BATCH_SIZE) {
                        tasks[c] = buffer->get();
                        if (!tasks[c]) {break;} 
                        c++;
                    }

                    PROF(profiler.stop_fetch());

                    PROF(profiler.start_logic());
                    if (c == 0) {continue;}

                    if (c == 1) { /* 1 task, to the fastest */
                        buffers[bts[0]]->put(tasks[0]);
                        c = 0;
                        continue;
                    }

                    if (bts_len >= c){  /* the remaining tasks TODO: i don't think so, but lets see */
                        for (int j=0; j<c; j++)
                            buffers[bts[j]]->put(tasks[j]);
                        c = 0;
                        continue;
                    }

                    /* velocity = 1 / time */
                    total_speed = 0.0;
                    for (int j=0; j<bts_len; j++){
                        double single_matrix_ms = bts_map[bts[j]]->query(tasks[0]->M, tasks[0]->N, tasks[0]->K, tasks[0]->type);
                        acc_speed[j] = 1.0 / (single_matrix_ms * (double) c);        
                        total_speed += acc_speed[j];
                    }
                    
                    /* acc_speed = 1_matrix_ms * c, total_speed = acc_speed[j] */ 
                    remaining_c = c;                        
                    for (int j=0; j<bts_len; j++) {
                        double ratio = acc_speed[j] / total_speed;
                        n_acc_task[j] = (int)(ratio * c);
                        remaining_c -= n_acc_task[j];
                    }

                    PROF(profiler.stop_logic());

                    PROF(profiler.start_dispatch());
                    /* send task to the accellerators */
                    t = 0;
                    for (int j=0; j<bts_len; j++)
                        for (int w=0; w<n_acc_task[j]; w++){
                            buffers[bts[j]]->put(tasks[t]);
                            t++;
                        }

                    /* send the remaining to the fastest */
                    if (remaining_c != 0)
                        for (int j=c-remaining_c; j<c; j++)
                            buffers[bts[0]]->put(tasks[j]);
                   
                    c = 0; 

                    PROF(profiler.stop_dispatch());
                    PROF(profiler.record_sample());
                } 
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

            for (BT i : bts) {
                shared_buffers[i] = make_unique<SharedBuffer>(BUFFER_LENGHT);
                threads[i] = make_unique<thread>(&AMScheduler::worker_thread, this, 
                                                    i, shared_buffers[i].get());
            }

            if (bts.size() == 0) {PRINT("[SCHEDULER] ATTENTION, only CPU up\n");}

            //bts.push_back(BT::CPU); TODO remove this when implementing the cpus and add init
            /* default CPU thread */
            shared_buffers[BT::CPU] = make_unique<SharedBuffer>(BUFFER_LENGHT);
            threads[BT::CPU] = make_unique<thread>(&AMScheduler::worker_thread, this, 
                                                    BT::CPU, shared_buffers[BT::CPU].get());
            /* coordinator */
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

            /* shut down back ends */
            for (auto i : bts) 
                free_acc(i);

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

                if (filesystem::exists(filename_f)){
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

                if (filesystem::exists(filename_f) && filesystem::exists(filename_h)){
                    bts_map[i] = make_unique<PerformanceMap>(STEP_SIZE,filename_f,filename_h);
                } else {
                     cerr << "[SCHEDULER] File: " << filename_f <<  " or " << filename_h << " not found for map init\n"; 
                     exit(EXIT_FAILURE);
                }
           }
        }

        /**********************************  Public Interface ***********************************/

    public: 
        /* constructur init threads, benchmark files, regression models */
        AMScheduler(Logic logic) : threads_keep_running(true), strategy(logic) {
            print_logic(logic);
            init_threads();            
            PRINT("[SCHEDULER] Threads started.\n");
            init_benchmarks();            
            PRINT("[SCHEDULER] Benchmarks done.\n");
            init_maps();
            PRINT("[SCHEDULER] Regression done.\n");
        }

        /* destructur, stop the threads*/
        ~AMScheduler() {
            stop_threads();
        }

        /* just send the tasks to the coordinator */
        void do_tasks(task* tasks, size_t n) {
            for (int i=0; i<n; i++) {
                tasks[i].start_time = chrono::high_resolution_clock::now();                
                shared_buffers[BT::CORDINATOR]->put(&tasks[i]);
            }
            PRINT("[SCHEDULER] Task dispached\n");
        }

        /* check if all the buffers are empty */
        void wait() {
            PRINT("[SCHEDULER] Waiting.. \n");
            /* wait coordinator */
            while (!shared_buffers[0]->is_empty()){
                usleep(10000);
            }
            PRINT("[SCHEDULER] Coordinator empty.. \n");

            /* wait other threads */
            for (int i=1; i<BT::COUNT; i++) {
                if (shared_buffers[i] != nullptr) {
                    while (!shared_buffers[i]->is_empty()){
                        usleep(10000);
                    }
                }
            }

            PRINT("[SCHEDULER] All empty.. \n");

            // TODO: maybe better shutdown? 
            sleep(2); /* wait for unfinished matrix */
        }

        /* return the profiler for printing the stats */
        void print_profiler_stats() {profiler.print_stats(bts);}
};

/**********************************  Helper Functions ***********************************/

/* benchmark the accellerator and write in the csv*/
inline void benchmark_acc(BT bt, Type type, string filename, int M, int N, int K) {
    chrono::high_resolution_clock::time_point start_time;
    chrono::high_resolution_clock::time_point end_time;
    chrono::duration<double, std::milli> duration;

    const int N_WARMUP = 2;
    const int N_RUNS = 5;

    double total_ms = 0.0;

    /* we simulate a thread */
    task* task = init_tasks(1, M, N, K, type);

    for (int i = 0; i < N_WARMUP; i++)
        handle_task(bt, &task[0]);

    for (int i = 0; i < N_RUNS; i++) {
        start_time = chrono::high_resolution_clock::now();
        handle_task(bt, task);
        end_time = chrono::high_resolution_clock::now();
        duration = end_time - start_time;
        total_ms += duration.count();
    }

    clean_tasks(task, 1, type);

    double avg_ms = total_ms / N_RUNS;
    ofstream file(filename, ios::app);

    if (file.is_open()) {
        if (filesystem::file_size(filename) == 0) {
            file << "Accelerator,DataType,M,N,K,Avg_Time_ms\n";
        }
        
        string acc_str = (bt == BT::CUDA ? "CUDA" : (bt == BT::SYCL ? "SYCL" : "OPENVINO"));
        file << acc_str << "," << type << "," << M << "," << N << "," << K << "," << avg_ms << "\n";
        
        PRINT("result: " << acc_str << "," << type << "," << M << "," << N << "," << K << "," << avg_ms << "\n");
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
        if (task->type == Type::FLOAT)
            cuda_gemm_32bit(task->A,task->B,task->C, task->M, task->N, task->K);
        if (task->type == Type::HALF)
            cuda_gemm_16bit(task->A,task->B,task->C, task->M, task->N, task->K);
        break;    

    case BT::SYCL:
        if (task->type == Type::FLOAT)
            sycl_gemm_32bit(task->A,task->B,task->C, task->M, task->N, task->K);
        if (task->type == Type::HALF)
            sycl_gemm_16bit(task->A,task->B,task->C, task->M, task->N, task->K);
        break;    

    case BT::OPENVINO:
        if (task->type == Type::FLOAT)
            ov_gemm_32bit(task->A,task->B,task->C, task->M, task->N, task->K);
        if (task->type == Type::HALF)
            ov_gemm_16bit(task->A,task->B,task->C, task->M, task->N, task->K);
        break;    

    //case BT::CPU:
    //    // TODO
    //    break;    

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
        cuda_init(MAX_SIZE,MAX_SIZE,MAX_SIZE); //TODO 
    case BT::SYCL:
        sycl_init(MAX_SIZE,MAX_SIZE,MAX_SIZE); //TODO
        break;    
    case BT::OPENVINO:
        ov_init();
        break;    
    case BT::CPU:
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
        cuda_free();
    case BT::SYCL:
        sycl_free();
        break;    
    case BT::OPENVINO:
        ov_free();
        break;    
    case BT::CPU:
        break;    
    default: 
        cerr << "[SCHEDULER] ERROR Change Status\n";
        exit(EXIT_FAILURE);
    }
}

inline void print_logic(Logic l) {
    string logic;
    switch (l) {
        case Logic::ROUND_ROBIN:
            logic = "ROUND_ROBIN"; break;
        case Logic::CUDA_ONLY:
            logic = "CUDA_ONLY"; break;
        case Logic::STATIC_PARTITIONING:
            logic = "STATIC_PARTITIONING"; break;
        default:
            cerr << "[SCHEDULER] ERROR print_logic \n";
            exit(EXIT_FAILURE);
    }
    PRINT("[SCHEDULER] Started with " << logic << " logic \n");
}

/* return the accellerator string name */
inline string get_acc_string(BT backend_type) {
    string res;
    switch (backend_type) {
        case BT::CUDA:
            res = "CUDA"; break;
        case BT::SYCL:
            res = "SYCL"; break;
        case BT::OPENVINO: 
            res = "OPENVINO"; break;
        default:
            cerr << "[SCHEDULER] ERROR get_acc_string \n";
            exit(EXIT_FAILURE);
    }
    return res;
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
