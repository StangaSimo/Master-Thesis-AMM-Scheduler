#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "cuda_wrapper.h"
#include "ov_wrapper.h"
#include "sycl_wrapper.h"

#include "sharedbuffer.hpp"
#include "performancemap.hpp"
#include "tasks.hpp"

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


using namespace std;

enum {BUFFER_LENGHT = 1024};

enum {STEP_SIZE = 512};

enum Logic : size_t { /* Scheduler Logic Type */
    ROUND_ROBIN,
    CUDA_ONLY,
    STATIC_PARTITIONING,
};

enum BT : size_t { /* backend_type */
    CORDINATOR,
    CUDA, 
    SYCL, 
    OPENVINO,
    CPU,
    COUNT,
};

/******* Prototypes ********/
inline void handle_task(BT backend_type, task *task);
inline string get_benchmark_filename(BT bt, int type); 
inline void init_acc(BT backend_type); 
inline void free_acc(BT backend_type); 
inline void benchmark_acc(BT bt, Type type, string filename, int M, int N, int K);
inline void benchmark(BT bt, Type type, string filename);
inline void handle_task(BT backend_type, task *task);
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

        /* Perfomance Map for each accellerator*/
        array<unique_ptr<PerformanceMap>, BT::COUNT> bts_map; 

/**********************************  Worker  ***********************************/

        void worker_thread(BT bt, SharedBuffer *buffer) { 
            task* task = nullptr;

            while(threads_keep_running) { 

                /* blocking get for max 50ms */
                task = buffer->get();
                /* if there isn't a task we check if we have to shutdown */
                if (!task) {continue;}

                handle_task(bt, task);
            }
        } 

/**********************************  Coordinator  ***********************************/

        /* if there is no task, we check if we have to shutdown */
        void coordinator_thread(SharedBuffer_T &buffers) { 
            int i=0;
            int bts_len = bts.size();
            task* task = nullptr;
            SharedBuffer *buffer = buffers[BT::CORDINATOR].get();
            
            switch (strategy) {
                /* only cuda cards */
                case Logic::CUDA_ONLY:
                    while(threads_keep_running) { 
                        task = buffer->get();
                        if (!task) {continue;} 
                        buffers[BT::CUDA]->put(task);
                    } 
                    break;

                /* basic logic, if there is one that don't do nothing just send them request */
                case Logic::ROUND_ROBIN: 
                    while(threads_keep_running) { 
                        task = buffer->get();
                        if (!task) {continue;} 

                        buffers[bts[i]]->put(task);

                        i = (i+1) % bts_len;
                    } 
                    break;

                /* batch style partitioning */ 
                case Logic::STATIC_PARTITIONING: 
                    while(threads_keep_running) { 
                        task = buffer->get();
                        if (!task) {continue;} 
                    } 
                    break;
                default:
                    printf("[SCHEDULER] Error in chosing the logic.\n");
                    exit(EXIT_FAILURE);
            }
        }

/**********************************  Init and Stop  ***********************************/

        void init_threads() {

#ifdef ENABLE_OPENVINO
            init_acc(BT::OPENVINO); /* have to init here and not in threads */
            bts.push_back(BT::OPENVINO);
#endif
#ifdef ENABLE_CUDA
            init_acc(BT::CUDA);
            bts.push_back(BT::CUDA);
#endif
#ifdef ENABLE_SYCL
            init_acc(BT::SYCL);
            bts.push_back(BT::SYCL);
#endif
            for (BT i : bts) {
                shared_buffers[i] = make_unique<SharedBuffer>(BUFFER_LENGHT);
                threads[i] = make_unique<thread>(&AMScheduler::worker_thread, this, 
                                                    i, shared_buffers[i].get());
            }

            if (bts.size() == 0) {cout << "[SCHEDULER] ATTENTION, only CPU up\n";}

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
            for (auto i : bts) {
                free_acc(i);
            }

            cout << "[SCHEDULER] Threads stopped.\n";
        }

        /**********************************  Init Benchmarks and Map ***********************************/
        /* upload the banchmark files and generate them if not presents */
        void init_benchmarks() {
            for (BT i : bts) {
                string filename; 
                //TODO: maybe more that 1 type 
                
                filename = get_benchmark_filename(i, Type::FLOAT);
                if (filesystem::exists(filename)){
                     cout << "[SCHEDULER] File: " << filename <<  " opened.\n"; 
                } else {
                     cout << "[SCHEDULER] File: " << filename <<  " not present\n"; 
                     benchmark(i, Type::FLOAT ,filename);
                     cout << "[SCHEDULER] Benchmark file created\n"; 
                }
            }
        }

        /* create the map for each accellerator */
        void init_maps() {
            for (BT i : bts) {
                string filename; 
                filename = get_benchmark_filename(i, Type::FLOAT);
                if (filesystem::exists(filename)){

                    bts_map[i] = make_unique<PerformanceMap>(STEP_SIZE,filename);

                    double res = bts_map[i]->query(512,2460,512);
                    string s = get_acc_string(i);
                    cout << "\n\n" << s << " prova regression: " << res <<  "\n\n"; 
                } else {

                     cerr << "[SCHEDULER] File: " << filename <<  " not found for map init\n"; 
                     exit(EXIT_FAILURE);
                }
            }
        }

        /**********************************  Public Interface ***********************************/

    public: 
        /* constructur init threads, benchark files, regression models */
        AMScheduler(Logic logic) : threads_keep_running(true), strategy(logic) {
            print_logic(logic);
            init_threads();            
            cout << "[SCHEDULER] Threads started.\n";
            init_benchmarks();            
            cout << "[SCHEDULER] Benchmarks done.\n";
            init_maps();
            cout << "[SCHEDULER] Regression done.\n";
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
            cout << "[SCHEDULER] Task dispached\n";
        }

        /* check if all the buffers are empty */
        void wait() {
            cout << "[SCHEDULER] Waiting.. \n";
            /* wait coordinator */
            while (!shared_buffers[0]->is_empty()){
                usleep(10000);
            }
            cout << "[SCHEDULER] Coordinator empty.. \n";

            /* wait other threads */
            for (int i=1; i<BT::COUNT; i++) {
                if (shared_buffers[i] != nullptr) {
                    while (!shared_buffers[i]->is_empty()){
                        usleep(10000);
                    }
                }
            }

            cout << "[SCHEDULER] All empty.. \n";
            sleep(2); /* wait for unfinished matrix */
        }
};

/**********************************  Helper Functions ***********************************/

/* benchmark the accellerator and write in the csv*/
inline void benchmark_acc(BT bt, Type type, string filename, int M, int N, int K) {
    chrono::high_resolution_clock::time_point start_time;
    chrono::high_resolution_clock::time_point end_time;
    chrono::duration<double, std::milli> duration;

    const int N_WARMUP = 3;
    const int N_RUNS = 10;

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
        
        cout << "result: " << acc_str << "," << type << "," << M << "," << N << "," << K << "," << avg_ms << "\n";
        file.close();
    } else {
        cerr << "[SCEDULER] ERROR Cannot open file: " << filename << "\n";
    }
}

/* call benchmark_acc */
inline void benchmark(BT bt, Type type, string filename) {
    vector<int> dims;
    for (int d = STEP_SIZE; d <= STEP_SIZE*6; d += STEP_SIZE) {
        dims.push_back(d);
    }

    for (int m : dims) {
        for (int n : dims) {
            for (int k : dims) {
                //if (bt == BT::OPENVINO && m >= 3500) {break;} /* NPU doesn't like more than this */  
                //if (bt == BT::OPENVINO && n >= 3500) {break;} 
                //if (bt == BT::OPENVINO && k >= 3500) {break;}
                benchmark_acc(bt, type, filename, m, n, k);
            }
        }
    }

    cout << "[SCHEDULER] Benchmark ===\n";
    cout << "Results saved in bin/csv/" << filename << "\n";
}

/* task handler for the workers, choose the right backend_type */
inline void handle_task(BT backend_type, task *task) {
    switch (backend_type) {
    case BT::CUDA:
        cuda_gemm_32bit((float*)task->A,(float*)task->B,(float*)task->C, task->M, task->N, task->K);
        break;    
        
    case BT::SYCL:
        sycl_gemm_32bit((float*)task->A,(float*)task->B,(float*)task->C, task->M, task->N, task->K);
        break;    
    
    case BT::OPENVINO:
        ov_gemm_32bit((float*)task->A,(float*)task->B,(float*)task->C, task->M, task->N, task->K);
        break;    

    //case BT::CPU:
    //    // TODO
    //    break;    

    default: 
        cout << "[SCHEDULER] ERROR handle task\n";
        exit(EXIT_FAILURE);
    }

    task->end_time = chrono::high_resolution_clock::now();
}

/* call the init func of the accellerator */
inline void init_acc(BT backend_type) {
    switch (backend_type) {
    case BT::CUDA:
        cuda_init(4092,4092,4092); //TODO 
    case BT::SYCL:
        sycl_init();
        break;    
    case BT::OPENVINO:
        ov_init();
        break;    
    case BT::CPU:
        break;    
    default: 
        cout << "[SCHEDULER] ERROR Change Status\n";
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
        cout << "[SCHEDULER] ERROR Change Status\n";
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
            logic = "AUTO_PARTITIONING"; break;
        default:
            cout << "[SCHEDULER] ERROR print_logic \n";
            exit(EXIT_FAILURE);
    }
    cout << "[SCHEDULER] Started with " << logic << " logic \n";
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
            cout << "[SCHEDULER] ERROR get_acc_string \n";
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
            cout << "[SCHEDULER] ERROR type get_benchmark_filename \n";
            exit(EXIT_FAILURE);
    }

    return base_path + acc_name + "_" + type_name + ".csv";
}


#endif
