#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "cuda_wrapper.h"
#include "ov_wrapper.h"
#include "sycl_wrapper.h"

#include "sharedbuffer.hpp"
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


using namespace std;

enum {BUFFER_LENGHT = 1024};

enum Logic : size_t { /* Scheduler Logic Type */
    ROUND_ROBIN,
    CUDA_ONLY,
    AUTO_PARTITIONING,
};

enum BT : size_t { /* backend_type */
    CORDINATOR,
    CUDA, 
    SYCL, 
    OPENVINO,
    CPU,
    COUNT,
};

/* for simplicity this is the buffer pointer */
using SharedBuffer_T = array<unique_ptr<SharedBuffer>, BT::COUNT>;

inline void print_logic(Logic l) {
    switch (l) {
    case Logic::ROUND_ROBIN:
        cout << "[SCHEDULER] Started with ROUND_ROBIN logic \n";
        break;
    case Logic::CUDA_ONLY:
        cout << "[SCHEDULER] Started with CUDA_ONLY logic \n";
        break;
    case Logic::AUTO_PARTITIONING:
        cout << "[SCHEDULER] Started with AUTO_PARTITIONING logic \n";
        break;
    default:
        cout << "[SCHEDULER] ERROR logic \n";
        exit(EXIT_FAILURE);
    }
}

inline string get_benchmark_filename(BT bt, int type) {
    string base_path = "../../bin/csv/";
    
    if (!filesystem::exists(base_path)) {
        filesystem::create_directories(base_path);
    }

    string acc_name;
    switch (bt) {
        case BT::CUDA:
            acc_name = "CUDA"; break;
        case BT::SYCL:
            acc_name = "SYCL"; break;
        case BT::OPENVINO: 
            acc_name = "OPENVINO"; break;
        default:
            cout << "[SCHEDULER] ERROR get_benchmark_filename \n";
            exit(EXIT_FAILURE);
    }

    string type_name;
    switch (type) {
        case 1: type_name = "FP32"; break;
        case 2: type_name = "FP16"; break; 
        case 3: type_name = "INT8"; break;
        default:
            cout << "[SCHEDULER] ERROR get_benchmark_filename \n";
            exit(EXIT_FAILURE);
    }

    return base_path + acc_name + "_" + type_name + ".csv";
}

inline void change_status(BT backend_type, bool onoff) {
    switch (backend_type) {
    case BT::CUDA:
        if (onoff) {
            cout << "[SCHEDULER] CUDA Thread Alive\n";
            cuda_init(1024,1024,1024); //TODO 
        } else {
            cout << "[SCHEDULER] CUDA Thread Shutting Down\n";
            cuda_free();
        }
        break;    
        
    case BT::SYCL:
        if (onoff) {
            sycl_init();
            cout << "[SCHEDULER] SYCL Thread Alive\n";
        } else {
            sycl_free();
            cout << "[SCHEDULER] SYCL Thread Shutting Down\n";
        }
        break;    
    
    case BT::OPENVINO:
        if (onoff) {
            ov_init();
            cout << "[SCHEDULER] OPENVINO Thread Alive\n";
        } else {
            ov_free();
            cout << "[SCHEDULER] OPENVINO Thread Shutting Down\n";
        }
        break;    

    case BT::CPU:
        if (onoff)
            cout << "[SCHEDULER] CPU Thread Alive\n";
        else 
            cout << "[SCHEDULER] CPU Thread Shutting Down\n";
        break;    

    default: 
        cout << "[SCHEDULER] ERROR Change Status\n";
        exit(EXIT_FAILURE);
    }
}

inline void benchmark_acc(BT bt, Type type, string filename, int M, int N, int K) {
    int step = 64; 
    
    chrono::high_resolution_clock::time_point start_time;
    chrono::high_resolution_clock::time_point end_time;

    const int N_WARMUP = 3;
    const int N_RUNS = 20;

    double total_ms = 0.0;
    task* task_array = init_tasks(1, M, N, K, type);

    switch (bt) {
    case BT::CUDA:
        if (type == 1)
        //cuda_gemm_32bit((float*)task->A,(float*)task->B,(float*)task->C, task->M, task->N, task->K);
        break;    
        
    case BT::SYCL:
        //sycl_gemm_32bit((float*)task->A,(float*)task->B,(float*)task->C, task->M, task->N, task->K);
        break;    
    
    case BT::OPENVINO:
        //ov_gemm_32bit((float*)task->A,(float*)task->B,(float*)task->C, task->M, task->N, task->K);
        break;    

    default: 
        cout << "[SCHEDULER] ERROR benchmark_acc \n";
        exit(EXIT_FAILURE);
    }

}

inline void benchmark(BT bt, Type type, string filename) {
    int step = 64;
    for (int size = 32; size <= 4096; size += step) {
        benchmark_acc(bt, type, filename, size, size, size);
    }
    cout << "[SCHEDULER] Benchmark ===\n";
    cout << "Results saved in ../../bin/csv/" << filename << "\n";
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

/**********************************  Worker  ***********************************/

        void worker_thread(BT bt, SharedBuffer *buffer) { 
            change_status(bt, true);
            task* task = nullptr;

            while(threads_keep_running) { 

                /* blocking get for max 50ms */
                task = buffer->get();

                /* if there isn't a task we check if we have to shutdown */
                if (!task) {continue;}

                handle_task(bt, task);
            }
            change_status(bt, false);
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
                case Logic::AUTO_PARTITIONING: 
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

#ifdef ENABLE_CUDA
            bts.push_back(BT::CUDA);
#endif
#ifdef ENABLE_OPENVINO
            bts.push_back(BT::OPENVINO);
#endif
#ifdef ENABLE_SYCL
            bts.push_back(BT::SYCL);
#endif
            for (BT i : bts) {
                shared_buffers[i] = make_unique<SharedBuffer>(BUFFER_LENGHT);
                threads[i] = make_unique<thread>(&AMScheduler::worker_thread, this, 
                                                    i, shared_buffers[i].get());
            }

            if (bts.size() == 0) {cout << "[SCHEDULER] ATTENTION, only CPU up\n";}

            //bts.push_back(BT::CPU); TODO remove this when implementing the cpus 
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

            cout << "[SCHEDULER] Threads stopped.\n";
        }

        /**********************************  Init Benchmarks ***********************************/
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

        /**********************************  Public Interface ***********************************/

    public: 
        /* constructur */
        AMScheduler(Logic l) : threads_keep_running(true), strategy(l) {
            print_logic(l);
            init_threads();            
            cout << "[SCHEDULER] Threads started.\n";
            init_benchmarks();            
            cout << "[SCHEDULER] Benchmarks done.\n";
        }

        /* destructur, stop the threads*/
        ~AMScheduler() {
            stop_threads();
            cout << "[SCHEDULER] Destructur complete..\n";
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

#endif
