#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "cuda_wrapper.h"
#include "sharedbuffer.hpp"

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <atomic>
#include <stdlib.h>
#include <thread>
#include <array>
#include <memory>
#include <unistd.h>


using namespace std;

enum {BUFFER_LENGHT = 512};

enum BT : size_t { /* backend_type */
    CORDINATOR=0,
    CUDA=1, 
    SYCL=2, 
    OPENVINO=3,
    CPU=4, /* CPU WHEN? */
    COUNT=5,
};

/* for simplicity */
using SharedBuffer_T = array<unique_ptr<SharedBuffer>, BT::COUNT>;

inline void print_up_status(BT bt) {
    switch (bt) {
    case BT::CUDA:
        cout << "[SCHEDULER] CUDA Thread up\n";
        break;    
        
    case BT::SYCL:
        cout << "[SCHEDULER] SYCL Thread up\n";
        break;    
    
    case BT::OPENVINO:
        cout << "[SCHEDULER] OPENVINO Thread up\n";
        break;    

    case BT::CPU:
        cout << "[SCHEDULER] CPU Thread up\n";
        break;    

    default: 
        cout << "[SCHEDULER] ERROR Thread Spawn\n";
        exit(EXIT_FAILURE);
        break;
    }
}

inline void handle_task_thread(BT bt, task *task) {
    switch (bt) {
    case BT::CUDA:
        cout << "[SCHEDULER] CUDA gemm done\n";
        cuda_gemm_32bit((float*)task->A,(float*)task->B,(float*)task->C, task->M, task->N, task->K);
        break;    
        
    case BT::SYCL:
        break;    
    
    case BT::OPENVINO:
        break;    

    default: 
        exit(EXIT_FAILURE);
        break;
    }
}

/**********************************  CLASS ************************************/
class AMScheduler {
    private:
        /* atomic for shutting down threads */
        atomic<bool> threads_keep_running;
        /* shared buffers beetween threads */
        SharedBuffer_T shared_buffers;
        /* array of threads, one for each accellerator */
        array<unique_ptr<thread>, BT::COUNT> threads; 

/**********************************  Worker  ***********************************/
        void worker_thread(BT bt, SharedBuffer *buffer) { 
            print_up_status(bt);
            task* task = nullptr;

            while(threads_keep_running) { 
                task = buffer->get();
                if (!task)  { 
                    usleep(10000);
                    continue;
                }
                handle_task_thread(bt, task);
            }
        } 

/**********************************  Coordinator  ***********************************/
        void coordinator_thread(SharedBuffer_T &buffers) { 
            task* task = nullptr;
            SharedBuffer *buffer = buffers[BT::CORDINATOR].get();
            SharedBuffer *cuda_buffer = buffers[BT::CUDA].get();

            while(threads_keep_running) { 
                task = buffer->get();
                if (!task)  { 
                    cout << "buffer vuoto\n";
                    usleep(100000);
                    continue;
                }

                cout << "task presa\n";
                cuda_buffer->put(task);
                cout << "task messa\n";
            } 
        }

/**********************************  Init and Stop  ***********************************/
        void init_threads() {
#ifdef ENABLE_CUDA
            shared_buffers[BT::CUDA] = make_unique<SharedBuffer>(BUFFER_LENGHT);
            threads[BT::CUDA] = make_unique<thread>(&AMScheduler::worker_thread, this, 
                                                    BT::CUDA, 
                                                    shared_buffers[BT::CUDA].get());
#endif

#ifdef ENABLE_OPENVINO
            shared_buffers[BT::OPENVINO] = make_unique<SharedBuffer>(BUFFER_LENGHT);
            threads[BT::OPENVINO] = make_unique<thread>(&AMScheduler::worker_thread, this, 
                                                   BT::OPENVINO,
                                                   shared_buffers[BT::OPENVINO].get());
#endif            

#ifdef ENABLE_SYCL
            shared_buffers[BT::SYCL] = make_unique<SharedBuffer>(BUFFER_LENGHT);
            threads[BT::SYCL] = make_unique<thread>(&AMScheduler::worker_thread, this, 
                                                   BT::SYCL,
                                                   shared_buffers[BT::SYCL].get());
#endif

            /* default CPU thread */
            shared_buffers[BT::CPU] = make_unique<SharedBuffer>(BUFFER_LENGHT);
            threads[BT::CPU] = make_unique<thread>(&AMScheduler::worker_thread, this, 
                                                    BT::CPU,
                                                    shared_buffers[BT::CPU].get());

            /* coordinator */
            shared_buffers[BT::CORDINATOR] = make_unique<SharedBuffer>(BUFFER_LENGHT);
            threads[BT::CORDINATOR] = make_unique<thread>(&AMScheduler::coordinator_thread,this, 
                                        ref(shared_buffers)); /* ref because the thread function
                                                                try to copy all the parameters */
        }

        /* shut down threads */
        void stop_threads() {
            threads_keep_running = false;

            for (auto& thread : threads) 
                if (thread) /* unique pointer are null if not initializated */
                    if(thread->joinable())
                        thread->join();

            cout << "[SCHEDULER] Threads Stopped\n";
        }
       

/**********************************  Public Interface ***********************************/
    public: 
        /* constructur */
        AMScheduler() : threads_keep_running(true) {
            cout << "[SCHEDULER] Starting..\n";
            init_threads();            
            cout << "[SCHEDULER] Constructur Complete.\n";
        }

        /* destructur, stop the threads*/
        ~AMScheduler() {
            cout << "[SCHEDULER] Destructur  Starting..\n";
            stop_threads();
            cout << "[SCHEDULER] Destructur  Complete..\n";
        }

        /* just send the tasks to the coordinator */
        void do_tasks(task* tasks, size_t n) {
            for (int i=0; i<n; i++) {
                shared_buffers[0]->put(&tasks[i]);
            }
        }
};
#endif
