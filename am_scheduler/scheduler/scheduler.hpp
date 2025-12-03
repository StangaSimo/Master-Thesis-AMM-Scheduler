#ifndef SCHEDULER_H
#define SCHEDULER_H

#include <iostream>
#include <atomic>
#include <stdlib.h>
#include <thread>
#include <array>
#include <memory>
#include <unistd.h>

enum class BackendType : int { 
    CUDA = 0, 
    SYCL = 1, 
    OPENVINO = 2,
    CPU = 3, /* CPU WHEN? */

    COUNT /* 4 */
};

using namespace std;

class AMScheduler {
    private:
        /* atomic for shutting down threads */
        atomic<bool> threads_keep_running;

        /* array of threads, one for each accellerator */
        array<unique_ptr<thread>, (int) BackendType::COUNT> threads; 

        /* runner for each thread */
        void Gemm_Thread(BackendType backend_type) { 
            while(threads_keep_running) { 
                cout << "Thread Vivo\n"; 
                sleep(1);
            } 
        }

        /* init the threads */
        void init_threads() {
#ifdef ENABLE_CUDA
            cout << "[SCHEDULER] CUDA Thread up\n";
            threads[(int)BackendType::CUDA] = make_unique<thread>(&AMScheduler::Gemm_Thread, 
                                                                            this, BackendType::CUDA);
#endif

#ifdef ENABLE_OPENVINO
            cout << "[SCHEDULER] OPENVINO Thread up\n";
            threads[(int)BackendType::OPENVINO] = make_unique<thread>(&AMScheduler::Gemm_Thread, 
                                                                            this, BackendType::OPENVINO);
#endif            

#ifdef ENABLE_SYCL
            cout << "[SCHEDULER] SYCL Thread up\n";
            threads[(int)BackendType::SYCL] = make_unique<thread>(&AMScheduler::Gemm_Thread, 
                                                                            this, BackendType::SYCL);
#endif
            cout << "[SCHEDULER] CPU Thread up\n";
            threads[(int)BackendType::CPU] = make_unique<thread>(&AMScheduler::Gemm_Thread, 
                                                                            this, BackendType::CPU);
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
};

#endif
