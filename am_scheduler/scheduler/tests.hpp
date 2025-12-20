#ifndef TEST_H
#define TEST_H

/* this is only for testing pourpuse of the dynamic libraries */

#include "sharedbuffer.hpp"
#include "tasks.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

#include <random>
#include <iostream>
#include "scheduler.hpp"

#ifdef ENABLE_OPENVINO
#include "ov_wrapper.h"
#endif

#ifdef ENABLE_CUDA
#include "cuda_wrapper.h"
#endif

#ifdef ENABLE_SYCL
#include "sycl_wrapper.h"
#endif
 
/***************************** Helpers *******************************/
inline void init_32bit(float* A, float* B, float* C, int M, int N, int K) {
    random_device rd;
    mt19937 gen(rd());
    
    uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < M * K; ++i) 
        A[i] = dis(gen);

    for (int i = 0; i < K * N; ++i) 
        B[i] = dis(gen);

    for (int i = 0; i < M * N; ++i) 
        C[i] = 0.0f;
}

inline bool compare_cpu_float(float* A, float* B, float* C, int M, int N, int K) {
    vector<float> C_cpu(M * N, 0.0f);

    /* C = A * B */
    for (int m = 0; m < M; ++m) 
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C_cpu[m * N + n] = sum;
        }

    /* epsilon because the NPU will intruduce slightly differnce errors in the float mul */
    const float epsilon = 0.03f;; /* NPU fault */
    bool match = true;
    float max_diff = 0.0f;

    for (int i = 0; i < M * N; ++i) {
        float diff = fabs(C_cpu[i] - C[i]);
        if (diff > max_diff) max_diff = diff;
        
        /* error */
        if (diff > epsilon) {
            match = false;
        }
    }

    /* we print the last 4 element of the matrix */
    if (!match) {
        cout << "\n[TESTS] ERROR Mismatch! Max Diff: " << max_diff << "\n";
        
        int start_col = (N > 4) ? N - 4 : 0;

        cout << "[TESTS] last 4 elem CPU (Reference): ";
        for (int n = start_col; n < N; ++n) {
            cout << fixed << setprecision(4) << C_cpu[(M - 1) * N + n] << " ";
        }
        cout << "\n";

        cout << "[TESTS] last 4 elem NPU (Input C):   ";
        for (int n = start_col; n < N; ++n) {
            cout << fixed << setprecision(4) << C[(M - 1) * N + n] << " ";
        }
        cout << "\n" << endl;
    } 
    else 
        PRINT("[TESTS] " <<  M << " " << N << " " << K << " PASSED \n");

    return match;
}
/******************************** Test Accellerators **********************************/

//TODO 16 bit implementation and test
inline bool test_openvino(float* A, float* B, float* C, int M, int N, int K) {
    ov_init();  
    ov_gemm_32bit(A,B,C,M,N,K);
    ov_free();  
    return compare_cpu_float(A,B,C,M,N,K);
}

inline bool test_cuda(float* A, float* B, float* C, int M, int N, int K) {
    cuda_init(M,N,K);  
    cuda_gemm_32bit(A,B,C,M,N,K);
    cuda_free(); 
    return compare_cpu_float(A,B,C,M,N,K);
}

inline bool test_sycl(float* A, float* B, float* C, int M, int N, int K) {
    sycl_init(2000); //TODO: change this  
    sycl_gemm_32bit(A,B,C,M,N,K);
    sycl_free();  
    return compare_cpu_float(A,B,C,M,N,K);
}

inline void test_accellerators() {
    int M = 1024;
    int N = 1024;
    int K = 512;
    
    float *A = new float[M*K];
    float *B = new float[K*N];
    float *C = new float[M*N];

    init_32bit(A,B,C,M,N,K);

#ifdef ENABLE_OPENVINO
    if(!test_openvino(A,B,C,M,N,K)){
        cout << "[TESTS] Openvino Error\n";
        exit(EXIT_FAILURE);
    }
#endif
cout << "[TESTS] OpenVino PASSED \n";

#ifdef ENABLE_CUDA
    if(!test_cuda(A,B,C,M,N,K)){
        cout << "[TESTS] CUDA Error\n";
        exit(EXIT_FAILURE);
    }
#endif
cout << "[TESTS] Cuda PASSED \n";

#ifdef ENABLE_SYCL
    if(!test_sycl(A,B,C,M,N,K)){
        cout << "[TESTS] SYCL Error\n";
        exit(EXIT_FAILURE);
    }
#endif
cout << "[TESTS] Sycl PASSED \n";

    delete[] A;
    delete[] B;
    delete[] C;
}
#endif

/***************************** helper test for main ***************************/

inline void test_compare_task(task* array_task, size_t n_task, Type t) {
    if (t == Type::FLOAT){
        for (int i = 0; i < n_task; i++) {

            float *A = (float*)array_task[i].A;
            float *B = (float*)array_task[i].B;
            float *C = (float*)array_task[i].C;

            compare_cpu_float(A, B, C, array_task[i].M, array_task[i].N, array_task[i].K);
        }
    }
}

inline void print_performance_stats(task* tasks, size_t num_tasks) {
    if (tasks == nullptr || num_tasks == 0) {
        std::cout << "[ERROR] print_performance_stats" << std::endl;
        exit(EXIT_FAILURE);
    }

    auto min_start = tasks[0].start_time;
    auto max_end = tasks[0].end_time;
    int M = tasks[0].M;
    int N = tasks[0].N;
    int K = tasks[0].K;
    
    std::chrono::duration<double, std::milli> first_time = tasks[0].end_time - tasks[0].start_time;
    double min_latency_ms = first_time.count();
    double max_latency_ms = first_time.count();

    double total_latency_sum_ms = 0.0;

    for (size_t i=1; i<num_tasks; i++) {

        if (tasks[i].start_time < min_start) 
            min_start = tasks[i].start_time;
        
        if (tasks[i].end_time > max_end) 
            max_end = tasks[i].end_time;
        
        std::chrono::duration<double, std::milli> duration = tasks[i].end_time - tasks[i].start_time;
        double curr_latency_ms = duration.count();
        
        total_latency_sum_ms += curr_latency_ms;

        if (curr_latency_ms < min_latency_ms) 
            min_latency_ms = curr_latency_ms;

        if (curr_latency_ms > max_latency_ms) 
            max_latency_ms = curr_latency_ms;
    }

    std::chrono::duration<double, std::milli> global_span = max_end - min_start;
    double average_latency_ms = total_latency_sum_ms / num_tasks;

    std::cout << "==================================" << std::endl;
    std::cout << "N Matrix: " << num_tasks << std::endl;
    std::cout << "M : " << M  << " N : " << N << " K : " << K << std::endl;
    std::cout << "\nAvg latency:      " << average_latency_ms << " ms" << std::endl;
    std::cout << "Min latency:      " << min_latency_ms << " ms" << std::endl;
    std::cout << "Max latency:      " << max_latency_ms << " ms" << std::endl;
    std::cout << "\nTotal ms:      " << global_span.count() << " ms" << std::endl;
    std::cout << "==================================" << std::endl;
    std::cout << "\n";
}

inline void test_cuda_streaming(int M, int N, int K, size_t num_task, Type type) {
    task* tasks = init_tasks(num_task, M, N, K, type);
    cuda_init(M,N,K); // cambiare, con max matrix size e via.

    for (int i=0; i<num_task; i++) {
        float* A = (float*)tasks[i].A;
        float* B = (float*)tasks[i].B;
        float* C = (float*)tasks[i].C;
        int M = tasks[i].M;
        int N = tasks[i].N;
        int K = tasks[i].K;
        tasks[i].start_time = chrono::high_resolution_clock::now();                

        //TODO: type 
        cuda_gemm_32bit(A, B, C, M, N, K);

        tasks[i].end_time = chrono::high_resolution_clock::now();                
    }

    cuda_free();

    //test_compare_task(tasks, num_task, Type::FLOAT);
    print_performance_stats(tasks, num_task);
    clean_tasks(tasks,num_task,type);
} 
