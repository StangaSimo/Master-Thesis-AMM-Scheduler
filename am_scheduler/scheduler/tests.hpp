#ifndef TEST_H
#define TEST_H

/* this is only for testing pourpuse of the dynamic libraries */
#include "tasks.hpp"
#include "profiler.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstdint>
#include <stdlib.h>
#include <cstring>

using namespace std;

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

#ifdef ENABLE_OPENBLAS
#include "cpu.hpp"
#endif
 
 
/***************************** Helpers *******************************/

inline bool compare_cpu_32bit(void* A_, void* B_, void* C_, int M, int N, int K) {
    float *A = (float*)A_;
    float *B = (float*)B_;
    float *C = (float*)C_;
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

inline bool compare_cpu_16bit(void* A, void* B, void* C, int M, int N, int K) {
    uint16_t* pA = (uint16_t*)A;
    uint16_t* pB = (uint16_t*)B;
    uint16_t* pC = (uint16_t*)C;

    const float epsilon = 0.5f; 
    bool match = true;
    float max_diff = 0.0f;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += half_to_float(pA[m * K + k]) * half_to_float(pB[k * N + n]);
            }
            
            float gpu_val = half_to_float(pC[m * N + n]);
            float diff = fabs(sum - gpu_val);
            
            if (diff > max_diff) max_diff = diff;

            if (diff > epsilon && diff > (fabs(sum) * 0.05f)) { // 5% tolleranza relativa
                match = false;
            }
        }
    }

    if (!match) cout << "\n[TESTS-16] ERROR Mismatch! Max Diff: " << max_diff << "\n";
    else PRINT("[TESTS-16] PASSED\n");

    return match;
}

inline bool compare_cpu_8bit(void* A, void* B, void* C, int M, int N, int K) {
    int8_t* pA = (int8_t*)A;
    int8_t* pB = (int8_t*)B;
    int32_t* pC = (int32_t*)C; /* output is 32 bit */

    bool match = true;
    int32_t max_diff = 0;

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int32_t sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += (int32_t)pA[m * K + k] * (int32_t)pB[k * N + n];
            }
            
            int32_t diff = abs(sum - pC[m * N + n]);
            if (diff > max_diff) max_diff = diff;

            if (diff != 0) { /* diff 0 for 32 bit output beacause we don't lose precision */
                match = false;
            }
        }
    }

    if (!match) cout << "\n[TESTS-08] ERROR Mismatch! Max Diff: " << max_diff << " (Should be 0)\n";
    else PRINT("[TESTS-08] PASSED\n");

    return match;
}
/******************************** Test Accellerators **********************************/

inline void test_accellerators() {
    int M = 512;
    int N = 512;
    int K = 256;
    
    /* ------------------- 32 BIT TEST ------------------- */
    cout << "\n--- Running 32-bit Tests ---\n";
    float *A = new float[M*K];
    float *B = new float[K*N];
    float *C = new float[M*N];

    init_32bit(A,B,C,M,N,K);

    #ifdef ENABLE_OPENVINO
        ov_init();
        ov_gemm_32bit(A,B,C,M,N,K);
        if(!compare_cpu_32bit(A,B,C,M,N,K)) { cout << "OpenVINO 32bit Fail\n"; exit(1); }
        cout << "--- OpenVino Passed\n";
    #endif

    #ifdef ENABLE_CUDA
        cuda_init(MAX_SIZE, MAX_SIZE, MAX_SIZE);
        cuda_gemm_32bit(A,B,C,M,N,K);
        if(!compare_cpu_32bit(A,B,C,M,N,K)) { cout << "CUDA 32bit Fail\n"; exit(1); }
        cout << "--- Cuda Passed\n";
    #endif

    #ifdef ENABLE_SYCL
        sycl_init(MAX_SIZE, MAX_SIZE, MAX_SIZE);
        sycl_gemm_32bit(A,B,C,M,N,K);
        if(!compare_cpu_32bit(A,B,C,M,N,K)) { cout << "SYCL 32bit Fail\n"; exit(1); }
        cout << "--- Sycl Passed\n";
    #endif

    #ifdef ENABLE_OPENBLAS
        cpu_init();
        cpu_gemm_32bit(A,B,C,M,N,K);
        if(!compare_cpu_32bit(A,B,C,M,N,K)) { cout << "SYCL 32bit Fail\n"; exit(1); }
        cout << "--- OpenBlas Passed\n";
    #endif

    delete[] A; delete[] B; delete[] C;
    cout << "--- PASSED 32-bit Tests ---\n";

    /* ------------------- 16 BIT TEST ------------------- */
    cout << "\n--- Running 16-bit FP16 Tests ---\n";
    uint16_t *A16 = new uint16_t[M*K];
    uint16_t *B16 = new uint16_t[K*N];
    uint16_t *C16 = new uint16_t[M*N];

    init_16bit(A16, B16, C16, M, N, K);

    #ifdef ENABLE_OPENVINO
        ov_gemm_16bit(A16, B16, C16, M, N, K);
        if(!compare_cpu_16bit(A16, B16, C16, M, N, K)) { cout << "OpenVINO 16bit Fail\n"; exit(1); }
        cout << "--- OpenVino Passed\n";
    #endif

    #ifdef ENABLE_CUDA
        cuda_gemm_16bit(A16, B16, C16, M, N, K);
        if(!compare_cpu_16bit(A16, B16, C16, M, N, K)) { cout << "CUDA 16bit Fail\n"; exit(1); }
        cout << "--- Cuda Passed\n";
    #endif

    #ifdef ENABLE_SYCL
        sycl_gemm_16bit(A16, B16, C16, M, N, K);
        if(!compare_cpu_16bit(A16, B16, C16, M, N, K)) { cout << "SYCL 16bit Fail\n"; exit(1); }
        cout << "--- Sycl Passed\n";
    #endif

    #ifdef ENABLE_OPENBLAS
        cpu_gemm_16bit(A16, B16, C16, M, N, K);
        if(!compare_cpu_16bit(A16, B16, C16, M, N, K)) { cout << "CPU 16bit Fail\n"; exit(1); }
        cout << "--- OpenBlas Passed\n";
    #endif

    delete[] A16; delete[] B16; delete[] C16;

    cout << "--- PASSED 32-bit Tests ---\n";

    /* ------------------- 8 BIT TEST ------------------- */
    //cout << "\n--- Running 8-bit INT8 Tests ---\n";
    // Input 1 byte, Output 4 byte (int32)
    //int8_t *A8 = new int8_t[M*K];
    //int8_t *B8 = new int8_t[K*N];
    //int32_t *C8 = new int32_t[M*N]; 

    //init_8bit(A8, B8, C8, M, N, K);

    //#ifdef ENABLE_OPENVINO
    //    ov_gemm_8bit(A8, B8, C8, M, N, K);
    //    if(!compare_cpu_8bit(A8, B8, C8, M, N, K)) { cout << "OpenVINO 8bit Fail\n"; exit(1); }
    //    ov_free();
    //#endif

    //#ifdef ENABLE_CUDA
    //    cuda_gemm_8bit(A8, B8, C8, M, N, K);
    //    if(!compare_cpu_8bit(A8, B8, C8, M, N, K)) { cout << "CUDA 8bit Fail\n"; exit(1); }
    //    cuda_free();
    //#endif

    //#ifdef ENABLE_SYCL
    //    sycl_gemm_8bit(A8, B8, C8, M, N, K);
    //    if(!compare_cpu_8bit(A8, B8, C8, M, N, K)) { cout << "SYCL 8bit Fail\n"; exit(1); }
    //    sycl_free();
    //#endif

    //delete[] A8; delete[] B8; delete[] C8;
    //
    //cout << "\n[ALL TESTS PASSED]\n" << endl;
}
/***************************** helper test for main ***************************/

inline void test_compare_task(task* array_task, size_t n_task) {
    Type t = array_task->type;

    if (t == Type::FLOAT)
        for (int i = 0; i < n_task; i++)
            compare_cpu_32bit(array_task[i].A, array_task[i].B, array_task[i].C, array_task[i].M, array_task[i].N, array_task[i].K);

    if (t == Type::HALF)
        for (int i = 0; i < n_task; i++) 
            compare_cpu_16bit(array_task[i].A, array_task[i].B, array_task[i].C, array_task[i].M, array_task[i].N, array_task[i].K);

    if (t == Type::UINT8)
        for (int i = 0; i < n_task; i++) 
            compare_cpu_8bit(array_task[i].A, array_task[i].B, array_task[i].C, array_task[i].M, array_task[i].N, array_task[i].K);
}

#ifdef ENABLE_CUDA
inline void test_cuda_streaming(int M, int N, int K, size_t num_task, Type type) {
    cout << "\nCuda only \n";
    task* tasks = init_tasks(num_task, M, N, K, type);
    cuda_init(M,N,K);

    for (int i=0; i<num_task; i++) {
        void* A = tasks[i].A;
        void* B = tasks[i].B;
        void* C = tasks[i].C;
        int M = tasks[i].M;
        int N = tasks[i].N;
        int K = tasks[i].K;

        if (type == Type::FLOAT) {
            tasks[i].start_time = chrono::high_resolution_clock::now();                
            cuda_gemm_32bit(A, B, C, M, N, K);
            tasks[i].end_time = chrono::high_resolution_clock::now();                
        }

        if (type == Type::HALF) {
            tasks[i].start_time = chrono::high_resolution_clock::now();                
            cuda_gemm_16bit(A, B, C, M, N, K);
            tasks[i].end_time = chrono::high_resolution_clock::now();                
        }

        if (type == Type::UINT8) {
            tasks[i].start_time = chrono::high_resolution_clock::now();                
            cuda_gemm_8bit(A, B, C, M, N, K);
            tasks[i].end_time = chrono::high_resolution_clock::now();                
        }
    }

    cuda_free();

    //test_compare_task(tasks, num_task);
    print_performance_stats(tasks, num_task);
    clean_tasks(tasks,num_task);
} 
#endif

#endif
