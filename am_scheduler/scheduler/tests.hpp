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
#include <cstdint>
#include <stdlib.h>
#include <cstring>

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

inline float half_to_float(uint16_t h) {
    uint32_t s = (h >> 15) & 0x00000001;
    uint32_t e = (h >> 10) & 0x0000001f;
    uint32_t m = h & 0x000003ff;
    
    if (e == 0) {
        if (m == 0) {
            uint32_t res = s << 31;
            float f; memcpy(&f, &res, 4); return f;
        } else {
            // Per il range -1..1 dei test, non dovremmo colpire troppo i denorm
            while (!(m & 0x00000400)) { m <<= 1; e -= 1; }
            e += 1; m &= ~0x00000400;
        }
    } else if (e == 31) {
        if (m == 0) { // Inf
             uint32_t res = (s << 31) | 0x7f800000;
             float f; memcpy(&f, &res, 4); return f;
        } else { // NaN
             uint32_t res = (s << 31) | 0x7f800000 | (m << 13);
             float f; memcpy(&f, &res, 4); return f;
        }
    }
    
    e = e + (127 - 15);
    m = m << 13;
    uint32_t res = (s << 31) | (e << 23) | m;
    float f; memcpy(&f, &res, 4); return f;
}

inline uint16_t float_to_half(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t s = (x >> 31) & 0x00000001;
    uint32_t e = (x >> 23) & 0x000000ff;
    uint32_t m = x & 0x007fffff;

    uint32_t h_e, h_m;

    if (e >= 143) { // Overflow -> Inf
        h_e = 31; h_m = 0;
    } else if (e <= 102) { // Underflow -> 0 (semplificato)
        h_e = 0; h_m = 0;
    } else {
        h_e = e - (127 - 15);
        h_m = m >> 13;
    }
    return (s << 15) | (h_e << 10) | h_m;
}

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

inline void init_16bit(void* A, void* B, void* C, int M, int N, int K) {
    uint16_t* pA = (uint16_t*)A;
    uint16_t* pB = (uint16_t*)B;
    uint16_t* pC = (uint16_t*)C;

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < M * K; ++i) pA[i] = float_to_half(dis(gen));
    for (int i = 0; i < K * N; ++i) pB[i] = float_to_half(dis(gen));
    memset(pC, 0, M * N * sizeof(uint16_t));
}

inline void init_8bit(void* A, void* B, void* C, int M, int N, int K) {
    int8_t* pA = (int8_t*)A;
    int8_t* pB = (int8_t*)B;
    int32_t* pC = (int32_t*)C;

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(-3, 3); 

    for (int i = 0; i < M * K; ++i) pA[i] = (int8_t)dis(gen);
    for (int i = 0; i < K * N; ++i) pB[i] = (int8_t)dis(gen);
    memset(pC, 0, M * N * sizeof(int32_t));
}

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

//TODO 16 bit implementation and test
inline bool test_openvino(float* A, float* B, float* C, int M, int N, int K) {
    ov_init();  
    ov_gemm_32bit(A,B,C,M,N,K);
    ov_free();  
    return compare_cpu_32bit(A,B,C,M,N,K);
}

inline bool test_cuda(float* A, float* B, float* C, int M, int N, int K) {
    cuda_init(M,N,K);  
    cuda_gemm_32bit(A,B,C,M,N,K);
    cuda_free(); 
    return compare_cpu_32bit(A,B,C,M,N,K);
}

inline bool test_sycl(float* A, float* B, float* C, int M, int N, int K) {
    sycl_init(M,N,K);
    sycl_gemm_32bit(A,B,C,M,N,K);
    sycl_free();  
    return compare_cpu_32bit(A,B,C,M,N,K);
}

inline void test_accellerators() {
    int M = 512;
    int N = 512;
    int K = 256;
    
    /* ------------------- 32 BIT TEST ------------------- */
    cout << "\n--- Running 32-bit FP32 Tests ---\n";
    float *A = new float[M*K];
    float *B = new float[K*N];
    float *C = new float[M*N];

    init_32bit(A,B,C,M,N,K);

    #ifdef ENABLE_OPENVINO
        ov_init();
        ov_gemm_32bit(A,B,C,M,N,K);
        if(!compare_cpu_32bit(A,B,C,M,N,K)) { cout << "OpenVINO 32bit Fail\n"; exit(1); }
    #endif

    #ifdef ENABLE_CUDA
        cuda_init(MAX_SIZE, MAX_SIZE, MAX_SIZE);
        cuda_gemm_32bit(A,B,C,M,N,K);
        if(!compare_cpu_32bit(A,B,C,M,N,K)) { cout << "CUDA 32bit Fail\n"; exit(1); }
    #endif

    #ifdef ENABLE_SYCL
        sycl_init(MAX_SIZE, MAX_SIZE, MAX_SIZE);
        sycl_gemm_32bit(A,B,C,M,N,K);
        if(!compare_cpu_32bit(A,B,C,M,N,K)) { cout << "SYCL 32bit Fail\n"; exit(1); }
    #endif

    delete[] A; delete[] B; delete[] C;


    /* ------------------- 16 BIT TEST ------------------- */
    cout << "\n--- Running 16-bit FP16 Tests ---\n";
    // Usiamo uint16_t per allocare la dimensione giusta (2 bytes)
    uint16_t *A16 = new uint16_t[M*K];
    uint16_t *B16 = new uint16_t[K*N];
    uint16_t *C16 = new uint16_t[M*N];

    init_16bit(A16, B16, C16, M, N, K);

    #ifdef ENABLE_OPENVINO
        ov_gemm_16bit(A16, B16, C16, M, N, K);
        if(!compare_cpu_16bit(A16, B16, C16, M, N, K)) { cout << "OpenVINO 16bit Fail\n"; exit(1); }
    #endif

    #ifdef ENABLE_CUDA
        cuda_gemm_16bit(A16, B16, C16, M, N, K);
        if(!compare_cpu_16bit(A16, B16, C16, M, N, K)) { cout << "CUDA 16bit Fail\n"; exit(1); }
    #endif

    #ifdef ENABLE_SYCL
        sycl_gemm_16bit(A16, B16, C16, M, N, K);
        if(!compare_cpu_16bit(A16, B16, C16, M, N, K)) { cout << "SYCL 16bit Fail\n"; exit(1); }
    #endif

    delete[] A16; delete[] B16; delete[] C16;


    /* ------------------- 8 BIT TEST ------------------- */
    //cout << "\n--- Running 8-bit INT8 Tests ---\n";
    // Input 1 byte, Output 4 byte (int32)
    int8_t *A8 = new int8_t[M*K];
    int8_t *B8 = new int8_t[K*N];
    int32_t *C8 = new int32_t[M*N]; 

    init_8bit(A8, B8, C8, M, N, K);

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

    delete[] A8; delete[] B8; delete[] C8;
    
    cout << "\n[ALL TESTS PASSED]\n" << endl;
}
/***************************** helper test for main ***************************/

inline void test_compare_task(task* array_task, size_t n_task, Type t) {
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

    //test_compare_task(tasks, num_task, Type::FLOAT);
    print_performance_stats(tasks, num_task);
    clean_tasks(tasks,num_task,type);
} 

#endif
