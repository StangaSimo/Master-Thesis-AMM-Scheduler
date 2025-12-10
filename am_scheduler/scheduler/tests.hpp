#ifndef TEST_H
#define TEST_H

/* this is only for testing pourpuse of the dynamic libraries */

#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

#include <random>
#include <iostream>

#ifdef ENABLE_OPENVINO
#include "ov_wrapper.h"
#endif

#ifdef ENABLE_CUDA
#include "cuda_wrapper.h"
#endif

#ifdef ENABLE_SYCL
#include "sycl_wrapper.h"
#endif
 

inline void init_32bit(float* A, float* B, float* C, int M, int N, int K) {
    random_device rd;
    mt19937 gen(rd());
    
    uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < M * K; ++i) 
        A[i] = dis(gen);

    for (int i = 0; i < K * N; ++i) 
        B[i] = dis(gen);

    for (int i = 0; i < M * N; ++i) {
        C[i] = 0.0f;
    }

    cout << "[TESTS] Init 32 bit Matrix Completed\n";
}

inline bool compare_cpu_32bit(float* A, float* B, float* C, int M, int N, int K) {
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
    const float epsilon = 0.02f;; /* NPU fault */
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
    } else 
        cout << "[TEST] GOOD\n";

    return match;
}

//TODO 16 bit implementation and test
//
inline bool test_openvino(float* A, float* B, float* C, int M, int N, int K) {
    ov_init();  
    ov_gemm_32bit(A,B,C,M,N,K);
    return compare_cpu_32bit(A,B,C,M,N,K);
}

inline bool test_cuda(float* A, float* B, float* C, int M, int N, int K) {
    cuda_init();  
    cuda_gemm_32bit(A,B,C,M,N,K);
    return compare_cpu_32bit(A,B,C,M,N,K);
}

inline bool test_sycl(float* A, float* B, float* C, int M, int N, int K) {
    sycl_init();  
    sycl_gemm_32bit(A,B,C,M,N,K);
    return compare_cpu_32bit(A,B,C,M,N,K);
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

#ifdef ENABLE_CUDA
    if(!test_cuda(A,B,C,M,N,K)){
        cout << "[TESTS] CUDA Error\n";
        exit(EXIT_FAILURE);
    }
#endif

#ifdef ENABLE_SYCL
    if(!test_sycl(A,B,C,M,N,K)){
        cout << "[TESTS] SYCL Error\n";
        exit(EXIT_FAILURE);
    }
#endif
    
    delete[] A;
    delete[] B;
    delete[] C;
}
#endif

