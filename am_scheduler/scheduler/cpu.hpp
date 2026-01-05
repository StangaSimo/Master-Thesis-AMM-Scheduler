#ifndef CPUGEM_H
#define CPUGEM_H

#include "tasks.hpp"
#include "config.hpp"
#include <cblas.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <omp.h>

inline void cpu_init() {
    openblas_set_num_threads(N_CORES);
    omp_set_num_threads(N_CORES);
}

inline void cpu_gemm_32bit(void* A, void* B, void* C, int M, int N, int K) {
    float* A_ = (float*) A;
    float* B_ = (float*) B;
    float* C_ = (float*) C;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f,
                A_, K,
                B_, N,
                0.0f,
                C_, N);
}

inline void cpu_gemm_16bit(void* A, void* B, void* C, int M, int N, int K) {
    uint16_t* pA = (uint16_t*)A;
    uint16_t* pB = (uint16_t*)B;
    uint16_t* pC = (uint16_t*)C;

    std::vector<float> A_f32(M*K);
    std::vector<float> B_f32(K*N);
    std::vector<float> C_f32(M*N);

    /* large overhead but we need it, not all cpus have 16 bit */
    #pragma omp parallel for
    for (int i = 0; i < M * K; ++i) {
        A_f32[i] = half_to_float(pA[i]);
    }

    for (int i = 0; i < K * N; ++i) {
        B_f32[i] = half_to_float(pB[i]);
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f,
                A_f32.data(), K,
                B_f32.data(), N,
                0.0f,
                C_f32.data(), N);

    #pragma omp parallel for
    for (int i = 0; i < M * N; ++i) {
        pC[i] = float_to_half(C_f32[i]);
    }
}

#endif

