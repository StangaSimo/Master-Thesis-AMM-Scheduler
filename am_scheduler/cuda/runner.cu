#include "cuda_wrapper.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(func) {				    \
    cudaError_t e = (func);			        \
    if(e != cudaSuccess)			        \
        printf ("%s %d CUDA ERROR: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e)); \
}

#define CHECK_CUBLAS(func) {                      \
    cublasStatus_t e = (func);                    \
    if(e != CUBLAS_STATUS_SUCCESS)                \
        printf ("%s %d CUBLAS ERROR: ", __FILE__, __LINE__); \
}

extern "C" {
    void gpu_init() {
        printf("[CUDA-LIB] Inizializzazione device...\n");
    }

    void run_cuda_32bit(float* A, float* B, float* C, int M, int N, int K) {

        float *d_A, *d_B, *d_C;

        CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaDeviceSynchronize());

        float alpha = 1.0f, beta = 0.0f;
        cublasHandle_t handle;  
        CHECK_CUBLAS(cublasCreate(&handle));

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    d_B, N, d_A, K, &beta, d_C, N);

        CHECK_CUBLAS(cublasDestroy(handle));
 
        CHECK_CUDA(cudaDeviceSynchronize()); //TODO: maybe took them off
                                           
        CHECK_CUDA(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

        CHECK_CUDA(cudaFree(d_A));
        CHECK_CUDA(cudaFree(d_B));
        CHECK_CUDA(cudaFree(d_C));
    }
}
