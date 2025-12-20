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
#define N_STREAM 2

cublasHandle_t handle;  
cudaStream_t streams[N_STREAM];
float *d_A[N_STREAM]; 
float *d_B[N_STREAM]; 
float *d_C[N_STREAM];
int i = 0; /* stream id*/

void cuda_gemm_32bit_p(float* A, float* B, float* C, int M, int N, int K) {

        cublasSetStream(handle, streams[i]);

        CHECK_CUDA(cudaMemcpyAsync(d_A[i], A, M * K * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(d_B[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice, streams[i]));

        float a = 1.0f, b = 0.0f;
        // 2. Impostiamo lo stream su cuBLAS


        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &a,
                d_B[i], N, d_A[i], K, &b, d_C[i], N);

        CHECK_CUDA(cudaMemcpyAsync(C, d_C[i], M * N * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
        CHECK_CUDA(cudaStreamSynchronize(streams[i])); /* blocking for latency metrics */ 

        i = (i + 1) % N_STREAM; /* change stream for the next call */
}

extern "C" {
    void cuda_init(int M, int N, int K) {
        CHECK_CUBLAS(cublasCreate(&handle));

        for (int j = 0; j < N_STREAM; ++j) {
            CHECK_CUDA(cudaStreamCreate(&streams[j]));
            CHECK_CUDA(cudaMalloc(&d_A[j], M * K * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_B[j], K * N * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_C[j], M * N * sizeof(float)));
        }
    }

    void cuda_gemm_32bit(float* A, float* B, float* C, int M, int N, int K) {
        cuda_gemm_32bit_p(A, B, C, M, N, K); 
    }

    void cuda_free() {
        for (int j = 0; j < N_STREAM; ++j) {
            CHECK_CUDA(cudaFree(d_A[j]));
            CHECK_CUDA(cudaFree(d_B[j]));
            CHECK_CUDA(cudaFree(d_C[j]));
            CHECK_CUDA(cudaStreamDestroy(streams[j]));
        }
        CHECK_CUBLAS(cublasDestroy(handle));
    }
}
