#include "cuda_wrapper.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdint.h>

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
void *d_A[N_STREAM]; 
void *d_B[N_STREAM]; 
void *d_C[N_STREAM];
int i = 0; /* stream id*/

void cuda_gemm_32bit_p(float* A, float* B, float* C, int M, int N, int K) {

        cublasSetStream(handle, streams[i]);

        CHECK_CUDA(cudaMemcpyAsync(d_A[i], A, M * K * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(d_B[i], B, K * N * sizeof(float), cudaMemcpyHostToDevice, streams[i]));

        ///* TODO: check this fix */
        //CHECK_CUDA(cudaMemsetAsync(d_C[i], 0, M * N * sizeof(float), streams[i]));


        float a = 1.0f, b = 0.0f;

        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K, 
                 &a, 
                 d_B[i], CUDA_R_32F, N, 
                 d_A[i], CUDA_R_32F, K, 
                 &b, 
                 d_C[i], CUDA_R_32F, N,
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        CHECK_CUDA(cudaMemcpyAsync(C, d_C[i], M * N * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
        CHECK_CUDA(cudaStreamSynchronize(streams[i])); /* blocking for latency metrics */ 

        i = (i + 1) % N_STREAM; /* change stream for the next call */
}

void cuda_gemm_16bit_p(__half* A, __half* B, __half* C, int M, int N, int K) {
    cublasSetStream(handle, streams[i]);

    CHECK_CUDA(cudaMemcpyAsync(d_A[i], A, M * K * sizeof(__half), cudaMemcpyHostToDevice, streams[i]));
    CHECK_CUDA(cudaMemcpyAsync(d_B[i], B, K * N * sizeof(__half), cudaMemcpyHostToDevice, streams[i]));

    ///* TODO: check this fix */
    //CHECK_CUDA(cudaMemsetAsync(d_C[i], 0, M * N * sizeof(__half), streams[i]));

    float a = 1.0f, b = 0.0f;

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K, 
                 &a, 
                 d_B[i], CUDA_R_16F, N,
                 d_A[i], CUDA_R_16F, K,
                 &b, 
                 d_C[i], CUDA_R_16F, N, 
                 CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    CHECK_CUDA(cudaMemcpyAsync(C, d_C[i], M * N * sizeof(__half), cudaMemcpyDeviceToHost, streams[i]));
    CHECK_CUDA(cudaStreamSynchronize(streams[i]));

    i = (i + 1) % N_STREAM;
}

void cuda_gemm_8bit_p(int8_t* A, int8_t* B, int32_t* C, int M, int N, int K) {
    cublasSetStream(handle, streams[i]);

    CHECK_CUDA(cudaMemcpyAsync(d_A[i], A, M * K * sizeof(int8_t), cudaMemcpyHostToDevice, streams[i]));
    CHECK_CUDA(cudaMemcpyAsync(d_B[i], B, K * N * sizeof(int8_t), cudaMemcpyHostToDevice, streams[i]));

    int32_t a = 1, b = 0; // Scalari interi per INT8

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K, 
                 &a, 
                 d_B[i], CUDA_R_8I, N,
                 d_A[i], CUDA_R_8I, K,
                 &b, 
                 d_C[i], CUDA_R_32I, N, /* Output : INT32 */
                 CUBLAS_COMPUTE_32I,    
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    CHECK_CUDA(cudaMemcpyAsync(C, d_C[i], M * N * sizeof(int32_t), cudaMemcpyDeviceToHost, streams[i]));
    CHECK_CUDA(cudaStreamSynchronize(streams[i]));

    i = (i + 1) % N_STREAM;
}

extern "C" {
    void cuda_init(int M, int N, int K) {
        CHECK_CUBLAS(cublasCreate(&handle));

        /* reset stream counter */
        i = 0;

        for (int j = 0; j < N_STREAM; ++j) {
            CHECK_CUDA(cudaStreamCreate(&streams[j]));
            CHECK_CUDA(cudaMalloc(&d_A[j], M * K * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_B[j], K * N * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_C[j], M * N * sizeof(float)));
        }
    }

    void cuda_gemm_32bit(void* A, void* B, void* C, int M, int N, int K) {
        cuda_gemm_32bit_p((float*)A, (float*)B, (float*)C, M, N, K); 
    }

    void cuda_gemm_16bit(void* A, void* B, void* C, int M, int N, int K) {
        cuda_gemm_16bit_p((__half*)A, (__half*)B, (__half*)C, M, N, K);
    }

    void cuda_gemm_8bit(void* A, void* B, void* C, int M, int N, int K) {
        cuda_gemm_8bit_p((int8_t*)A, (int8_t*)B, (int32_t*)C, M, N, K);
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
