#ifndef WRAPPER_H
#define WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

    void cuda_init(int M, int N, int K);
    void cuda_free();
    void cuda_gemm_32bit(float* A, float* B, float* C, int M, int N, int K);
    void benchmark_cuda_gemm_32bit(float* A, float* B, float* C, int M, int N, int K);


#ifdef __cplusplus
}
#endif

#endif
