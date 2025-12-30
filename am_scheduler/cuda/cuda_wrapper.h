#ifndef WRAPPER_H
#define WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

    void cuda_init(int M, int N, int K);
    void cuda_free();
    void cuda_gemm_32bit(void* A, void* B, void* C, int M, int N, int K);
    void cuda_gemm_16bit(void* A, void* B, void* C, int M, int N, int K);
    void cuda_gemm_8bit(void* A, void* B, void* C, int M, int N, int K);

#ifdef __cplusplus
}
#endif

#endif
