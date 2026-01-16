#ifndef SYCL_WRAPPER_H
#define SYCL_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

    void sycl_init();

    void sycl_gemm_32bit(void *A, void *B, void *C, int M, int N, int K); 
    void sycl_gemm_16bit(void *A, void *B, void *C, int M, int N, int K); 

    void sycl_free();

#ifdef __cplusplus
}
#endif

#endif
