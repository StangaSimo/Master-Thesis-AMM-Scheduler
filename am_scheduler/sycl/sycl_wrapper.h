#ifndef SYCL_WRAPPER_H
#define SYCL_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

    void sycl_init();

    void sycl_gemm_32bit(float *A, float *B, float *C, int M, int N, int K); 

    void sycl_free();

#ifdef __cplusplus
}
#endif

#endif
