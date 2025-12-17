#ifndef OV_WRAPPER_H
#define OV_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

    void ov_init();

    void ov_gemm_32bit(float* A, float* B, float* C, int M, int N, int K);

    void ov_free();

#ifdef __cplusplus
}
#endif

#endif
