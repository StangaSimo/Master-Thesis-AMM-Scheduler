#ifndef OV_WRAPPER_H
#define OV_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

    void ov_init();

    void ov_gemm_32bit(void* A, void* B, void* C, int M, int N, int K);
    void ov_gemm_16bit(void* A, void* B, void* C, int M, int N, int K);
    void ov_gemm_8bit(void* A, void* B, void* C, int M, int N, int K);

    void ov_free();

#ifdef __cplusplus
}
#endif

#endif
