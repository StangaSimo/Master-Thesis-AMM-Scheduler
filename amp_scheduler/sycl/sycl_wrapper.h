#ifndef SYCL_WRAPPER_H   // <--- Se qui c'è scritto CUDA_WRAPPER_H o WRAPPER_H...
#define SYCL_WRAPPER_H   // <--- ...e anche qui...

#ifdef __cplusplus
extern "C" {
#endif

    void sycl_init();
    void sycl_process(int N);

#ifdef __cplusplus
}
#endif

#endif
