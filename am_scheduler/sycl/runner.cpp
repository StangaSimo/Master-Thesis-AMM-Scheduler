#include <sycl/sycl.hpp>
#include <iostream>
#include "sycl_wrapper.h"

/* Zero-Copy for the buffers */
#define ZEROCOPY
using data_t = sycl::half; 

void init() {
    sycl::queue q(sycl::gpu_selector_v);
    auto dev = q.get_device();
    std::cout << "[SYCL] Device: " << dev.get_info<sycl::info::device::name>() << "\n";
}

void gemm_32bit(float *A, float *B, float *C, int M, int N, int K){
     
}

extern "C" {
    void sycl_init() {
        init();
    }

    void sycl_gemm_32bit(float *A, float *B, float *C, int M, int N, int K) {
        gemm_32bit(A,B,C,M,N,K);
    }
}
