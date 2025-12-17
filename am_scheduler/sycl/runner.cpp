#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>
#include <iostream>
#include <memory>

#define ZEROCOPY

static std::unique_ptr<sycl::queue> global_q;

void init() {
    try {
        global_q = std::make_unique<sycl::queue>(sycl::gpu_selector_v);
        
        auto dev = global_q->get_device();
        std::cout << "[SYCL] Device: " << dev.get_info<sycl::info::device::name>() << "\n";
        
        global_q->throw_asynchronous();
        
    } catch (sycl::exception const& e) {
        std::cerr << "[SYCL ERROR] Init fallito: " << e.what() << "\n";
    }
}

void sycl_gemm_32bit_p(float *A, float *B, float *C, int M, int N, int K) {
    if (!global_q) {
        init(); 
    }

    sycl::queue& q = *global_q;

    try {
        float *d_A, *d_B, *d_C;

        #ifdef ZEROCOPY
            d_A = sycl::malloc_shared<float>(M * K, q);
            d_B = sycl::malloc_shared<float>(K * N, q);
            d_C = sycl::malloc_shared<float>(M * N, q);
        #else
            // Memoria dedicata GPU (Explicit)
            d_A = sycl::malloc_device<float>(M * K, q);
            d_B = sycl::malloc_device<float>(K * N, q);
            d_C = sycl::malloc_device<float>(M * N, q);
        #endif

        if (!d_A || !d_B || !d_C) {
            std::cerr << "[SYCL ERROR] Allocazione memoria fallita!\n";
            return;
        }

        q.memcpy(d_A, A, M * K * sizeof(float));
        q.memcpy(d_B, B, K * N * sizeof(float));
        q.wait(); 

        oneapi::mkl::transpose trans = oneapi::mkl::transpose::nontrans;
        float alpha = 1.0f;
        float beta = 0.0f;

        // C = alpha * A * B + beta * C
        oneapi::mkl::blas::row_major::gemm(q, trans, trans, M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);
        q.wait(); // Attendere la fine del calcolo

        q.memcpy(C, d_C, M * N * sizeof(float));
        q.wait();

        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_C, q);

    } catch (sycl::exception const& e) {
        std::cerr << "[SYCL ERROR] GEMM fallita: " << e.what() << "\n";
    }
}

extern "C" {
    void sycl_init() {
        init();
    }

    void sycl_gemm_32bit(float *A, float *B, float *C, int M, int N, int K) {
        sycl_gemm_32bit_p(A, B, C, M, N, K);
    }

    void sycl_free() {
    }
}
