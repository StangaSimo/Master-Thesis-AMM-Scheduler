//#include <cstdlib>
//#include <sycl/sycl.hpp>
//#include <oneapi/mkl/blas.hpp>
//#include <iostream>
//#include <memory>
//
//#define ZEROCOPY
//
//using namespace std;
//
//static unique_ptr<sycl::queue> global_queue;
//    
//
//float *d_A = nullptr;
//float *d_B = nullptr;
//float *d_C = nullptr;
//
//
//void init_sycl(int max_matrix_size) {
//    try {
//        global_queue = std::make_unique<sycl::queue>(sycl::gpu_selector_v);
//
//        auto dev = global_queue->get_device();
//        //std::cout << "[SYCL] Device: " << dev.get_info<sycl::info::device::name>() << "\n";
//
//        global_queue->throw_asynchronous();
//
//    } catch (sycl::exception const& e) {
//        std::cerr << "[SYCL ERROR] Init fallito: " << e.what() << "\n";
//    }
//
//    //#ifdef ZEROCOPY
//    //    d_A = sycl::malloc_shared<float>(M * K, queue);
//    //    d_B = sycl::malloc_shared<float>(K * N, queue);
//    //    d_C = sycl::malloc_shared<float>(M * N, queue);
//    // #else
//    //    d_A = sycl::malloc_device<float>(M * K, q);
//    //    d_B = sycl::malloc_device<float>(K * N, q);
//    //    d_C = sycl::malloc_device<float>(M * N, q);
//    //#endif
//}
//
//void sycl_gemm_32bit_p(float *A, float *B, float *C, int M, int N, int K) {
//
//    if (!d_A || !d_B || !d_C) {
//        cerr << "[SYCL ERROR] Allocazione memoria fallita!\n";
//        exit(EXIT_FAILURE);
//    }
//
//    sycl::queue& queue = *global_queue;
//
//    d_A = sycl::malloc_shared<float>(M*K, queue);
//    d_B = sycl::malloc_shared<float>(K*N, queue);
//    d_C = sycl::malloc_shared<float>(M*N, queue);
//
//    queue.memcpy(d_A, A, M * K * sizeof(float));
//    queue.memcpy(d_B, B, K * N * sizeof(float));
//
//    float alpha = 1.0f;
//    float beta = 0.0f;
//
//    oneapi::mkl::transpose trans = oneapi::mkl::transpose::nontrans;
//    // C = alpha * A * B + beta * C
//    oneapi::mkl::blas::row_major::gemm(queue, trans, trans, M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);
//
//
//    queue.memcpy(C, d_C, M * N * sizeof(float));
//
//    sycl::free(d_A, *global_queue);
//    sycl::free(d_B, *global_queue);
//    sycl::free(d_C, *global_queue);
//
//    queue.wait();
//
//}
//
//extern "C" {
//    void sycl_init(int max_matrix_size) {
//        init_sycl(max_matrix_size);
//    }
//
//    void sycl_gemm_32bit(float *A, float *B, float *C, int M, int N, int K) {
//        sycl_gemm_32bit_p(A, B, C, M, N, K);
//    }
//
//    void sycl_free() {
//    }
//}


#include <cstdlib>
#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>
#include <iostream>
#include <memory>

#define ZEROCOPY

using namespace std;

static unique_ptr<sycl::queue> global_q;
    
void init() {
    try {
        global_q = std::make_unique<sycl::queue>(sycl::gpu_selector_v);
        
        auto dev = global_q->get_device();
        //std::cout << "[SYCL] Device: " << dev.get_info<sycl::info::device::name>() << "\n";
       
        if (!dev.has(sycl::aspect::fp16)) {
            std::cerr << "[SYCL WARNING] Il device potrebbe non supportare nativamente le operazioni fp16 (half).\n";
            exit(EXIT_FAILURE);
        }

        global_q->throw_asynchronous();
        
    } catch (sycl::exception const& e) {
        std::cerr << "[SYCL ERROR] Init fallito: " << e.what() << "\n";
        exit(EXIT_FAILURE);
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
            cerr << "[SYCL ERROR] Allocazione memoria fallita!\n";
            exit(EXIT_FAILURE);
        }

        q.memcpy(d_A, A, M * K * sizeof(float));
        q.memcpy(d_B, B, K * N * sizeof(float));

        oneapi::mkl::transpose trans = oneapi::mkl::transpose::nontrans;
        float alpha = 1.0f;
        float beta = 0.0f;

        // C = alpha * A * B + beta * C
        oneapi::mkl::blas::row_major::gemm(q, trans, trans, M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);

        q.memcpy(C, d_C, M * N * sizeof(float));
        q.wait();

        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_C, q);

    } catch (sycl::exception const& e) {
        std::cerr << "[SYCL ERROR] GEMM fallita: " << e.what() << "\n";
    }
}

void sycl_gemm_16bit_p(sycl::half *A, sycl::half *B, sycl::half *C, int M, int N, int K) {
    if (!global_q) init();
    sycl::queue& q = *global_q;

    try {
        sycl::half *d_A, *d_B, *d_C;

        #ifdef ZEROCOPY
            d_A = sycl::malloc_shared<sycl::half>(M * K, q);
            d_B = sycl::malloc_shared<sycl::half>(K * N, q);
            d_C = sycl::malloc_shared<sycl::half>(M * N, q);
        #else
            d_A = sycl::malloc_device<sycl::half>(M * K, q);
            d_B = sycl::malloc_device<sycl::half>(K * N, q);
            d_C = sycl::malloc_device<sycl::half>(M * N, q);
        #endif

        if (!d_A || !d_B || !d_C) {
            cerr << "[SYCL ERROR] Allocazione memoria (16bit) fallita!\n";
            exit(EXIT_FAILURE);
        }

        q.memcpy(d_A, A, M * K * sizeof(sycl::half));
        q.memcpy(d_B, B, K * N * sizeof(sycl::half));

        oneapi::mkl::transpose trans = oneapi::mkl::transpose::nontrans;
        
        sycl::half alpha = 1.0f; 
        sycl::half beta = 0.0f;

        oneapi::mkl::blas::row_major::gemm(q, trans, trans, M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);

        q.memcpy(C, d_C, M * N * sizeof(sycl::half));
        q.wait();

        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_C, q);

    } catch (sycl::exception const& e) {
        std::cerr << "[SYCL ERROR] GEMM 16-bit fallita: " << e.what() << "\n";
    }
}

extern "C" {
    void sycl_init(int M, int N, int K) {
        init();
    }

    void sycl_gemm_32bit(void *A, void *B, void *C, int M, int N, int K) {
        sycl_gemm_32bit_p((float*)A, (float*)B, (float*)C, M, N, K);
    }

    void sycl_gemm_16bit(void *A, void *B, void *C, int M, int N, int K) {
        sycl_gemm_16bit_p((sycl::half*)A, (sycl::half*)B, (sycl::half*)C, M, N, K);
    }

    void sycl_free() {
    }
}
