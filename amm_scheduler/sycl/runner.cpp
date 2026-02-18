#include <cstdlib>
#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>
#include <iostream>
#include <memory>

#define N_STREAMS 2

using namespace std;

struct SYCLStream {
    sycl::queue q;
    float *d_A_f, *d_B_f, *d_C_f;     
    sycl::half *d_A_h, *d_B_h, *d_C_h;
    
    SYCLStream(size_t max_elements) : q(sycl::gpu_selector_v, sycl::property::queue::in_order()) {
        d_A_f = sycl::malloc_device<float>(max_elements, q);
        d_B_f = sycl::malloc_device<float>(max_elements, q);
        d_C_f = sycl::malloc_device<float>(max_elements, q);
        
        d_A_h = sycl::malloc_device<sycl::half>(max_elements, q);
        d_B_h = sycl::malloc_device<sycl::half>(max_elements, q);
        d_C_h = sycl::malloc_device<sycl::half>(max_elements, q);

        if (q.get_device().get_info<sycl::info::device::name>()  != "Intel(R) Arc(TM) Graphics") {
            std::cout << "Intel GPU NOT PRESENT\n";
            exit(EXIT_FAILURE);
        }
    }

    ~SYCLStream() {
        sycl::free(d_A_f, q); sycl::free(d_B_f, q); sycl::free(d_C_f, q);
        sycl::free(d_A_h, q); sycl::free(d_B_h, q); sycl::free(d_C_h, q);
    }
};

static std::vector<std::unique_ptr<SYCLStream>> streams;
static int current_stream = 0;
static size_t MAX = (size_t)4096 * 4096;

void init() {
    try {

        /* stream persistenti */
        for(int i=0; i<N_STREAMS; i++) {
            streams.push_back(std::make_unique<SYCLStream>(MAX));
        }

    } catch (sycl::exception const& e) {
        std::cerr << "[SYCL ERROR] Init: " << e.what() << "\n";
        exit(1);
    }
}

void sycl_gemm_32bit_p(float *A, float *B, float *C, int M, int N, int K) {
    SYCLStream& s = *streams[current_stream];

    s.q.memcpy(s.d_A_f, A, M * K * sizeof(float));
    s.q.memcpy(s.d_B_f, B, K * N * sizeof(float));

    float alpha = 1.0f, beta = 0.0f;
    oneapi::mkl::blas::row_major::gemm(s.q, oneapi::mkl::transpose::nontrans, 
            oneapi::mkl::transpose::nontrans, 
            M, N, K, alpha, s.d_A_f, K, s.d_B_f, N, beta, s.d_C_f, N);

    s.q.memcpy(C, s.d_C_f, M * N * sizeof(float));
    s.q.wait(); 

    current_stream = (current_stream + 1) % N_STREAMS;
}

void sycl_gemm_16bit_p(sycl::half *A, sycl::half *B, 
                                sycl::half *C, int M, int N, int K) {
    SYCLStream& s = *streams[current_stream];

    s.q.memcpy(s.d_A_h, A, M * K * sizeof(sycl::half));
    s.q.memcpy(s.d_B_h, B, K * N * sizeof(sycl::half));

    sycl::half alpha = 1.0f, beta = 0.0f;
    oneapi::mkl::blas::row_major::gemm(s.q, oneapi::mkl::transpose::nontrans, 
            oneapi::mkl::transpose::nontrans, 
            M, N, K, alpha, s.d_A_h, K, s.d_B_h, N, beta, s.d_C_h, N);

    s.q.memcpy(C, s.d_C_h, M * N * sizeof(sycl::half));
    s.q.wait();

    current_stream = (current_stream + 1) % N_STREAMS;
}

extern "C" {
    void sycl_init() {
        init();
    }

    void sycl_gemm_32bit(void *A, void *B, void *C, int M, int N, int K) {
        sycl_gemm_32bit_p((float*)A, (float*)B, (float*)C, M, N, K);
    }

    void sycl_gemm_16bit(void *A, void *B, void *C, int M, int N, int K) {
        sycl_gemm_16bit_p((sycl::half*)A, (sycl::half*)B, (sycl::half*)C, M, N, K);
    }

    void sycl_free() {
        streams.clear();
    }
}
