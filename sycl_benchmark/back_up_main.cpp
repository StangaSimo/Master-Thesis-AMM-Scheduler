#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>
#include <iostream>
#include <vector>
#include <chrono>

constexpr size_t MATRIX_SIZE = 1024;
constexpr size_t M = MATRIX_SIZE;
constexpr size_t N = MATRIX_SIZE;
constexpr size_t K = MATRIX_SIZE/2;

namespace mkl = oneapi::mkl;

int main() {
    sycl::queue q(sycl::default_selector_v);
    std::cout << "Esecuzione sul dispositivo: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Usando oneMKL BLAS (gemm)\n";

    // 2. PREPARAZIONE DATI
    std::vector<float> a_host(M * K);
    std::vector<float> b_host(K * N);
    std::vector<float> c_host(M * N);

    for (size_t i = 0; i < M * K; ++i)
        a_host[i] = 1.0f;
    for (size_t i = 0; i < K * N; ++i)
        b_host[i] = 2.0f;
    for (size_t i = 0; i < M * N; ++i)
        c_host[i] = 0.0f;

    try
    {
        sycl::buffer<float, 1> a_buf(a_host.data(), sycl::range<1>(M * K));
        sycl::buffer<float, 1> b_buf(b_host.data(), sycl::range<1>(K * N));
        sycl::buffer<float, 1> c_buf(c_host.data(), sycl::range<1>(M * N));
           
        mkl::transpose transA = mkl::transpose::nontrans;
        mkl::transpose transB = mkl::transpose::nontrans;

        // C = 1.0 * A*B + 0.0 * C
        float alpha = 1.0f;
        float beta = 0.0f;

        std::int64_t lda = K;
        std::int64_t ldb = N;
        std::int64_t ldc = N;

        // 4. WARMUP RUN
        // Chiamiamo oneMKL gemm una volta a vuoto
        mkl::blas::row_major::gemm(q, transA, transB, M, N, K, alpha, a_buf, lda, b_buf, ldb, beta, c_buf, ldc);
        q.wait(); // Aspetta che il warmup finisca

        // 5. TIMING DEL CALCOLO
        auto start_compute = std::chrono::high_resolution_clock::now();
        
        // ECCO LA CHIAMATA EFFICIENTE E PORTABILE
        // Non c'è nessun kernel parallel_for!
        mkl::blas::row_major::gemm(q, transA, transB, M, N, K, alpha, a_buf, lda, b_buf, ldb, beta, c_buf, ldc);
        
        q.wait();
        
        auto end_compute = std::chrono::high_resolution_clock::now();

        // 6. CALCOLO GFLOPS
        double ms_compute = std::chrono::duration<double, std::milli>(end_compute - start_compute).count();
        double gflops = (2.0 * M * N * K) / (ms_compute * 1e6);

        std::cout << "------------------------------------------\n";
        std::cout << "Dimensione Matrice: " << M << "x" << K << " * " << K << "x" << N << "\n";
        std::cout << "Tempo di calcolo: " << ms_compute << " ms\n";
        std::cout << "Prestazioni:      " << gflops << " GFLOPS\n";
        std::cout << "------------------------------------------\n";
    }
    catch (const sycl::exception &e)
    {
        std::cerr << "Errore SYCL/MKL: " << e.what() << std::endl;
        return 1;
    }

    // 7. VERIFICA (sull'Host/CPU)
    float expected_result = K * 2.0f;
    
    std::cout << "Verifica C[0][0]: " << c_host[0] << "\n";
    std::cout << "Risultato atteso: " << expected_result << "\n";
    
    if (std::abs(c_host[0] - expected_result) < 0.001) {
        std::cout << "Verifica PASSATA!\n";
    } else {
        std::cout << "Verifica FALLITA!\n";
    }

    return 0;
}
