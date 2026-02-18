#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <thread>
#include <atomic>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdio>
#include <cstdint> // Per uint16_t

// sycl::half invece di float
using data_t = sycl::half; 

struct PowerStats {
    double avg;
    double min;
    double max;
};

PowerStats compute_stats(const std::vector<double>& v) {
    if (v.empty()) return {0.0, 0.0, 0.0};
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return { sum / v.size(), *std::min_element(v.begin(), v.end()), 
                        *std::max_element(v.begin(), v.end()) };
}

void save_results_to_csv(const std::string& filename, 
                         const std::string& device,
                         int M, int N, int K, int runs, 
                         double avg_compute_ms, double avg_memcpy_ms, double gflops, 
                         const PowerStats& gpu_stats) {
    
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "[ERRORE] CSV\n";
        return;
    }

    file.seekp(0, std::ios::end);
    if (file.tellp() == 0) {
        file << "Timestamp,Device,Mode,Precision,M,N,K,Runs,Avg_Compute_ms,Avg_Memcpy_ms,GFLOPS,"
             << "GPU_Avg_W,GPU_Max_W\n";
    }

    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    
    #ifdef ZEROCOPY
    std::string mode = "ZeroCopy";
    #else
    std::string mode = "Explicit";
    #endif

    // Identifichiamo la precisione
    std::string prec = (sizeof(data_t) == 2) ? "FP16" : "FP32";

    file << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S") << ","
         << device << ","
         << mode << ","
         << prec << ","
         << M << "," << N << "," << K << "," 
         << runs << ","
         << std::fixed << std::setprecision(3)
         << avg_compute_ms << "," 
         << avg_memcpy_ms << "," 
         << gflops << ","
         << gpu_stats.avg << "," 
         << gpu_stats.max << "\n";

    std::cout << ">> Results in: " << filename << "\n";
}

class GpuPowerMonitor {
private:
    std::atomic<bool> sampling;
    std::vector<double> gpu_samples;
    std::thread monitor_thread;

    void thread_loop() {
        FILE* fp = popen("sudo intel_gpu_top -c -s 100", "r");
        if (!fp) return;

        char buffer[2048];
        bool header_skipped = false;

        while (sampling) {
            if (fgets(buffer, sizeof(buffer), fp)) {
                if (!header_skipped) {
                    header_skipped = true;
                } else {
                    std::stringstream ss(buffer);
                    std::string token;
                    for (int i = 0; i <= 4; i++) std::getline(ss, token, ',');
                    try { 
                        double val = std::stod(token);
                        if(val > 0) gpu_samples.push_back(val);
                    } catch (...) {}
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        pclose(fp);
    }

public:
    GpuPowerMonitor() : sampling(false) {}

    void start() {
        if (sampling) return;
        gpu_samples.clear();
        sampling = true;
        monitor_thread = std::thread(&GpuPowerMonitor::thread_loop, this);
    }

    void stop() {
        if (!sampling) return;
        sampling = false;
        if (monitor_thread.joinable()) monitor_thread.join();
    }

    PowerStats get_stats() { return compute_stats(gpu_samples); }
};

namespace mkl = oneapi::mkl;

void benchmark(sycl::queue& q, int M, int N, int K, int runs) {
    #ifdef ZEROCOPY
    std::cout << "\n=== Benchmark FP16 [ZERO-COPY]: " << M << "x" << N << "x" << K << " ===" << "\n";
    #else
    std::cout << "\n=== Benchmark FP16 [EXPLICIT]: " << M << "x" << N << "x" << K << " ===" << "\n";
    #endif

    std::vector<data_t> h_A(M * K, data_t(1.0f));
    std::vector<data_t> h_B(K * N, data_t(2.0f));
    std::vector<data_t> h_C(M * N, data_t(0.0f));

    try {
        data_t *d_A, *d_B, *d_C;

        #ifdef ZEROCOPY
            d_A = sycl::malloc_shared<data_t>(M * K, q);
            d_B = sycl::malloc_shared<data_t>(K * N, q);
            d_C = sycl::malloc_shared<data_t>(M * N, q);
        #else
            d_A = sycl::malloc_device<data_t>(M * K, q);
            d_B = sycl::malloc_device<data_t>(K * N, q);
            d_C = sycl::malloc_device<data_t>(M * N, q);
        #endif

        if (!d_A || !d_B || !d_C) {
            std::cerr << "[ERROR] Memory GPU\n";
            return;
        }

        #ifdef ZEROCOPY
            std::copy(h_A.begin(), h_A.end(), d_A);
            std::copy(h_B.begin(), h_B.end(), d_B);
        #else
            q.memcpy(d_A, h_A.data(), M * K * sizeof(data_t));
            q.memcpy(d_B, h_B.data(), K * N * sizeof(data_t));
        #endif
        q.wait();

        mkl::transpose trans = mkl::transpose::nontrans;
        data_t alpha = data_t(1.0f); 
        data_t beta = data_t(0.0f);

        mkl::blas::row_major::gemm(q, trans, trans, M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);
        q.wait();

        GpuPowerMonitor monitor;
        monitor.start();

        double total_compute_ms = 0.0;
        double total_memcpy_ms = 0.0;

        for (int i = 0; i < runs; ++i) {
            
            #ifndef ZEROCOPY
            auto t_copy_start = std::chrono::high_resolution_clock::now();
            
            auto e1 = q.memcpy(d_A, h_A.data(), M * K * sizeof(data_t));
            auto e2 = q.memcpy(d_B, h_B.data(), K * N * sizeof(data_t));
            sycl::event::wait({e1, e2});
            
            auto t_copy_end = std::chrono::high_resolution_clock::now();
            total_memcpy_ms += std::chrono::duration<double, std::milli>(t_copy_end - t_copy_start).count();
            #endif

            auto t_compute_start = std::chrono::high_resolution_clock::now();
            
            auto e_gemm = mkl::blas::row_major::gemm(q, trans, trans, M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);
            e_gemm.wait();
            
            auto t_compute_end = std::chrono::high_resolution_clock::now();
            total_compute_ms += std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count();
        }

        monitor.stop();

        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_C, q);

        double avg_compute = total_compute_ms / runs;
        double avg_memcpy = total_memcpy_ms / runs;
        double gflops = (2.0 * static_cast<double>(M) * N * K) / (avg_compute * 1e6);
        PowerStats p_stats = monitor.get_stats();

        std::cout << ">> Avg Compute: " << avg_compute << " ms\n";
        std::cout << ">> Avg Memcpy:  " << avg_memcpy << " ms\n";
        std::cout << ">> GFLOPS:      " << gflops << "\n";
        std::cout << ">> Power:       " << p_stats.avg << " W\n";

        std::string dev_name = q.get_device().get_info<sycl::info::device::name>();
        save_results_to_csv("benchmark_usm_fp16.csv", dev_name, M, N, K, runs, avg_compute, avg_memcpy, gflops, p_stats);

    } catch (sycl::exception const& e) {
        std::cerr << "[SYCL ERROR]: " << e.what() << "\n";
    }
}

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v);
        auto dev = q.get_device();
        std::cout << "Device: " << dev.get_info<sycl::info::device::name>() << "\n";
        
        if (!dev.has(sycl::aspect::fp16)) {
            std::cerr << "[ERROR] no FP16!\n";
            return 1;
        }
        
        benchmark(q, 4096, 4096, 4096, 10);
        
    } catch (sycl::exception const& e) {
        std::cerr << "[ERROR] SYCL\n";
        return 1;
    }
    return 0;
}
