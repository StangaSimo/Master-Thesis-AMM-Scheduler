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

// Usa 'float' o 'sycl::half' a seconda della precisione desiderata
using data_t = float; 

// ==========================================
// STRUTTURE DATI
// ==========================================

struct PowerStats {
    double avg;
    double min;
    double max;
};

// ==========================================
// FUNZIONI AUSILIARIE
// ==========================================

PowerStats compute_stats(const std::vector<double>& v) {
    if (v.empty()) return {0.0, 0.0, 0.0};
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return { sum / v.size(), *std::min_element(v.begin(), v.end()), *std::max_element(v.begin(), v.end()) };
}

void save_results_to_csv(const std::string& filename, 
                         const std::string& device,
                         int M, int N, int K, int runs, 
                         double avg_compute_ms, double avg_memcpy_ms, double gflops, 
                         const PowerStats& gpu_stats) {
    
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "[ERRORE] Impossibile aprire il file CSV.\n";
        return;
    }

    file.seekp(0, std::ios::end);
    if (file.tellp() == 0) {
        file << "Timestamp,Device,M,N,K,Runs,Avg_Compute_ms,Avg_Memcpy_ms,GFLOPS,"
             << "GPU_Avg_W,GPU_Max_W\n";
    }

    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    
    file << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S") << ","
         << device << ","
         << M << "," << N << "," << K << "," 
         << runs << ","
         << std::fixed << std::setprecision(3)
         << avg_compute_ms << "," 
         << avg_memcpy_ms << "," 
         << gflops << ","
         << gpu_stats.avg << "," 
         << gpu_stats.max << "\n";

    std::cout << ">> Risultati salvati in: " << filename << "\n";
}

// ==========================================
// MONITORAGGIO GPU (intel_gpu_top)
// ==========================================

class GpuPowerMonitor {
private:
    std::atomic<bool> sampling;
    std::vector<double> gpu_samples;
    std::thread monitor_thread;

    void thread_loop() {
        // Nota: Assicurati di eseguire con 'sudo'
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
                    // Intel GPU Top CSV: solitamente la potenza è nelle prime colonne
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

// ==========================================
// BENCHMARK CON USM (Unified Shared Memory)
// ==========================================

namespace mkl = oneapi::mkl;

void esegui_benchmark(sycl::queue& q, int M, int N, int K, int runs) {
    std::cout << "\n=== Benchmark USM (Compute + Memcpy): " << M << "x" << N << "x" << K << " ===" << "\n";

    // 1. Dati Host
    std::vector<data_t> h_A(M * K, 1.0f);
    std::vector<data_t> h_B(K * N, 2.0f);
    std::vector<data_t> h_C(M * N, 0.0f);

    try {
        // 2. Allocazione Device (USM Explicit)
        // Allocare memoria direttamente sulla GPU ci permette di controllare le copie
        data_t* d_A = sycl::malloc_device<data_t>(M * K, q);
        data_t* d_B = sycl::malloc_device<data_t>(K * N, q);
        data_t* d_C = sycl::malloc_device<data_t>(M * N, q);

        if (!d_A || !d_B || !d_C) {
            std::cerr << "Errore allocazione memoria GPU!\n";
            return;
        }

        mkl::transpose trans = mkl::transpose::nontrans;
        data_t alpha = 1.0f, beta = 0.0f;

        // 3. Warmup
        // Copia e calcolo a vuoto
        q.memcpy(d_A, h_A.data(), M * K * sizeof(data_t));
        q.memcpy(d_B, h_B.data(), K * N * sizeof(data_t));
        mkl::blas::row_major::gemm(q, trans, trans, M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);
        q.wait();

        // 4. Inizio Benchmark
        GpuPowerMonitor monitor;
        monitor.start();

        double total_compute_ms = 0.0;
        double total_memcpy_ms = 0.0;

        for (int i = 0; i < runs; ++i) {
            // A. Misura Memcpy Host -> Device
            auto t_copy_start = std::chrono::high_resolution_clock::now();
            
            auto e1 = q.memcpy(d_A, h_A.data(), M * K * sizeof(data_t));
            auto e2 = q.memcpy(d_B, h_B.data(), K * N * sizeof(data_t));
            // Aspettiamo che la copia finisca per misurarla
            sycl::event::wait({e1, e2}); 
            
            auto t_copy_end = std::chrono::high_resolution_clock::now();
            total_memcpy_ms += std::chrono::duration<double, std::milli>(t_copy_end - t_copy_start).count();

            // B. Misura Compute (GEMM)
            auto t_compute_start = std::chrono::high_resolution_clock::now();
            
            auto e_gemm = mkl::blas::row_major::gemm(q, trans, trans, M, N, K, alpha, d_A, K, d_B, N, beta, d_C, N);
            e_gemm.wait(); // Aspettiamo che il calcolo finisca
            
            auto t_compute_end = std::chrono::high_resolution_clock::now();
            total_compute_ms += std::chrono::duration<double, std::milli>(t_compute_end - t_compute_start).count();
        }

        monitor.stop();

        // 5. Cleanup
        sycl::free(d_A, q);
        sycl::free(d_B, q);
        sycl::free(d_C, q);

        // 6. Statistiche
        double avg_compute = total_compute_ms / runs;
        double avg_memcpy = total_memcpy_ms / runs;
        double gflops = (2.0 * static_cast<double>(M) * N * K) / (avg_compute * 1e6);
        PowerStats p_stats = monitor.get_stats();

        std::cout << ">> Avg Compute Time: " << avg_compute << " ms\n";
        std::cout << ">> Avg Memcpy Time:  " << avg_memcpy << " ms (Host->Device)\n";
        std::cout << ">> GFLOPS:           " << gflops << "\n";
        std::cout << ">> Power Avg:        " << p_stats.avg << " W\n";

        // 7. Salvataggio
        std::string dev_name = q.get_device().get_info<sycl::info::device::name>();
        save_results_to_csv("benchmark_usm.csv", dev_name, M, N, K, runs, avg_compute, avg_memcpy, gflops, p_stats);

    } catch (sycl::exception const& e) {
        std::cerr << "[SYCL ERROR]: " << e.what() << "\n";
    }
}

int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v);
        std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";
        
        esegui_benchmark(q, 4096, 4096, 4096, 10);
        
    } catch (sycl::exception const& e) {
        std::cerr << "Errore inizializzazione SYCL.\n";
        return 1;
    }
    return 0;
}
