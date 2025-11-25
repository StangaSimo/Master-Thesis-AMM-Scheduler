#include <openvino/openvino.hpp>
#include <openvino/op/matmul.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <chrono>
//#include <cmath>
#include <fstream>
#include <thread>
#include <atomic>
#include <iomanip>
#include <sstream>
#include <algorithm>

#define DEBUG 1

struct PowerSample {
    double pkg_w, core_w, uncore_w, gpu_w, npu_w;
    double timestamp;
};

std::atomic<bool> sampling{false};
std::vector<PowerSample> samples;

double read_value(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) return -1;
    double value;
    file >> value;
    return value;
}

double read_rapl_energy(const std::string &domain) {
    std::string path = "/sys/class/powercap/" + domain + "/energy_uj";
    return read_value(path);
}

void power_monitor_thread() {
    auto t0 = std::chrono::high_resolution_clock::now();

    double pkg_prev = read_rapl_energy("intel-rapl:0");
    double core_prev = read_rapl_energy("intel-rapl:0/intel-rapl:0:0");
    double uncore_prev = read_rapl_energy("intel-rapl:0/intel-rapl:0:1");

    FILE *fp = popen("sudo intel_gpu_top -c -s 100", "r");
    if (!fp) {
        std::cerr << "[ERROR] Impossibile avviare intel_gpu_top, userò la stima RAPL per la GPU.\n";
    }

    char buffer[2048];
    double gpu_w = 0.0;
    bool header_skipped = false;

    while (sampling) {
        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - t0).count();
        t0 = now;

        double pkg_now = read_rapl_energy("intel-rapl:0");
        double core_now = read_rapl_energy("intel-rapl:0/intel-rapl:0:0");
        double uncore_now = read_rapl_energy("intel-rapl:0/intel-rapl:0:1");

        if (pkg_now < 0 || core_now < 0 || uncore_now < 0) {
            std::cout << "[ERROR] RAPL read failed\n";
            break;
        }

        double pkg_w = (pkg_now - pkg_prev) * 1e-6 / dt;
        double core_w = (core_now - core_prev) * 1e-6 / dt;
        double uncore_w = (uncore_now - uncore_prev) * 1e-6 / dt;

        pkg_prev = pkg_now;
        core_prev = core_now;
        uncore_prev = uncore_now;

        if (fp && fgets(buffer, sizeof(buffer), fp)) {
            if (!header_skipped) {
                header_skipped = true;
                continue;
            }
            std::string line(buffer);
            std::stringstream ss(line);
            std::string token;
            for (int i = 0; i <= 4; i++)
                std::getline(ss, token, ',');
            try {
                gpu_w = std::stod(token);
            } catch (...) {
                gpu_w = 0;
            }
        } else {
            gpu_w = 0;
        }

        double npu_w = pkg_w - (core_w + uncore_w);
        if (npu_w < 0.0) npu_w = 0.0;

        double timestamp = std::chrono::duration<double>(now.time_since_epoch()).count();
        samples.push_back({pkg_w, core_w, uncore_w, gpu_w, npu_w, timestamp});

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (fp)
        pclose(fp);
}

struct PowerStats {
    double avg, min, max;
};

PowerStats compute_stats(std::vector<double> &v) {
    if (v.empty()) return {0, 0, 0};
    double sum = 0;
    for (auto &x : v) sum += x;
    return {
        sum / v.size(),
        *std::min_element(v.begin(), v.end()),
        *std::max_element(v.begin(), v.end())
    };
}

struct PowerSummary {
    PowerStats pkg, core, uncore, gpu, npu;
};

PowerSummary compute_power_summary() {
    std::vector<double> pkg, core, uncore, gpu, npu;
    for (auto &s : samples) {
        pkg.push_back(s.pkg_w);
        core.push_back(s.core_w);
        uncore.push_back(s.uncore_w);
        gpu.push_back(s.gpu_w);
        npu.push_back(s.npu_w);
    }
    return {compute_stats(pkg), compute_stats(core), compute_stats(uncore), compute_stats(gpu), compute_stats(npu)};
}

void save_results_to_csv(const std::string &filename,
                         const std::string &device,
                         int M, int N, int K, int runs,
                         const PowerSummary &summary,
                         double avg_time_compute, double avg_time_memcpy, double avg_gflops) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "[Error]: can't open " << filename << "\n";
        return;
    }

    auto timestamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    file.seekp(0, std::ios::end);

    /* if the file is new, write the intestation */
    if (file.tellp() == 0) {
        file << "Timestamp,Device,M,N,K,Runs,"
                "Avg_Compute_ms,Avg_MemCpy_ms,Avg_GFLOPS,"
                "Pkg_Avg_W,Core_Avg_W,Uncore_Avg_W,GPU_Avg_W,NPU_Avg_W,"
                "Pkg_Min_W,Core_Min_W,Uncore_Min_W,GPU_Min_W,NPU_Min_W,"
                "Pkg_Max_W,Core_Max_W,Uncore_Max_W,GPU_Max_W,NPU_Max_W\n";
    }

    /* csv format */
    file << timestamp << ","
         << device << ","
         << M << "," << N << "," << K << ","
         << runs << ","
         << std::fixed << std::setprecision(3)
         << avg_time_compute << ","
         << avg_time_memcpy << ","
         << avg_gflops << ","
         << summary.pkg.avg << "," << summary.core.avg << "," << summary.uncore.avg << "," << summary.gpu.avg << "," << summary.npu.avg << ","
         << summary.pkg.min << "," << summary.core.min << "," << summary.uncore.min << "," << summary.gpu.min << "," << summary.npu.min << ","
         << summary.pkg.max << "," << summary.core.max << "," << summary.uncore.max << "," << summary.gpu.max << "," << summary.npu.max
         << "\n";

    file.close();

#ifdef DEBUG
    std::cout << "\n=== Results ===\n";
    std::cout << "Device: " << device << "\n";
    std::cout << "Matrix: " << M << "x" << N << "x" << K << " | Runs: " << runs << "\n";
    std::cout << "Avg Compute Time: " << avg_time_compute << " ms | Avg MemCpy Time: " << avg_time_memcpy << " ms | Avg GFLOPS: " << avg_gflops << "\n";
    std::cout << "Power (W):\n"
              << "  Package: " << summary.pkg.avg << " (min " << summary.pkg.min << ", max " << summary.pkg.max << ")\n"
              << "  Core:    " << summary.core.avg << " (min " << summary.core.min << ", max " << summary.core.max << ")\n"
              << "  Uncore:  " << summary.uncore.avg << " (min " << summary.uncore.min << ", max " << summary.uncore.max << ")\n"
              << "  GPU:     " << summary.gpu.avg << " (min " << summary.gpu.min << ", max " << summary.gpu.max << ")\n"
              << "  NPU:     " << summary.npu.avg << " (min " << summary.npu.min << ", max " << summary.npu.max << ")\n";
    std::cout << "================================\n\n";
#endif
}


struct TestResult {
    double ms_compute; 
    double ms_memcpy;
    double gflops;
    bool correct;
};

TestResult run_matmul(ov::Core &core, const std::string &device_name,
                      int M, int N, int K,
                      const std::vector<ov::float16> &h_A,
                      const std::vector<ov::float16> &h_B) {
    auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{(u_long)M, (u_long)K});
    auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{(u_long)K, (u_long)N});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(A, B);
    auto result = std::make_shared<ov::op::v0::Result>(matmul);
    auto model = std::make_shared<ov::Model>(result, ov::ParameterVector{A, B});

    ov::CompiledModel compiled_model = core.compile_model(model, device_name);
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    ov::Tensor tensor_A(ov::element::f16, ov::Shape{(u_long)M, (u_long)K}, const_cast<ov::float16 *>(h_A.data()));
    ov::Tensor tensor_B(ov::element::f16, ov::Shape{(u_long)K, (u_long)N}, const_cast<ov::float16 *>(h_B.data()));

    /* memcpy or wrapper timing */
    auto start_memcpy = std::chrono::high_resolution_clock::now();
    infer_request.set_input_tensor(0, tensor_A);
    infer_request.set_input_tensor(1, tensor_B);
    auto end_memcpy = std::chrono::high_resolution_clock::now();
    double ms_memcpy = std::chrono::duration<double, std::milli>(end_memcpy - start_memcpy).count();

    /* warmup */
    infer_request.infer();

    /* computing timing */
    auto start_compute = std::chrono::high_resolution_clock::now();
    infer_request.infer();
    auto end_compute = std::chrono::high_resolution_clock::now();

    double ms_compute = std::chrono::duration<double, std::milli>(end_compute - start_compute).count();
    double gflops = (2.0 * M * N * K) / (ms_compute * 1e6);

    return {ms_compute, ms_memcpy, gflops, true};
}

void benchmark_device(const std::string &device_name, int M, int N, int K, int runs) {
    ov::Core core;

    bool available = false;
    for (auto &d : core.get_available_devices())
        if (d.find(device_name) != std::string::npos)
            available = true;
    if (!available) {
        std::cout << "Device " << device_name << " non disponibile\n";
        return;
    }


#ifdef DEBUG
    std::cout << "\n=== Benchmark: " << device_name << " ===\n";
    std::cout << "Matrix size: " << M << "x" << N << "x" << K << " | Runs: " << runs << "\n";
#endif 

    std::vector<ov::float16> h_A(M * K), h_B(K * N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto &x : h_A) x = ov::float16(dis(gen));
    for (auto &x : h_B) x = ov::float16(dis(gen));

    double sum_ms_compute = 0.0, sum_ms_memcpy = 0.0, sum_gflops = 0.0; // Modificato

    samples.clear();
    sampling = true;
    std::thread mon(power_monitor_thread);

    for (int i = 0; i < runs; i++) {
        TestResult r = run_matmul(core, device_name, M, N, K, h_A, h_B);
        sum_ms_compute += r.ms_compute;
        sum_ms_memcpy += r.ms_memcpy;
        sum_gflops += r.gflops;
    }

    sampling = false;
    mon.join();

    double avg_time_compute = sum_ms_compute / runs;
    double avg_time_memcpy = sum_ms_memcpy / runs;
    double avg_gflops = sum_gflops / runs;
    auto summary = compute_power_summary();

#ifdef DEBUG
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Average Compute Time: " << avg_time_compute << " ms | Average MemCpy Time: " << avg_time_memcpy << " ms | Average GFLOPS: " << avg_gflops << "\n";
    std::cout << "Core avg: " << summary.core.avg << "W (min " << summary.core.min << ", max " << summary.core.max << ")\n";
    std::cout << "Uncore avg: " << summary.uncore.avg << "W (min " << summary.uncore.min << ", max " << summary.uncore.max << ")\n";
    std::cout << "GPU avg: " << summary.gpu.avg << "W (min " << summary.gpu.min << ", max " << summary.gpu.max << ")\n";
    std::cout << "NPU est.: " << summary.npu.avg << "W (min " << summary.npu.min << ", max " << summary.npu.max << ")\n";
#endif 

    save_results_to_csv("power_benchmarks.csv", device_name, M, N, K, runs, summary, avg_time_compute, avg_time_memcpy, avg_gflops);
}

int main() {
    benchmark_device("CPU", 4096, 4096, 2048, 10);
    benchmark_device("GPU", 4096, 4096, 2048, 10);
    benchmark_device("NPU", 4096, 4096, 2048, 10);
    return 0;
}
