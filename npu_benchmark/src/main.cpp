#include <openvino/openvino.hpp>
#include <openvino/op/matmul.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <chrono>
#include <cmath>

#include <fstream>
#include <thread>
#include <atomic>

//#define CHECK_RESULT  

struct TestResult {
    double ms;
    double gflops;
    bool correct;
};

void cpu_gemm_fp16(const ov::float16* A, const ov::float16* B, ov::float16* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += static_cast<float>(A[i * K + k]) * static_cast<float>(B[k * N + j]);
            }
            C[i * N + j] = ov::float16(sum);
        }
    }
}

TestResult run_matmul(ov::Core& core, const std::string& device_name,
                      int M, int N, int K,
                      const std::vector<ov::float16>& h_A,
                      const std::vector<ov::float16>& h_B) {

    auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{(u_long)M, (u_long)K});
    auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{(u_long)K, (u_long)N});
    auto matmul = std::make_shared<ov::op::v0::MatMul>(A, B);
    auto result = std::make_shared<ov::op::v0::Result>(matmul);
    auto model = std::make_shared<ov::Model>(result, ov::ParameterVector{A, B});

    ov::CompiledModel compiled_model = core.compile_model(model, device_name);
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    ov::Tensor tensor_A(ov::element::f16, ov::Shape{(u_long)M, (u_long)K}, const_cast<ov::float16*>(h_A.data()));
    ov::Tensor tensor_B(ov::element::f16, ov::Shape{(u_long)K, (u_long)N}, const_cast<ov::float16*>(h_B.data()));

    infer_request.set_input_tensor(0, tensor_A);
    infer_request.set_input_tensor(1, tensor_B);

    infer_request.infer();

    auto start = std::chrono::high_resolution_clock::now();
    infer_request.infer();
    auto end = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops = (2.0 * M * N * K) / (ms * 1e6);

    bool correct = true;

#ifdef CHECK_RESULT
    static std::vector<ov::float16> h_C_ref;
    static bool calculated = false;
    if (!calculated) {
        std::cout << "[CHECK_RESULT] Calcolo reference CPU FP16...\n";
        h_C_ref.resize(M * N);
        cpu_gemm_fp16(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
        calculated = true;
        std::cout << "[CHECK_RESULT] Reference CPU completata\n";
    }

    ov::Tensor output_tensor = infer_request.get_output_tensor();
    ov::float16* C_ov = output_tensor.data<ov::float16>();

    /* norma L2 per confrontare le matrici */
    double diff_norm = 0.0, ref_norm = 0.0;
    for (int i = 0; i < M * N; i++) {
        double d = static_cast<double>(h_C_ref[i]) - static_cast<double>(C_ov[i]);
        diff_norm += d * d;
        ref_norm += static_cast<double>(h_C_ref[i]) * static_cast<double>(h_C_ref[i]);
    }
    double rel_error = std::sqrt(diff_norm) / (std::sqrt(ref_norm) + 1e-12);

    const double tol = 5e-2;  
    correct = (rel_error < tol);
    // std::cout << "Relative L2 error: " << rel_error << std::endl; 
#endif

    return {ms, gflops, correct};
}

void benchmark_device(const std::string& device_name, int M, int N, int K, int runs) {
    ov::Core core;

    bool available = false;
    for (auto& d : core.get_available_devices())
        if (d.find(device_name) != std::string::npos) available = true;
    if (!available) {
        std::cout << "Device " << device_name << " non disponibile\n";
        return;
    }

    std::cout << "Benchmark: " << device_name << " | "
              << M << "x" << N << " (K=" << K << "), "
              << runs << " runs\n";

    std::vector<ov::float16> h_A(M * K), h_B(K * N);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& x : h_A) x = ov::float16(dis(gen));
    for (auto& x : h_B) x = ov::float16(dis(gen));

    double sum_ms = 0.0, sum_gflops = 0.0;
    bool all_correct = true;

    for (int i = 0; i < runs; i++) {
        TestResult r = run_matmul(core, device_name, M, N, K, h_A, h_B);
        sum_ms += r.ms;
        sum_gflops += r.gflops;
        all_correct &= r.correct;
    }

    std::cout << "-----------------------------\n";
    std::cout << "Average time:   " << sum_ms / runs << " ms\n";
    std::cout << "Average GFLOPS: " << sum_gflops / runs << " GFLOPS\n";
#ifdef CHECK_RESULT
    std::cout << "Correctness:    " << (all_correct ? "✓ OK" : "✗ ERROR") << "\n";
#endif
    std::cout << "\n";
}

std::atomic<bool> stop_gpu_monitor(false);

// legge l'energia totale in microjoule dal file RAPL
double read_energy_uj(const std::string& path) {
    std::ifstream file(path);
    double val = 0.0;
    if (file.is_open()) file >> val;
    return val;
}

// thread di monitoraggio: legge energia ogni interval_ms e calcola potenza istantanea
void monitor_gpu_power(const std::string& path, int interval_ms, std::vector<double>& powers) {
    double last_energy = read_energy_uj(path);
    auto last_time = std::chrono::high_resolution_clock::now();

    while (!stop_gpu_monitor) {
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));

        double current_energy = read_energy_uj(path);
        auto current_time = std::chrono::high_resolution_clock::now();

        double delta_energy_j = (current_energy - last_energy) / 1e6; // µJ -> J
        double delta_time_s = std::chrono::duration<double>(current_time - last_time).count();

        double power_w = delta_energy_j / delta_time_s;
        powers.push_back(power_w);

        std::cout << "[GPU Power] " << power_w << " W\n";

        last_energy = current_energy;
        last_time = current_time;
    }
}

int main() {
    const int RUNS = 10;

    std::vector<double> gpu_powers;
    std::string gpu_energy_path = "/sys/class/powercap/intel-rapl/intel-rapl:0:1/energy_uj";

    std::thread gpu_monitor(monitor_gpu_power, gpu_energy_path, 500, std::ref(gpu_powers));

    const int M = 2048, N = 2048, K = 1024;
    std::cout << "Ops totali: " << (2.0 * M * N * K / 1e9) << " GFLOPs\n\n";

    benchmark_device("GPU", M, N, K, RUNS);

    stop_gpu_monitor = true;
    gpu_monitor.join();

    std::cout << "Potenza GPU registrata su " << gpu_powers.size() << " intervalli.\n";

    return 0;
}

    // -------------------------
    // Piccolo (low-latency / batch ridotto)
    // -------------------------
    //{
    //    const int M = 16, N = 128, K = 512;
    //    std::cout << "Benchmark OpenVINO MatMul FP16 (Piccolo)\n";
    //    std::cout << "Ops totali: " << (2.0 * M * N * K / 1e9) << " GFLOPs\n\n";

    //    benchmark_device("CPU", M, N, K, RUNS);
    //    benchmark_device("GPU", M, N, K, RUNS);
    //    benchmark_device("NPU", M, N, K, RUNS);
    //}

    // -------------------------
    // Medio (tipico layer fully connected)
    // -------------------------
    //{
    //    const int M = 1024, N = 1024, K = 512;
    //    std::cout << "Benchmark OpenVINO MatMul FP16 (Medio)\n";
    //    std::cout << "Ops totali: " << (2.0 * M * N * K / 1e9) << " GFLOPs\n\n";

    //    benchmark_device("CPU", M, N, K, RUNS);
    //    benchmark_device("GPU", M, N, K, RUNS);
    //    benchmark_device("NPU", M, N, K, RUNS);
    //}

    // -------------------------
    // Grande (peak compute / stress memoria)
    // -------------------------
    //{
    //    const int M = 2048, N = 2048, K = 1024;
    //    std::cout << "Benchmark OpenVINO MatMul FP16 (Grande)\n";
    //    std::cout << "Ops totali: " << (2.0 * M * N * K / 1e9) << " GFLOPs\n\n";

    //    benchmark_device("CPU", M, N, K, RUNS);
    //    benchmark_device("GPU", M, N, K, RUNS);
    //    benchmark_device("NPU", M, N, K, RUNS);
    //}

    // -------------------------
    // Alta e stretta (stress memoria e cache)
    // -------------------------
    //{
    //    const int M = 4096, N = 512, K = 1024;
    //    std::cout << "Benchmark OpenVINO MatMul FP16 (Alta e stretta)\n";
    //    std::cout << "Ops totali: " << (2.0 * M * N * K / 1e9) << " GFLOPs\n\n";

    //    benchmark_device("CPU", M, N, K, RUNS);
    //    benchmark_device("GPU", M, N, K, RUNS);
    //    benchmark_device("NPU", M, N, K, RUNS);
    //}

    // -------------------------
    // Lunga e bassa (stress memoria e cache)
    // -------------------------
    //{
    //    const int M = 512, N = 4096, K = 1024;
    //    std::cout << "Benchmark OpenVINO MatMul FP16 (Lunga e bassa)\n";
    //    std::cout << "Ops totali: " << (2.0 * M * N * K / 1e9) << " GFLOPs\n\n";

    //    benchmark_device("CPU", M, N, K, RUNS);
    //    benchmark_device("GPU", M, N, K, RUNS);
    //    benchmark_device("NPU", M, N, K, RUNS);
    //}

