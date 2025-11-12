#include "../include/benchmark/mem_runner.hpp"
#include "../include/power.hpp"

TestResult run_mem(KernelType kernel, size_t bytes_per_elem, const int N, dim3 blockSize, dim3 gridSize, float alpha) {
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, N * sizeof(float)));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    switch(kernel) {
        case KernelType::COPY:
            copy_kernel<<<gridSize, blockSize>>>(d_C, d_B, N);
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaEventRecord(start));
            copy_kernel<<<gridSize, blockSize>>>(d_C, d_B, N);
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaDeviceSynchronize());
            break;
        case KernelType::SCALE:
            scale_kernel<<<gridSize, blockSize>>>(d_C, d_B, alpha, N);
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaEventRecord(start));
            scale_kernel<<<gridSize, blockSize>>>(d_C, d_B, alpha ,N);
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaDeviceSynchronize());
            break;
            
        case KernelType::TRIAD:
            triad_kernel<<<gridSize, blockSize>>>(d_C, d_B, d_A, alpha, N);
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaEventRecord(start));
            triad_kernel<<<gridSize, blockSize>>>(d_C, d_B, d_A, alpha, N);
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaDeviceSynchronize());
            break;
        case KernelType::ADD:
            add_kernel<<<gridSize, blockSize>>>(d_C, d_B, d_A, N);
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaEventRecord(start));
            add_kernel<<<gridSize, blockSize>>>(d_C, d_B, d_A, N);
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaDeviceSynchronize());
            break;
        default:
            throw std::invalid_argument("Unknown kernel type: " +  std::to_string(static_cast<int>(kernel)));
            break;
    }

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    double GBs = (bytes_per_elem * (double)N) / (ms * 1e6);

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return {ms, GBs};
}

void write_result_csv(const std::string &filename,
                      const std::string &kernelName,
                      int runs, 
                      std::vector<RunData> &results)
{
    if (results.empty()) {
        std::cerr << "[ERROR] results empty\n";
        return;
    }

    auto [min_ms_it, max_ms_it] = std::minmax_element(results.begin(), results.end(),
                                                      [](const RunData &a, const RunData &b) { return a.ms_compute < b.ms_compute; });
    auto [min_bw_it, max_bw_it] = std::minmax_element(results.begin(), results.end(),
                                                      [](const RunData &a, const RunData &b) { return a.gflops < b.gflops; });

    double avg_ms = 0.0, avg_bw = 0.0, avg_power = 0.0;
    double min_power = results.front().min_power;
    double max_power = results.front().max_power;

    for (const auto &r : results) {
        avg_ms += r.ms_compute;
        avg_bw += r.gflops;
        avg_power += r.avg_power;
        min_power = std::min(min_power, r.min_power);
        max_power = std::max(max_power, r.max_power);
    }

    avg_ms /= results.size();
    avg_bw /= results.size();
    avg_power /= results.size();

    auto timestamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "[ERROR]: can't open " << filename << "\n";
        return;
    }

    file.seekp(0, std::ios::end);
    if (file.tellp() == 0) {
        file << "Timestamp,Kernel,Size,Runs,"
                "Avg_Time_ms,Min_Time_ms,Max_Time_ms,"
                "Avg_Bandwidth_GBs,Min_Bandwidth_GBs,Max_Bandwidth_GBs,"
                "Avg_Power_W,Min_Power_W,Max_Power_W\n";
    }

    file << timestamp << ","
         << kernelName << ","
         << runs << ","
         << std::fixed << std::setprecision(3)
         << avg_ms << "," << min_ms_it->ms_compute << "," << max_ms_it->ms_compute << ","
         << avg_bw << "," << min_bw_it->gflops << "," << max_bw_it->gflops << ","
         << avg_power << "," << min_power << "," << max_power
         << "\n";

    file.close();

    std::cout << "\nResults (" << kernelName << ")\n";
    std::cout << runs << " run(s)\n";
    std::cout << "Timestamp: " << timestamp << "\n\n";
    std::cout << "avg ms: " << avg_ms << " (min " << min_ms_it->ms_compute << ", max " << max_ms_it->ms_compute << ")\n";
    std::cout << "avg Bandwith : " << avg_bw << " GB/s (min " << min_bw_it->gflops << ", max " << max_bw_it->gflops << ")\n";
    std::cout << "avg Power : " << avg_power << " W (min " << min_power << ", max " << max_power << ")\n";
    std::cout << "--------------------------------------------\n";
}

void benchmark_mem(KernelType kernel, size_t bytes_per_elem, const int N, dim3 blockSize, dim3 gridSize , float alpha, const int runs) {
    std::cout << "=========== Benchmark: " << getKernelName(kernel) 
              << " | " << N << " elements, " << runs << " runs\n";

    GpuPowerSampler sampler(0, 100);

    std::vector<RunData> results;
    results.reserve(runs);

    for (int i = 0; i < runs; i++) {
        unsigned int power_before, power_after;

        sampler.start();
        TestResult r = run_mem(kernel, bytes_per_elem, N, blockSize, gridSize, alpha);

        sampler.stop();

        double avg_power = sampler.averagePower();
        double min_power = sampler.minPower();
        double max_power = sampler.maxPower();

        results.push_back({r.ms_compute, r.gflops, avg_power, min_power, max_power});


#ifdef DEBUG
        std::cout << "[DEBUG]: Run " << i + 1 << ": "
                  << r.ms_compute << " ms, "
                  << r.gflops << " GB/s, "
                  << " | Power (avg/min/max): "
                  << avg_power << "/" << min_power << "/" << max_power << " W\n";
#endif
    }

    write_result_csv("mem_benchmark_results.csv", getKernelName(kernel), runs, results);
}

