#include "../include/benchmark/mem_runner.hpp"

TestResult run_mem(const std::string& name, size_t bytes_per_elem, int n, dim3 blockSize, dim3 gridSize, float alpha) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * sizeof(float));
    cudaMalloc(&d_B, n * sizeof(float));
    cudaMalloc(&d_C, n * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (name == "Copy"){
        copy_kernel<<<gridSize, blockSize>>>(d_C, d_B, n);

        cudaDeviceSynchronize();
        cudaEventRecord(start);
        copy_kernel<<<gridSize, blockSize>>>(d_C, d_B, n);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
    } else if (name == "Scale"){
        scale_kernel<<<gridSize, blockSize>>>(d_C, d_B, alpha, n);

        cudaDeviceSynchronize();
        cudaEventRecord(start);
        scale_kernel<<<gridSize, blockSize>>>(d_C, d_B, alpha ,n);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
    } else if (name == "Add") {
        add_kernel<<<gridSize, blockSize>>>(d_C, d_B, d_A, n);

        cudaDeviceSynchronize();
        cudaEventRecord(start);
        add_kernel<<<gridSize, blockSize>>>(d_C, d_B, d_A, n);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
    } else if (name == "Triad") {
        triad_kernel<<<gridSize, blockSize>>>(d_C, d_B, d_A, alpha, n);

        cudaDeviceSynchronize();
        cudaEventRecord(start);
        triad_kernel<<<gridSize, blockSize>>>(d_C, d_B, d_A, alpha, n);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
    }

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double GBs = (bytes_per_elem * (double)n) / (ms * 1e6);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return {ms, GBs};
}

void benchmark_mem(const std::string& name, size_t bytes_per_elem, int n, dim3 blockSize, dim3 gridSize , float alpha, int runs) {
    std::cout << "Benchmark: " << name << " | "
              << n << " elements, " << runs << " runs\n";

    double sum_ms = 0.0, sum_gbs = 0.0;
    for (int i = 0; i < runs; i++) {
        TestResult r = run_mem(name, bytes_per_elem, n, blockSize, gridSize, alpha);
        sum_ms += r.ms;
        sum_gbs += r.gflops;
        std::cout << "Run " << i+1 << ": "
                  << r.ms << " ms, "
                  << r.gflops << " GB/s\n";
    }
    std::cout << "Average time: " << sum_ms / runs << " ms\n";
    std::cout << "Average Bandwidth: " << sum_gbs / runs << " GB/s\n\n";
}

