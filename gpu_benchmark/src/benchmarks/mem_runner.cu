#include "../include/benchmark/mem_runner.hpp"

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

void benchmark_mem(KernelType kernel, size_t bytes_per_elem, const int N, dim3 blockSize, dim3 gridSize , float alpha, const int runs) {
    std::cout << "Benchmark: " << getKernelName(kernel) << " | "
              << N << " elements, " << runs << " runs\n";

    double sum_ms = 0.0, sum_gbs = 0.0;
    for (int i = 0; i < runs; i++) {
        TestResult r = run_mem(kernel, bytes_per_elem, N, blockSize, gridSize, alpha);
        sum_ms += r.ms;
        sum_gbs += r.gflops;
#ifdef DEBUG
        std::cout << "[DEBUG]: Run " << i+1 << ": " << r.ms << " ms, " << r.gflops << " GB/s\n";
#endif
    }
    std::cout << "Average time: " << sum_ms / runs << " ms\n";
    std::cout << "Average Bandwidth: " << sum_gbs / runs << " GB/s\n\n";
}

