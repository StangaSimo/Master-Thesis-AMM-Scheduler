#include "../include/benchmark/gem_runner.hpp"

TestResult run_gem(const std::string& kernel, int M, int N, int K,
                           dim3 blockSize, dim3 gridSize) {
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& x : h_A) x = dis(gen);
    for (auto& x : h_B) x = dis(gen);
    for (auto& x : h_C) x = 0.0f;

    float *d_A, *d_B, *d_C;

    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    if (kernel == "simple_gem"){
        simple_gemm_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);

        cudaDeviceSynchronize();
        cudaEventRecord(start);
        simple_gemm_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
    } else if (kernel == "title_gem"){
        //title_gem_kernel<TILE><<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);

        //cudaDeviceSynchronize();
        //cudaEventRecord(start);
        //title_gem_kernel<TILE><<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
        //cudaEventRecord(stop);
        //cudaDeviceSynchronize();
    } else if (kernel == "dense_gem"){
        //dense_gem_kernel<><<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);

        cudaDeviceSynchronize();
        cudaEventRecord(start);
        //dense_gem_kernel<><<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
    }

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    double gflops = (2.0 * M * N * K) / (ms * 1e6);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return {ms, gflops};
}

void benchmark_gem(const std::string& kernel, int M, int N, int K, dim3 blockSize, dim3 gridSize, int runs) {
    std::cout << "Benchmark: " << kernel << " | "
              << M << "x" << N << " (K=" << K << "), "
              << runs << " runs\n";

    double sum_ms = 0.0, sum_gflops = 0.0;

    for (int i = 0; i < runs; i++) {
        TestResult r = run_gem(kernel, M, N, K, blockSize, gridSize);
        sum_ms += r.ms;
        sum_gflops += r.gflops;
#ifdef DEBUG
        std::cout << "Run " << i+1 << ": "
                  << r.ms << " ms, "
                  << r.gflops << " GFLOPS\n";
#endif 
    }

    std::cout << "-----------------------------\n";
    std::cout << "Average time: " << sum_ms / runs << " ms\n";
    std::cout << "Average GFLOPS: " << sum_gflops / runs << " GFLOPS\n\n";
}

