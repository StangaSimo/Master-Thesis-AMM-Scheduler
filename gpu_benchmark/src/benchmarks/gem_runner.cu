#include "../include/benchmark/gem_runner.hpp"

TestResult run_gem(KernelType kernel, const int M, const int N, const int K, dim3 blockSize, dim3 gridSize) {

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

    switch(kernel) {
        case KernelType::NAIVE:
            simple_gemm_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaEventRecord(start));
            simple_gemm_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaDeviceSynchronize());
            simple_gemm_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
            break;
        case KernelType::TILED:
            tile_gem_kernel<TILE_SIZE><<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaEventRecord(start));
            tile_gem_kernel<TILE_SIZE><<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaDeviceSynchronize());
            break;
        case KernelType::DENSE:
            dense_gem_kernel<DENSE_BLOCK_M, DENSE_BLOCK_K, DENSE_BLOCK_N, DENSE_THREAD_Y, DENSE_THREAD_X>
                            <<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K, DENSE_ALPHA, DENSE_BETA);
            CHECK_CUDA(cudaDeviceSynchronize()); 

            CHECK_CUDA(cudaEventRecord(start));
            dense_gem_kernel<DENSE_BLOCK_M, DENSE_BLOCK_K, DENSE_BLOCK_N, DENSE_THREAD_Y, DENSE_THREAD_X>
                            <<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K, DENSE_ALPHA, DENSE_BETA);
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaDeviceSynchronize());
            break;
        case KernelType::CUBLAS:
            {
                float alpha = 1.0f;            
                float beta = 0.0f;
                cublasHandle_t blas_handle;  
                //CHECK_CUBLAS(cublasCreate(&blas_handle));

                //cublasSgemm (blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                //    N, M, K, &alpha, 
                //    d_B, N, d_A, K, &beta, d_C, N
                //);

                //CHECK_CUDA(cudaDeviceSynchronize()); 
                //CHECK_CUDA(cudaEventRecord(start));
                //cublasSgemm (blas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                //    N, M, K, &alpha, 
                //    d_B, N, d_A, K, &beta, d_C, N
                //);
                //CHECK_CUDA(cudaEventRecord(stop));
                //CHECK_CUDA(cudaDeviceSynchronize());
                //cublasDestroy(blas_handle); 
            }
            break;
        default:
            throw std::invalid_argument("Unknown kernel type: " +  std::to_string(static_cast<int>(kernel)));
            break;
    }

    float ms = 0.0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    double gflops = (2.0 * M * N * K) / (ms * 1e6);

#ifdef CHECK_RESULT
    static std::vector<float> h_C_cpu(M * N); 
    static bool calculated = false;
    if (!calculated) {
        std::cout << "------ [DEBUG RESULT]: compute matrix on cpu... ";
        cpu_gemm(h_A.data(), h_B.data(), h_C_cpu.data(), M, N, K);
        std::cout << " ...cpu complete \n";
        calculated = true;
    }

    std::vector<float> h_C_gpu(M * N);
    CHECK_CUDA(cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    if (compare_matrix(h_C_gpu.data(), h_C_cpu.data(), M * N)) {
        std::cout << "------ [DEBUG RESULT] ✓ Results are CORRECT for " << getKernelName(kernel) << std::endl;
    } else {
        std::cerr << "------ [DEBUG RESULT] ✗ Results are INCORRECT for " << getKernelName(kernel) << std::endl;
    }
#endif

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return {ms, gflops};
}

void benchmark_gem(KernelType kernel, const int M, const int N, const int K, dim3 blockSize, dim3 gridSize, const int runs) {
    std::cout << "Benchmark: " << getKernelName(kernel) << " | " << M << "x" << N << " (K=" << K << "), " << runs << " runs\n";

    double sum_ms = 0.0, sum_gflops = 0.0;

    for (int i = 0; i < runs; i++) {
        TestResult r = run_gem(kernel, M, N, K, blockSize, gridSize);
        sum_ms += r.ms;
        sum_gflops += r.gflops;

#ifdef DEBUG
        std::cout << "[DEBUG]: Run " << i+1 << ": " << r.ms << " ms, " << r.gflops << " GFLOPS\n";
#endif 

    }

    std::cout << "-----------------------------\n";
    std::cout << "Average time: " << sum_ms / runs << " ms\n";
    std::cout << "Average GFLOPS: " << sum_gflops / runs << " GFLOPS\n\n";
}

