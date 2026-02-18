#include "../include/benchmark/gem_runner.hpp"
#include "../include/power.hpp"

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

    cudaEvent_t start_mem, stop_mem;
    CHECK_CUDA(cudaEventCreate(&start_mem));
    CHECK_CUDA(cudaEventCreate(&stop_mem));

    CHECK_CUDA(cudaEventRecord(start_mem)); /* memory timer */
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop_mem));

    float ms_memcpy = 0.0;
    CHECK_CUDA(cudaEventSynchronize(stop_mem)); /* make sure all the op are done */
    CHECK_CUDA(cudaEventElapsedTime(&ms_memcpy, start_mem, stop_mem));

    cudaEvent_t start_compute, stop_compute;
    CHECK_CUDA(cudaEventCreate(&start_compute));
    CHECK_CUDA(cudaEventCreate(&stop_compute));

    switch(kernel) {
        case KernelType::NAIVE:
            simple_gemm_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaEventRecord(start_compute));
            simple_gemm_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
            CHECK_CUDA(cudaEventRecord(stop_compute));
            CHECK_CUDA(cudaDeviceSynchronize());
            simple_gemm_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
            break;
        case KernelType::TILED:
            tile_gem_kernel<TILE_SIZE><<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaEventRecord(start_compute));
            tile_gem_kernel<TILE_SIZE><<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
            CHECK_CUDA(cudaEventRecord(stop_compute));
            CHECK_CUDA(cudaDeviceSynchronize());
            break;
        case KernelType::DENSE:
            dense_gem_kernel<DENSE_BLOCK_M, DENSE_BLOCK_K, DENSE_BLOCK_N, DENSE_THREAD_Y, DENSE_THREAD_X>
                            <<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K, DENSE_ALPHA, DENSE_BETA);
            CHECK_CUDA(cudaDeviceSynchronize()); 

            CHECK_CUDA(cudaEventRecord(start_compute));
            dense_gem_kernel<DENSE_BLOCK_M, DENSE_BLOCK_K, DENSE_BLOCK_N, DENSE_THREAD_Y, DENSE_THREAD_X>
                            <<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K, DENSE_ALPHA, DENSE_BETA);
            CHECK_CUDA(cudaEventRecord(stop_compute));
            CHECK_CUDA(cudaDeviceSynchronize());
            break;
        case KernelType::CUBLAS:
            {
                float alpha = 1.0f, beta = 0.0f;
                cublasHandle_t handle;  
                CHECK_CUBLAS(cublasCreate(&handle));

                cublasSgemm (handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    N, M, K, &alpha, 
                    d_B, N, d_A, K, &beta, d_C, N
                );

                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUDA(cudaEventRecord(start_compute));
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K, &alpha,
                            d_B, N, d_A, K, &beta, d_C, N);
                CHECK_CUDA(cudaEventRecord(stop_compute));
                CHECK_CUDA(cudaDeviceSynchronize());
                CHECK_CUBLAS(cublasDestroy(handle));
                break;
            }
        case KernelType::TENSOR:
            {
                half *d_A16, *d_B16;

                /* 16 bit matrix for input */
                CHECK_CUDA(cudaMalloc(&d_A16, M * K * sizeof(half)));
                CHECK_CUDA(cudaMalloc(&d_B16, K * N * sizeof(half)));

                /* converto da 32 to 16 */
                cudaDeviceSynchronize();
                cudaMemcpy(d_A16, d_A, M * K * sizeof(half), cudaMemcpyDeviceToDevice);
                cudaMemcpy(d_B16, d_B, K * N * sizeof(half), cudaMemcpyDeviceToDevice);
                cublasHandle_t handle;
                CHECK_CUBLAS(cublasCreate(&handle));

                /* enable tensor core */
                CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
                const float alpha = 1.0f, beta = 0.0f;

                CHECK_CUBLAS(cublasGemmEx(handle,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          N, M, K,
                                          &alpha,
                                          d_B16, CUDA_R_16F, N,
                                          d_A16, CUDA_R_16F, K,
                                          &beta,
                                          d_C, CUDA_R_32F, N,
                                          CUDA_R_32F,
                                          /* force tensor core */
                                          CUBLAS_GEMM_DEFAULT_TENSOR_OP));

                cudaDeviceSynchronize();
                CHECK_CUDA(cudaEventRecord(start_compute));
                CHECK_CUBLAS(cublasGemmEx(handle,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          N, M, K,
                                          &alpha,
                                          d_B16, CUDA_R_16F, N,
                                          d_A16, CUDA_R_16F, K,
                                          &beta,
                                          d_C, CUDA_R_32F, N,
                                          CUDA_R_32F,
                                          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                CHECK_CUDA(cudaEventRecord(stop_compute));
                cudaDeviceSynchronize();

                cublasDestroy(handle);
                cudaFree(d_A16);
                cudaFree(d_B16);
                break;
            }
            default:
                throw std::invalid_argument("Unknown kernel type: " + std::to_string(static_cast<int>(kernel)));
                break;
            }

    float ms_compute = 0.0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_compute, start_compute, stop_compute));
    double gflops = (2.0 * M * N * K) / (ms_compute * 1e6);
   
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
        std::cout << "------ [DEBUG RESULT] CORRECT for " << getKernelName(kernel) << std::endl;
    } else {
        std::cerr << "------ [DEBUG RESULT] INCORRECT for " << getKernelName(kernel) << std::endl;
    }
#endif

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start_compute));
    CHECK_CUDA(cudaEventDestroy(stop_compute));

    return {ms_compute , gflops, ms_memcpy};
}

void write_result_csv(const std::string &filename,
                         const std::string &kernelName,
                        int M, int N, int K, int runs, 
                        std::vector<RunData> &results)
{
    if (results.empty()) {
        std::cerr << "️[ERROR] result empty \n";
        return;
    }

    // Calcola statistiche
    auto [min_ms_compute_it, max_ms_compute_it] = std::minmax_element(results.begin(), results.end(),
        [](const RunData &a, const RunData &b) { return a.ms_compute < b.ms_compute; });
    
    auto [min_ms_memcpy_it, max_ms_memcpy_it] = std::minmax_element(results.begin(), results.end(),
        [](const RunData &a, const RunData &b) { return a.ms_memcpy < b.ms_memcpy; });

    auto [min_gflops_it, max_gflops_it] = std::minmax_element(results.begin(), results.end(),
        [](const RunData &a, const RunData &b) { return a.gflops < b.gflops; });

    double avg_ms_compute = 0.0, avg_ms_memcpy = 0.0, avg_gflops = 0.0, avg_power = 0.0;
    double min_power = results.front().min_power;
    double max_power = results.front().max_power;

    for (const auto &r : results) {
        avg_ms_compute += r.ms_compute;
        avg_ms_memcpy += r.ms_memcpy;
        avg_gflops += r.gflops;
        avg_power += r.avg_power;
        min_power = std::min(min_power, r.min_power);
        max_power = std::max(max_power, r.max_power);
    }

    avg_ms_compute /= results.size();
    avg_ms_memcpy /= results.size();
    avg_gflops /= results.size();
    avg_power /= results.size();
   
    auto timestamp = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    std::ofstream file(filename, std::ios::app); /* append the file */
    if (!file.is_open()) {
        std::cerr << "[ERROR]: can't open " << filename << "\n";
        return;
    }

    /* if the file is empty */
    file.seekp(0, std::ios::end);
    if (file.tellp() == 0) {
        file << "Timestamp,Kernel,M,N,K,Runs,"
             << "Avg_Compute_ms,Min_Compute_ms,Max_Compute_ms,"
             << "Avg_MemCpy_ms,Min_MemCpy_ms,Max_MemCpy_ms,"
             << "Avg_GFLOPS,Min_GFLOPS,Max_GFLOPS,"
             << "Avg_Power_W,Min_Power_W,Max_Power_W\n";
    }

    file << timestamp << ","
         << kernelName << ","
         << M << "," << N << "," << K << ","
         << runs << ","
         << std::fixed << std::setprecision(3)
         << avg_ms_compute << "," << min_ms_compute_it->ms_compute << "," << max_ms_compute_it->ms_compute << ","
         << avg_ms_memcpy << "," << min_ms_memcpy_it->ms_memcpy << "," << max_ms_memcpy_it->ms_memcpy << ","
         << avg_gflops << "," << min_gflops_it->gflops << "," << max_gflops_it->gflops << ","
         << avg_power << "," << min_power << "," << max_power
         << "\n";

    file.close();

    std::cout << "\nResults (" << kernelName << ")\n";
    std::cout << "Matrix: " << M << "x" << N << "x" << K << "\n";
    std::cout << runs << " run(s)\n";
    std::cout << "Timestamp: " << timestamp << "\n\n";
    std::cout << "avg compute ms: " << avg_ms_compute << " ms (min " << min_ms_compute_it->ms_compute << ", max " << max_ms_compute_it->ms_compute << ")\n";
    std::cout << "avg memcpy ms : " << avg_ms_memcpy << " ms (min " << min_ms_memcpy_it->ms_memcpy << ", max " << max_ms_memcpy_it->ms_memcpy << ")\n";
    std::cout << "avg GFLOPS : " << avg_gflops << " GF (min " << min_gflops_it->gflops << ", max " << max_gflops_it->gflops << ")\n";
    std::cout << "avg Power : " << avg_power << " W (min " << min_power << ", max " << max_power << ")\n";
    std::cout << "--------------------------------------------\n";
}
                        
void benchmark_gem(KernelType kernel, const int M, const int N, const int K, dim3 blockSize, dim3 gridSize, const int runs) {
    std::cout << "=========== Benchmark: " << getKernelName(kernel) 
              << " | " << M << "x" << N 
              << " (K=" << K << "), " 
              << runs << " runs\n";

    double avg_power = 0.0,
        min_power = 0.0,
        max_power = 0.0;

    /* thread for gpu power sampling */
    GpuPowerSampler sampler(0, 100);

    /* results vector for each run */
    std::vector<RunData> results;
    results.reserve(runs);


    for (int i = 0; i < runs; i++) {

        sampler.start();

        /* running kernel */
        TestResult r = run_gem(kernel, M, N, K, blockSize, gridSize);

        sampler.stop();

        avg_power = sampler.averagePower();
        min_power = sampler.minPower();
        max_power = sampler.maxPower();

        results.push_back({r.ms_compute, r.ms_memcpy, r.gflops, avg_power, min_power, max_power});

#ifdef DEBUG
        std::cout << "[DEBUG]: Run " << i+1 << ": "
                  << "Compute " << r.ms_compute << " ms, "
                  << "MemCpy " << r.ms_memcpy << " ms, " 
                  << r.gflops << " GFLOPS, "
                  << " | Power (avg/min/max): " 
                  << avg_power << "/" << min_power << "/" << max_power << " W\n";
#endif
    }

    write_result_csv("gem_benchmark_results.csv", getKernelName(kernel), M, N, K, runs, results);
}
