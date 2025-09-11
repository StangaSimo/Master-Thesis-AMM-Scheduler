#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <functional>

#define TILE 16
#define RUNS 5
#define M_m 4096
#define N_m 4096
#define K_m 2048
#define DEBUG

__global__ void simple_gemm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void gemm_tiled_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ float As[TILE][TILE + 1];
    __shared__ float Bs[TILE][TILE + 1];

    float acc = 0.0f;
    int numTiles = (K + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

        Bs[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

__global__ void copy_kernel(float* C, const float* A, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i];
}

/* some computation */
__global__ void scale_kernel(float* C, const float* A, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = alpha * A[i];
}

/* write and read  */
__global__ void add_kernel(float* C, const float* A, const float* B, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

/* write read and some computation */
__global__ void triad_kernel(float* C, const float* A, const float* B, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + alpha * B[i];
}

struct TestResult {
    double ms;
    double gflops;
};

typedef void (*KernelFunc)(const float*, const float*, float*, int, int, int);

TestResult run_gemm(KernelFunc kernel, int M, int N, int K,
                           dim3 blockSize, dim3 gridSize) {
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (auto& x : h_A) x = dis(gen);
    for (auto& x : h_B) x = dis(gen);
    for (auto& x : h_C) x = 0.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warmup
    kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

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

void benchmark_gem(const std::string& name, KernelFunc kernel, int M, int N, int K, dim3 blockSize, dim3 gridSize) {
    std::cout << "Benchmark: " << name << " | "
              << M << "x" << N << " (K=" << K << "), "
              << RUNS << " runs\n";

    double sum_ms = 0.0, sum_gflops = 0.0;

    for (int i = 0; i < RUNS; i++) {
        TestResult r = run_gemm(kernel, M, N, K, blockSize, gridSize);
        sum_ms += r.ms;
        sum_gflops += r.gflops;
#ifdef DEBUG
        std::cout << "Run " << i+1 << ": "
                  << r.ms << " ms, "
                  << r.gflops << " GFLOPS\n";
#endif 
    }

    std::cout << "-----------------------------\n";
    std::cout << "Average time: " << sum_ms / RUNS << " ms\n";
    std::cout << "Average GFLOPS: " << sum_gflops / RUNS << " GFLOPS\n\n";
}

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

void benchmark_mem(const std::string& name, size_t bytes_per_elem, int n, dim3 blockSize, dim3 gridSize , float alpha) {
    std::cout << "Benchmark: " << name << " | "
              << n << " elements, " << RUNS << " runs\n";

    double sum_ms = 0.0, sum_gbs = 0.0;
    for (int i = 0; i < RUNS; i++) {
        TestResult r = run_mem(name, bytes_per_elem, n, blockSize, gridSize, alpha);
        sum_ms += r.ms;
        sum_gbs += r.gflops;
        std::cout << "Run " << i+1 << ": "
                  << r.ms << " ms, "
                  << r.gflops << " GB/s\n";
    }
    std::cout << "Average time: " << sum_ms / RUNS << " ms\n";
    std::cout << "Average Bandwidth: " << sum_gbs / RUNS << " GB/s\n\n";
}


int main() {

    dim3 blockNaive(16, 16);
    //std::cout << "COSI: " << blockNaive.x << "\n";
    dim3 gridNaive((N_m + blockNaive.x - 1) / blockNaive.x,
                   (M_m + blockNaive.y - 1) / blockNaive.y);

    dim3 blockTiled(TILE, TILE);
    dim3 gridTiled((N_m + TILE - 1) / TILE,
                   (M_m + TILE - 1) / TILE);

    benchmark_gem("Naive GEMM", simple_gemm_kernel, M_m, N_m, K_m, blockNaive, gridNaive);
    benchmark_gem("Tiled GEMM", gemm_tiled_kernel, M_m, N_m, K_m, blockTiled, gridTiled);

    int N = 1<<27; // ~16M elements
    dim3 block(256), grid((N+255)/256);

    float alpha = 2.5f;

    benchmark_mem("Copy", 2*sizeof(float), N, block, grid, alpha);
    benchmark_mem("Scale", 2*sizeof(float), N, block, grid, alpha);
    benchmark_mem("Add", 3*sizeof(float), N, block, grid, alpha);
    benchmark_mem("Triad", 3*sizeof(float), N, block, grid, alpha);

    return 0;
}