#ifndef HELP_HPP
#define HELP_HPP

#include <iostream> 
#include "config.hpp"
#include "types.hpp"

#define CHECK_CUDA(func) {				    \
    cudaError_t e = (func);			        \
    if(e != cudaSuccess)			        \
        printf ("%s %d CUDA ERROR: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e)); \
}

#define CHECK_CUBLAS(func) {                      \
    cublasStatus_t e = (func);                    \
    if(e != CUBLAS_STATUS_SUCCESS)                \
        printf ("%s %d CUBLAS ERROR: ", __FILE__, __LINE__); \
}

inline const char* getKernelName(KernelType type) {
    switch(type) {
        case KernelType::NAIVE:     return "Naive";
        case KernelType::TILED:     return "Tiled";
        case KernelType::DENSE:     return "Dense";
        case KernelType::ADD:       return "Add";
        case KernelType::TRIAD:     return "Triad";
        case KernelType::SCALE:     return "Scale";
        case KernelType::COPY:      return "Copy";
        case KernelType::CUBLAS:    return "Cublas";
        case KernelType::TENSOR:    return "Tensor";
        default:                    return "Unknown";
    }
}

struct GPUInfo {
    int device_id;
    std::string name;
    int compute_capability_major;
    int compute_capability_minor;
    int max_threads_per_block;
    int max_threads_per_multiprocessor;
    int multiprocessor_count;
    int warp_size;
    size_t total_global_mem;
    size_t shared_mem_per_block;
    int regs_per_block;
};

inline GPUInfo getGPUInfo(int device_id = -2) {
    GPUInfo info;
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    
    info.device_id = device_id;
    info.name = prop.name;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.max_threads_per_block = prop.maxThreadsPerBlock;
    info.max_threads_per_multiprocessor = prop.maxThreadsPerMultiProcessor;
    info.multiprocessor_count = prop.multiProcessorCount;
    info.warp_size = prop.warpSize;
    info.total_global_mem = prop.totalGlobalMem;
    info.shared_mem_per_block = prop.sharedMemPerBlock;
    info.regs_per_block = prop.regsPerBlock;
    
    return info;
}

inline void printGPUInfo(const GPUInfo& info) {
    std::cout << "=== GPU Information ===" << std::endl;
    std::cout << "Device: " << info.device_id << " - " << info.name << std::endl;
    std::cout << "Compute Capability: " << info.compute_capability_major  << "." << info.compute_capability_minor << std::endl;
    std::cout << "Max Threads per Block: " << info.max_threads_per_block << std::endl;
    std::cout << "Multiprocessors: " << info.multiprocessor_count << std::endl;
    std::cout << "Warp Size: " << info.warp_size << std::endl;
    std::cout << "Total Global Memory: " << info.total_global_mem / (1024*1024) << " MB" << std::endl;
    std::cout << "Shared Memory per Block: " << info.shared_mem_per_block << " bytes" << std::endl;
    std::cout << "Registers per Block: " << info.regs_per_block << std::endl;
    std::cout << "=======================" << std::endl << std::endl;
}

#ifdef CHECK_RESULT
//#include <cmath>
//#include <algorithm>

inline void cpu_gemm(const float* A, const float* B, float* C, 
                    int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

inline bool compare_matrix(const float* gpu_C, const float* cpu_C, 
                          int size, float tolerance = 1e-3f) {
    for (int i = 0; i < size; ++i) {
        float diff = std::abs(gpu_C[i] - cpu_C[i]);
        float max_val = std::max(std::abs(gpu_C[i]), std::abs(cpu_C[i]));
        
        if (diff > tolerance && diff > tolerance * max_val) {
            std::cerr << "------ [DEBUG RESULT] Mismatch at index " << i 
                      << ": GPU=" << gpu_C[i] 
                      << ", CPU=" << cpu_C[i] 
                      << ", diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}


#endif

#endif
