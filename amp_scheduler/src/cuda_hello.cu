#include "amp_scheduler/cuda_hello.hpp"
#include <iostream>
#include <cuda_runtime.h>

__global__ void addTenKernel(int *val) {
    *val = *val + 10;
}

void runCudaHello() {
    int host_val = 5;
    int *device_val;

    cudaMalloc(&device_val, sizeof(int));

    cudaMemcpy(device_val, &host_val, sizeof(int), cudaMemcpyHostToDevice);

    addTenKernel<<<1, 1>>>(device_val);
    
    cudaDeviceSynchronize();

    cudaMemcpy(&host_val, device_val, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "[CUDA] Hello! Il risultato del kernel e': " << host_val << std::endl;

    cudaFree(device_val);
}