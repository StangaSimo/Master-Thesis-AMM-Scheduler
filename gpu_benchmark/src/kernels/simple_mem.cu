#include "../include/kernels/simple_mem.hpp"

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