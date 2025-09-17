#ifndef SIMPLE_MEM_HPP
#define SIMPLE_MEM_HPP

#include <cuda_runtime.h>

__global__ void copy_kernel(float* C, const float* A, int n);
__global__ void scale_kernel(float* C, const float* A, float alpha, int n);
__global__ void add_kernel(float* C, const float* A, const float* B, int n);
__global__ void triad_kernel(float* C, const float* A, const float* B, float alpha, int n);

#endif