#ifndef SIMPLE_MEM_HPP
#define SIMPLE_MEM_HPP

#include <cuda_runtime.h>

__global__ void copy_kernel(float* C, const float* A, const int n);
__global__ void scale_kernel(float* C, const float* A, float alpha, const int n);
__global__ void add_kernel(float* C, const float* A, const float* B, const int n);
__global__ void triad_kernel(float* C, const float* A, const float* B, float alpha, const int n);

#endif