#ifndef SIMPLE_GEM_HPP
#define SIMPLE_GEM_HPP

#include <cuda_runtime.h>

__global__ void simple_gemm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K);

#endif