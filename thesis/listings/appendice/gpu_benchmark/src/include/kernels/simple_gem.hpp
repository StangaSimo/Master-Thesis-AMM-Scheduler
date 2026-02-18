#ifndef SIMPLE_GEM_HPP
#define SIMPLE_GEM_HPP

__global__ void simple_gemm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int M, const int N, const int K);

#endif