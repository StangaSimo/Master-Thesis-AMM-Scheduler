#ifndef TITLE_GEM_HPP
#define TITLE_GEM_HPP

#include <cuda_runtime.h>

template <const int TILE>
__global__ void title_gem_kernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int M, int N, int K);

#endif