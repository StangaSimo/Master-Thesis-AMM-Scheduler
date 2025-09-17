#ifndef DENSE_GEM_HPP
#define DENSE_GEM_HPP

#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

template <const int BLOCK_SIZE_M, const int BLOCK_SIZE_K, const int BLOCK_SIZE_N,
          const int THREAD_SIZE_Y, const int THREAD_SIZE_X>
__global__ void dense_gem_kernel(float* __restrict__ A,
                                 float* __restrict__ B,
                                 float* __restrict__ C,
                                 const int M, const int K, const int N,
                                 float alpha, float beta);

#endif