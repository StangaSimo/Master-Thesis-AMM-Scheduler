#ifndef MEM_RUNNER_HPP
#define MEM_RUNNER_HPP

#include "../kernels/simple_mem.hpp"
#include "../types.hpp"
#include "../help_func.hpp"
#include "../config.hpp"

#include <string>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <functional>

void benchmark_mem(KernelType kernel, size_t bytes_per_elem, const int N, 
                   dim3 blockSize, dim3 gridSize, float alpha, const int runs);

#endif