#ifndef MEM_RUNNER_HPP
#define MEM_RUNNER_HPP

#include <string>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <functional>

#include "../kernels/simple_mem.hpp"
#include "../types.hpp"
#include "../help_func.hpp"

void benchmark_mem(const std::string& name, size_t bytes_per_elem, int n, 
                   dim3 blockSize, dim3 gridSize, float alpha, int runs);

#endif