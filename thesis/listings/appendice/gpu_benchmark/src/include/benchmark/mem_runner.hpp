#ifndef MEM_RUNNER_HPP
#define MEM_RUNNER_HPP

#include <cuda_runtime.h>
#include <nvml.h>

#include "../kernels/simple_mem.hpp"
#include "../types.hpp"
#include "../help_func.hpp"
#include "../config.hpp"

#include <string>
#include <iostream>
#include <random>
#include <vector>
#include <functional>
#include <thread>
#include <fstream>
#include <iomanip>
#include <algorithm>

void benchmark_mem(KernelType kernel, size_t bytes_per_elem, const int N, 
                   dim3 blockSize, dim3 gridSize, float alpha, const int runs);

#endif