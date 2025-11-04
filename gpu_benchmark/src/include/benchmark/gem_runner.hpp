#ifndef GEM_RUNNER_HPP
#define GEM_RUNNER_HPP

#include "../kernels/simple_gem.hpp"
#include "../../kernels/tile_gem.hpp" 
#include "../../kernels/dense_gem.hpp"

#include "../types.hpp"
#include "../help_func.hpp"
#include "../config.hpp"

#include <string>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <functional>
#include <nvml.h>
#include <thread>
#include <fstream>
#include <iomanip>


void benchmark_gem(KernelType kernel, const int M, const int N, const int K, 
                   dim3 blockSize, dim3 gridSize, const int runs);

#endif