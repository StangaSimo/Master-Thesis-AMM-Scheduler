#ifndef GEM_RUNNER_HPP
#define GEM_RUNNER_HPP

#include <string>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <functional>

#include "../kernels/simple_gem.hpp"
#include "../kernels/title_gem.hpp" 
#include "../kernels/dense_gem.hpp"
#include "../types.hpp"
#include "../help_func.hpp"



void benchmark_gem(const std::string& kernel, int M, int N, int K, 
                   dim3 blockSize, dim3 gridSize, int runs);

#endif