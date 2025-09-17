#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <functional>

#include "include/benchmark/gem_runner.hpp"
#include "include/benchmark/mem_runner.hpp"

#define DEBUG
#define GEM
#define MEM

#define TILE 16
#define RUNS 5
#define M_m 4096
#define N_m 4096
#define K_m 2048

int main() {

#ifdef GEM
    dim3 blockNaive(16, 16);

    dim3 gridNaive((N_m + blockNaive.x - 1) / blockNaive.x,
                   (M_m + blockNaive.y - 1) / blockNaive.y);

    dim3 blockTiled(TILE, TILE);
    dim3 gridTiled((N_m + TILE - 1) / TILE,
                   (M_m + TILE - 1) / TILE);

    benchmark_gem("Naive", M_m, N_m, K_m, blockNaive, gridNaive, RUNS);
    benchmark_gem("Tiled", M_m, N_m, K_m, blockTiled, gridTiled, RUNS);
#endif 

#ifdef MEM
    int N = 1<<24; // ~16M elements
    dim3 block(256), grid((N+255)/256);

    float alpha = 2.5f;

    benchmark_mem("Copy", 2*sizeof(float), N, block, grid, alpha, RUNS);
    benchmark_mem("Scale", 2*sizeof(float), N, block, grid, alpha, RUNS);
    benchmark_mem("Add", 3*sizeof(float), N, block, grid, alpha, RUNS);
    benchmark_mem("Triad", 3*sizeof(float), N, block, grid, alpha, RUNS);
    
#endif 

    return 0;
}