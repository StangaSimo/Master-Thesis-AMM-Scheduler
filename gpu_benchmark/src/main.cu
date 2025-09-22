#include "include/benchmark/gem_runner.hpp"
#include "include/benchmark/mem_runner.hpp"

int main() {

    /* TODO: GPU consumption and efficiency?  SIIIII*/
    /* TODO: Sparse gem?  SIIII */ 
    /* TODO: more than one precision? solo double e float */

    /* Open vino on windoows for gem? with efficency and gflops SIIIIIIIII */

    /* init: TENSORE CORE */

    printGPUInfo(getGPUInfo(0));

#ifdef GEM
    dim3 blockNaive(16, 16);
    dim3 gridNaive((N_SIZE + blockNaive.x - 1) / blockNaive.x, (M_SIZE + blockNaive.y - 1) / blockNaive.y);

    benchmark_gem(KernelType::NAIVE, M_SIZE, N_SIZE, K_SIZE, blockNaive, gridNaive, RUNS);

    dim3 blockTiled(TILE_SIZE, TILE_SIZE);
    dim3 gridTiled((N_SIZE + TILE_SIZE - 1) / TILE_SIZE, (M_SIZE + TILE_SIZE - 1) / TILE_SIZE);

    benchmark_gem(KernelType::TILED, M_SIZE, N_SIZE, K_SIZE, blockTiled, gridTiled, RUNS);

    dim3 blockDense(DENSE_BLOCK_N / DENSE_THREAD_X, DENSE_BLOCK_M / DENSE_THREAD_Y);
    dim3 gridDense(N_SIZE/ DENSE_BLOCK_N, M_SIZE / DENSE_BLOCK_M);

    if (N_SIZE % DENSE_BLOCK_N != 0)
        gridDense.x++;
    if (M_SIZE % DENSE_BLOCK_M != 0)
        gridDense.y++;

    benchmark_gem(KernelType::DENSE, M_SIZE, N_SIZE, K_SIZE, blockDense, gridDense, RUNS);

    benchmark_gem(KernelType::CUBLAS, M_SIZE, N_SIZE, K_SIZE, NULL, NULL, RUNS);
#endif 

#ifdef MEM
    int N = 1<<24; // ~16M elements
    dim3 block(256), grid((N+255)/256);

    float alpha = 2.5f;

    benchmark_mem(KernelType::COPY, 2*sizeof(float), N, block, grid, alpha, RUNS);
    benchmark_mem(KernelType::SCALE, 2*sizeof(float), N, block, grid, alpha, RUNS);
    benchmark_mem(KernelType::ADD, 3*sizeof(float), N, block, grid, alpha, RUNS);
    benchmark_mem(KernelType::TRIAD, 3*sizeof(float), N, block, grid, alpha, RUNS);
    
#endif 

    return 0;
}