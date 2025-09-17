#include "../include/kernels/dense_gem.hpp"

// transfer float4
template <
    const int BLOCK_SIZE_M,  // width of block of C that each thread block calculate
    const int BLOCK_SIZE_K,  // height of block of A that each thread block load into shared memory
    const int BLOCK_SIZE_N,  // height of block of C that each thread block calculate
    const int THREAD_SIZE_Y, // height of block of C that each thread calculate
    const int THREAD_SIZE_X  // width of block of C that each thread calculate
    > 
__global__ void dense_gem_kernel( 
    float * __restrict__ A,
    float * __restrict__ B,
    float * __restrict__ C, 
    const int M,
    const int K,
    const int N,
    float alpha,
    float beta
    ) {
    
    const int bszx = BLOCK_SIZE_N / THREAD_SIZE_X;
    const int bszy = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int THREAD_NUM_PER_BLOCK = bszy * bszx;

    const int tid = threadIdx.y * bszx + threadIdx.x;

    __shared__ float As[BLOCK_SIZE_M][BLOCK_SIZE_K]; 
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];
    float accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {0};
    
    const int A_TILE_ROW = tid / BLOCK_SIZE_K;
    const int B_TILE_ROW = tid / BLOCK_SIZE_N;

    const int A_TILE_COL = tid % BLOCK_SIZE_K;
    const int B_TILE_COL = tid % BLOCK_SIZE_N;
    
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_N;

    const int A_S = BLOCK_SIZE_M / THREAD_SIZE_Y;
    const int B_S = BLOCK_SIZE_N / THREAD_SIZE_X;

    for (int tile_idx = 0 ; tile_idx < K ; tile_idx += BLOCK_SIZE_K) {

        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_M ; i += A_TILE_ROW_STRIDE) {
            const int row = BLOCK_SIZE_M * blockIdx.y + i + A_TILE_ROW ;
            const int col = A_TILE_COL + tile_idx;
            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1) {
                As[i + A_TILE_ROW ][A_TILE_COL] = row < M && col < K ? A[OFFSET(
                    row,
                    col, 
                    K )] : 0;
            } else {
                As[i + A_TILE_ROW ][A_TILE_COL] = A[OFFSET(
                    row, 
                    col, 
                    K )];
            }
        }

        #pragma unroll
        for ( int i = 0 ; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {
            const int row = tile_idx + i + B_TILE_ROW;
            const int col = B_TILE_COL + BLOCK_SIZE_N * blockIdx.x;
            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1) {
                Bs[i + B_TILE_ROW][B_TILE_COL] = row < K && col < N ? B[OFFSET(
                    row, 
                    col, 
                    N )] : 0;
            } else {
                Bs[i + B_TILE_ROW][B_TILE_COL] = B[OFFSET(
                    row, 
                    col, 
                    N )];
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++ k) {
            #pragma unroll
            for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
                #pragma unroll
                for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
                    accum[thread_y][thread_x] += As[thread_y * A_S + threadIdx.y][k] * Bs[k][thread_x * B_S + threadIdx.x];
                }
            }
            
        }
        __syncthreads();
    }

    // store back to C
    #pragma unroll
    for (int thread_y = 0; thread_y < THREAD_SIZE_Y; ++thread_y) {
        #pragma unroll
        for (int thread_x = 0; thread_x < THREAD_SIZE_X; ++thread_x) {
            const int row = BLOCK_SIZE_M * blockIdx.y + thread_y * A_S + threadIdx.y;
            const int col = BLOCK_SIZE_N * blockIdx.x + thread_x * B_S + threadIdx.x;
            if (blockIdx.x == gridDim.x -1 || blockIdx.y == gridDim.y - 1) {
                if (row < M && col < N) {
                    C[OFFSET(row, col, N)] = C[OFFSET(row, col, N)] * beta + accum[thread_y][thread_x] * alpha;
                }
            } else {
                C[OFFSET(row, col, N)] = C[OFFSET(row, col, N)] * beta + accum[thread_y][thread_x] * alpha;
            }
        }
    }
}