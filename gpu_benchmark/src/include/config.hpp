#ifndef CONFIG_HPP
#define CONFIG_HPP

#define DENSE_BLOCK_M 96
#define DENSE_BLOCK_K 32  
#define DENSE_BLOCK_N 64
#define DENSE_THREAD_Y 6
#define DENSE_THREAD_X 4
#define DENSE_ALPHA 1.0f
#define DENSE_BETA 0.0f

/* Debug mode */
//#define DEBUG
//#define CHECK_RESULT
#define GEM
#define MEM

#define RUNS 5
#define TILE_SIZE 16
#define M_SIZE 2048
#define N_SIZE 2048
#define K_SIZE 512

#endif