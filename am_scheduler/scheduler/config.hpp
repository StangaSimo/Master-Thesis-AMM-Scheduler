#ifndef CONFIG_H
#define CONFIG_H

/********** debug ************/
#define DEBUG
#define ENABLE_PROFILING           /* profiler for timing the scheduler */
#define ENABLE_INTEL_POWER_PROFILE /* profiling the power consumption with intel rapl */

/********** tests ************/
#define M_ 1024
#define N_ 1024
#define K_ 512 
#define N_MATRIX 800 

#define M_split 14000 /* big matrix split size, max is MAX_SIZE/4 */
#define N_split 4000 
#define K_split 4000 

/********** scheduler ************/
#define MAX_SIZE 4092

/* large matrix multiplication */
//#define JIT_MS_SYCL 200            /* mitigate jit time for sycl */
//#define JIT_MS_OV 100            /* mitigate jit time for openvino */

/* streaming */
#define JIT_MS_SYCL 80
#define JIT_MS_OV 120

enum {BATCH_SIZE = 30};       /* static partitioning batch size */
enum {SPLIT_MATRIX_ITERATION = 3};       /* static partitioning batch size */

/********** buffers ************/
//#define SLEEP
enum {GET_SLEEP = 500};       /* shared buffer sleep time */
enum {PUT_SLEEP = 500};       /* 5 ms */
enum {BUFFER_LENGHT = 1024};  /* shared buffer length */

/********** benchmark ************/
enum {STEP_SIZE = 512};       /* benchmark for csv step size */
enum {STEP_TOTAL = 8};       /* STEP_SIZE * STEP_TOTAL */

/********** CPU ************/
enum {N_CORES = 2};       /* number of cores for cpu gemm */

#endif
