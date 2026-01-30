#ifndef CONFIG_H
#define CONFIG_H

#include <string>

/********** debug ************/
//#define DEBUG
#define ENABLE_PROFILING           /* profiler for timing the scheduler */
#define ENABLE_INTEL_POWER_PROFILE /* profiling the power consumption with intel rapl */

/********** tests ************/
#define M_ 4092
#define N_ 4092
#define K_ 4092 
#define N_MATRIX 100 

inline int BATCH_SIZE = 30;
inline int BATCH_SIZE_HETERO = 40;
inline std::string csvname = "bin/csv/results.csv";

#define M_split 14000 /* big matrix split size, max is MAX_SIZE/4 */
#define N_split 4000 
#define K_split 4000 

/********** scheduler ************/
#define MAX_SIZE 4092

#define JIT_MS_SYCL 50  /* mitigate jit time for sycl */

//enum {BATCH_SIZE = 50};                  /* static partitioning batch size */
//enum {BATCH_SIZE_HETERO = 40};           
enum {SPLIT_MATRIX_ITERATION = 3};       

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
