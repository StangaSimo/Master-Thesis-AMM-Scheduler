#ifndef CONFIG_H
#define CONFIG_H

/********** debug ************/
//#define DEBUG
#define ENABLE_PROFILING /* profiler for timing the scheduler */
#define ENABLE_INTEL_POWER_PROFILE /* profiling the power consumption with intel rapl */


/********** tests ************/
#define M_ 1920
#define N_ 1920
#define K_ 1024 
#define N_MATRIX 300 
#define MAX_SIZE 4092

enum {BATCH_SIZE = 30};       /* static partitioning batch size */

#define M_split 14000 /* big matrix split size, max is MAX_SIZE/4 */
#define N_split 4000 
#define K_split 4000 

enum {SPLIT_MATRIX_ITERATION = 3};       /* static partitioning batch size */

/********** buffers ************/
//#define SLEEP
enum {GET_SLEEP = 500};       /* shared buffer sleep time */
enum {PUT_SLEEP = 500};       /* 5 ms */
enum {BUFFER_LENGHT = 1024};  /* shared buffer length */

/********** benchmark ************/
enum {STEP_SIZE = 256};       /* benchmark for csv step size */
enum {STEP_TOTAL = 12};       /* STEP_SIZE * STEP_TOTAL */

/********** CPU ************/
enum {N_CORES = 6};       /* number of cores for cpu gemm */

#endif
