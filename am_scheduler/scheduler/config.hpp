#ifndef CONFIG_H
#define CONFIG_H

/********** debug ************/
//#define DEBUG
#define ENABLE_PROFILING

/********** tests ************/
#define M_ 1024
#define N_ 1024
#define K_ 512 
#define N_MATRIX 300
#define MAX_SIZE 3200

enum {BATCH_SIZE = 30};       /* static partitioning batch size */

/********** buffers ************/
//#define SLEEP
enum {GET_SLEEP = 500};       /* shared buffer sleep time */
enum {PUT_SLEEP = 500};       /* 5 ms */
enum {BUFFER_LENGHT = 512};  /* shared buffer length */

/********** benchmark ************/
enum {STEP_SIZE = 256};       /* benchmark for csv step size */
enum {STEP_TOTAL = 12};       /* STEP_SIZE * STEP_TOTAL */

#endif
