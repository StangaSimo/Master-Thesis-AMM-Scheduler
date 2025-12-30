#ifndef CONFIG_H
#define CONFIG_H

//#define DEBUG

/********** tests ************/
#define M_ 1024
#define N_ 1024
#define K_ 512 
#define N_MATRIX 100
#define MAX_SIZE 2048

/********** buffers ************/
//#define SLEEP
enum {GET_SLEEP = 500};       /* shared buffer sleep time */
enum {PUT_SLEEP = 500};       /* 5 ms */
enum {BUFFER_LENGHT = 512};  /* shared buffer length */


enum {STEP_SIZE = 256};       /* benchmark for csv step size */
enum {STEP_TOTAL = 12};       /* STEP_SIZE * STEP_TOTAL */

enum {BATCH_SIZE = 80};       /* static partitioning batch size */

#endif
