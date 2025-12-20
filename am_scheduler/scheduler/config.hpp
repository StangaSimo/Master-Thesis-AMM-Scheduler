#ifndef CONFIG_H
#define CONFIG_H

//#define DEBUG

//#define SLEEP
#define M_ 2048
#define N_ 2048
#define K_ 1024 
#define N_MATRIX 100 

enum {GET_SLEEP = 500};       /* shared buffer sleep time */
enum {PUT_SLEEP = 500};       /* 5 ms */
enum {BUFFER_LENGHT = 1024};  /* shared buffer length */

enum {STEP_SIZE = 512};       /* benchmark for csv step size */
enum {BATCH_SIZE = 10};       /* static partitioning batch size */


#endif
