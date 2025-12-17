#ifndef SLEEPBUFFER_H
#define SLEEPBUFFER_H

#include <memory>
#include <chrono>
#include <unistd.h>

enum {GET_SLEEP = 10000}; /* 10 ms */
enum {PUT_SLEEP = 10000}; /* 10 ms */

using namespace std;

typedef struct {
    void *A;
    void *B;
    void *C;
    
    int type; /* 1 float, 2 half, 3 8bit*/
    int M; 
    int N; 
    int K; 

    /* benchmarking */
    chrono::high_resolution_clock::time_point start_time;
    chrono::high_resolution_clock::time_point end_time;
} task;


/* basic implementation of a simple ring buffer with blocking function */
class SharedBuffer {
    private: 
        unique_ptr<task*[]> data; /* unique pointer so delete is not needed */
        size_t size = 0;
        
        size_t write_i = 0;
        size_t read_i = 0; 

    public:
        /* constructor */
        SharedBuffer(size_t s) : size(s), data(new task*[s]) {
        }

        /* destructur */
        ~SharedBuffer() {
        }

        /* put one task pointer inside, sleep if full*/
        void put(task* ele) {
            while (((write_i + 1) % size) == read_i) {
                usleep(PUT_SLEEP);
            }
            data[write_i] = ele; 
            write_i = (write_i + 1) % size;
        }

        /* get one task pointer sleep if empty, return to the caller after
         * 5 attempt*/
        task* get() {
            int c = 0;

            while (write_i == read_i){
                c++;
                usleep(GET_SLEEP);
                if (c >= 5) {return nullptr;}
            }

            task* ele = data[read_i];
            read_i = (read_i + 1) % size;

            return ele; 
        }

        bool is_empty() {
            if (write_i == read_i)
                return true;
            else 
                return false;
        }
};
#endif
