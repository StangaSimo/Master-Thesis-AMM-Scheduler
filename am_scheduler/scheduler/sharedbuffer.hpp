#ifndef SLEEPBUFFER_H
#define SLEEPBUFFER_H

#include <memory>
#include <unistd.h>

typedef struct {
    void *A;
    void *B;
    void *C;

    /* 1 float, 2 half */
    int type; 
    int M; 
    int N; 
    int K; 
} task;

using namespace std;

/* basic implementation of circular buffer that do not wait */
class SharedBuffer {
    private: 
        unique_ptr<task*[]> data; /* unique pointer so delete is not needed */
        size_t size = 0;
        
        /* both indexes */
        size_t write_i = 0;
        size_t read_i = 0; 

    public:
        /* constructor */
        SharedBuffer(size_t s) : size(s), data(new task*[s]) {
        }

        /* destructur */
        //~ SharedBuffer() {
        //}

        /* blocking function */
        void put(task* ele) {
            /* buffer full */
            while (((write_i + 1) % size) == read_i) {
                usleep(100000);
            }
            data[write_i] = ele; 
            write_i = (write_i + 1) % size;
        }

        task* get() {
            if (write_i == read_i) {
                return NULL;
            }

            task* ele = data[read_i];
            read_i = (read_i + 1) % size;

            return ele; 
        }
};

#endif
