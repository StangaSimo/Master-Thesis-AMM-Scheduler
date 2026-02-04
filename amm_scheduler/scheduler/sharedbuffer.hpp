#ifndef SLEEPBUFFER_H
#define SLEEPBUFFER_H

#include <memory>
#include <unistd.h>
#include <atomic>
#include "tasks.hpp"
#include "config.hpp"

using namespace std;

/* basic implementation of a simple ring buffer with blocking function */
class SharedBuffer {
    private: 
        unique_ptr<task*[]> data;
        size_t size = 0;
        
        alignas(64) std::atomic<size_t> write_i{0};
        alignas(64) std::atomic<size_t> read_i{0};

    public:
        /* constructor */
        SharedBuffer(size_t s) : size(s), data(new task*[s]) {
        }

        /* destructur */
        ~SharedBuffer() {
        }

        /* put one task pointer inside, sleep if full*/
        void put(task* ele) {

#ifdef SLEEP
            while (((write_i + 1) % size) == read_i) {
                usleep(PUT_SLEEP);
            }
#else
            while (((write_i + 1) % size) == read_i) {}
#endif
            data[write_i] = ele; 
            write_i = (write_i + 1) % size;
        }

        /* get one task pointer sleep if empty, return to the caller after
         * 5 attempt*/
        task* get() {
            int c = 0;

#ifdef SLEEP
            while (write_i == read_i){
                c++;
                usleep(GET_SLEEP);
                if (c >= 5) {return nullptr;}
            }
#else
            while (write_i == read_i){
                c++;
                if (c >= 50) {return nullptr;}
            }
#endif
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
