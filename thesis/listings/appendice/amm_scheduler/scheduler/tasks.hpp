#ifndef TASKS_H
#define TASKS_H

#include <cstdint>
#include <cstring>
#include <chrono>
#include <iostream>
#include <random>

#include "config.hpp"

using namespace std;

/* backend_type */
enum BT : size_t { 
    CORDINATOR,
    CUDA, 
    SYCL, 
    OPENVINO,
    OPENBLAS,
    COUNT,
};

enum Type : int { 
    FLOAT,
    HALF,
    UINT8,
};

typedef struct {
    void *A;
    void *B;
    void *C;
    
    Type type; /* 1 float, 2 half, 3 8bit */
    int M; 
    int N; 
    int K; 

    /* benchmarking */
    chrono::high_resolution_clock::time_point start_time;
    chrono::high_resolution_clock::time_point end_time;
} task;

inline float half_to_float(uint16_t h);
inline uint16_t float_to_half(float f);
inline void init_16bit(void* A, void* B, void* C, int M, int N, int K);
inline void init_32bit(void* A, void* B, void* C, int M, int N, int K);
inline void clean_tasks(task* array_task, size_t n_task);

inline task* init_tasks(size_t n_task, int m, int n, int k, Type t) {
        task* array_task = new task[n_task];

        for (int i = 0; i < n_task; i++) {
            array_task[i].M = m;
            array_task[i].N = n;
            array_task[i].K = k;
            array_task[i].type = t;

            if (t == Type::FLOAT) {
                array_task[i].A = new float[m * k]; 
                array_task[i].B = new float[k * n];
                array_task[i].C = new float[m * n];

                init_32bit(array_task[i].A,array_task[i].B,array_task[i].C, m, n, k);

            } else if (t == Type::HALF) {
                array_task[i].A = new uint16_t[m * k]; 
                array_task[i].B = new uint16_t[k * n];
                array_task[i].C = new uint16_t[m * n];

                init_16bit(array_task[i].A, array_task[i].B, array_task[i].C, m, n, k);
            } 
        }

        return array_task;
}

inline task* init_hetero_tasks(size_t n_task, Type t) {
    task* array_task = new task[n_task];
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::vector<int> possible_sizes;
    possible_sizes.reserve(STEP_TOTAL);
    
    for (int s = 1; s <= STEP_TOTAL; s++) 
        possible_sizes.push_back(s * STEP_SIZE);

    std::uniform_int_distribution<> dis_idx(0, possible_sizes.size() - 1);

    for (size_t i = 0; i < n_task; i++) {
        int curr_m = possible_sizes[dis_idx(gen)];
        int curr_n = possible_sizes[dis_idx(gen)];
        int curr_k = possible_sizes[dis_idx(gen)];

        //cout << " M : " << curr_m << " N: " << curr_n << " K: " << curr_k << "\n";

        array_task[i].M = curr_m;
        array_task[i].N = curr_n;
        array_task[i].K = curr_k;
        array_task[i].type = t;

        size_t A = (size_t)curr_m * curr_k;
        size_t B = (size_t)curr_k * curr_n;
        size_t C = (size_t)curr_m * curr_n;

        if (t == Type::FLOAT) {
            array_task[i].A = new float[A]; 
            array_task[i].B = new float[B];
            array_task[i].C = new float[C];

            init_32bit(array_task[i].A, array_task[i].B, array_task[i].C, curr_m, curr_n, curr_k);

        } else if (t == Type::HALF) {
            array_task[i].A = new uint16_t[A]; 
            array_task[i].B = new uint16_t[B];
            array_task[i].C = new uint16_t[C];

            init_16bit(array_task[i].A, array_task[i].B, array_task[i].C, curr_m, curr_n, curr_k);
        } 
    }

    return array_task;
}

inline void clean_tasks(task* array_task, size_t n_task) {
    if (array_task == nullptr) return;

    Type type = array_task->type;

    if (type == Type::FLOAT) {
        for (int i = 0; i < n_task; i++) {
            delete[] static_cast<float*>(array_task[i].A);
            delete[] static_cast<float*>(array_task[i].B);
            delete[] static_cast<float*>(array_task[i].C);
        }
    }

    if (type == Type::HALF) {
        for (int i = 0; i < n_task; i++) {
            delete[] static_cast<uint16_t*>(array_task[i].A);
            delete[] static_cast<uint16_t*>(array_task[i].B);
            delete[] static_cast<uint16_t*>(array_task[i].C);
        }
    }

    delete[] array_task;
}

inline uint16_t float_to_half(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t s = (x >> 31) & 0x00000001;
    uint32_t e = (x >> 23) & 0x000000ff;
    uint32_t m = x & 0x007fffff;

    uint32_t h_e, h_m;

    if (e >= 143) { 
        h_e = 31; h_m = 0;
    } 
    else if (e < 113) { 
        h_e = 0; h_m = 0;
    } 
    else {
        h_e = e - 112; 
        h_m = m >> 13;
    }
    return (s << 15) | (h_e << 10) | h_m;
}

inline float half_to_float(uint16_t h) {
    uint32_t s = (h >> 15) & 0x00000001;
    uint32_t e = (h >> 10) & 0x0000001f;
    uint32_t m = h & 0x000003ff;
    
    if (e == 0) {
        if (m == 0) {
            uint32_t res = s << 31;
            float f; memcpy(&f, &res, 4); return f;
        } else {
            while (!(m & 0x00000400)) { m <<= 1; e -= 1; }
            e += 1; m &= ~0x00000400;
        }
    } else if (e == 31) {
        if (m == 0) { 
             uint32_t res = (s << 31) | 0x7f800000;
             float f; memcpy(&f, &res, 4); return f;
        } else { 
             uint32_t res = (s << 31) | 0x7f800000 | (m << 13);
             float f; memcpy(&f, &res, 4); return f;
        }
    }
    
    e = e + (127 - 15);
    m = m << 13;
    uint32_t res = (s << 31) | (e << 23) | m;
    float f; memcpy(&f, &res, 4); return f;
}

inline void init_32bit(void* A, void* B, void* C, int M, int N, int K) {
    float* pA = (float*)A;
    float* pB = (float*)B;
    float* pC = (float*)C;

    random_device rd;
    mt19937 gen(rd());
    
    uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < M * K; ++i) 
        pA[i] = dis(gen);

    for (int i = 0; i < K * N; ++i) 
        pB[i] = dis(gen);

    for (int i = 0; i < M * N; ++i) 
        pC[i] = 0.0f;
}

inline void init_16bit(void* A, void* B, void* C, int M, int N, int K) {
    uint16_t* pA = (uint16_t*)A;
    uint16_t* pB = (uint16_t*)B;
    uint16_t* pC = (uint16_t*)C;

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (int i = 0; i < M * K; ++i) pA[i] = float_to_half(dis(gen));
    for (int i = 0; i < K * N; ++i) pB[i] = float_to_half(dis(gen));
    memset(pC, 0, M * N * sizeof(uint16_t));
}

inline void init_8bit(void* A, void* B, void* C, int M, int N, int K) {
    int8_t* pA = (int8_t*)A;
    int8_t* pB = (int8_t*)B;
    int32_t* pC = (int32_t*)C; /* C 32 bit for the overflows */

    random_device rd;
    mt19937 gen(rd());
    
    uniform_int_distribution<int> dis(-127, 127); /* 8bit maximum and minimum */

    for (int i = 0; i < M * K; ++i) 
        pA[i] = (int8_t)dis(gen);

    for (int i = 0; i < K * N; ++i) 
        pB[i] = (int8_t)dis(gen);

    memset(pC, 0, M * N * sizeof(int32_t));
}

/* return the accellerator string name */
inline string get_acc_string(BT backend_type) {
    string res;
    switch (backend_type) {
        case BT::CUDA:
            res = "CUDA"; break;
        case BT::SYCL:
            res = "SYCL"; break;
        case BT::OPENVINO: 
            res = "OPENVINO"; break;
        case BT::OPENBLAS: 
            res = "OPENBLAS"; break;
        default:
            cerr << "[SCHEDULER] ERROR get_acc_string \n";
            exit(EXIT_FAILURE);
    }
    return res;
}

#endif
