#ifndef TASKS_H
#define TASKS_H

#include <iostream>
#include <chrono>
#include <random>

using namespace std;

enum Type : int { 
    FLOAT,
    HALF,
    UINT8,
};

typedef struct {
    void *A;
    void *B;
    void *C;
    
    Type type; /* 1 float, 2 half, 3 8bit*/
    int M; 
    int N; 
    int K; 

    /* benchmarking */
    chrono::high_resolution_clock::time_point start_time;
    chrono::high_resolution_clock::time_point end_time;
} task;

inline task* init_tasks(size_t n_task, int m, int n, int k, Type t) {
    if (t == Type::FLOAT) {
        task* array_task = new task[n_task];
        random_device rd;
        mt19937 gen(rd());

        uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (int i = 0; i < n_task; i++) {
            array_task[i].M = m;
            array_task[i].N = n;
            array_task[i].K = k;
            array_task[i].type = t;

            array_task[i].A = new float[m * k]; 
            array_task[i].B = new float[k * n];
            array_task[i].C = new float[m * n];

            float *A = (float*)array_task[i].A;
            float *B = (float*)array_task[i].B;
            float *C = (float*)array_task[i].C;

            for (int j = 0; j < m * k; ++j) 
                A[j] = dis(gen);

            for (int j = 0; j < k * n; ++j) 
                B[j] = dis(gen);

            for (int j = 0; j < m * n; ++j) 
                C[j] = 0;

        }
        return array_task;
    } else {
        cout << "[ERROR] Type not hnow ";
        exit(EXIT_FAILURE);
    }
}

inline void clean_tasks(task* array_task, size_t n_task, Type type) {
    if (array_task == nullptr) return;
    if (type == Type::FLOAT) {
        for (int i = 0; i < n_task; i++) {
            delete[] static_cast<float*>(array_task[i].A);
            delete[] static_cast<float*>(array_task[i].B);
            delete[] static_cast<float*>(array_task[i].C);

        }
    }
    delete[] array_task;
}
#endif

