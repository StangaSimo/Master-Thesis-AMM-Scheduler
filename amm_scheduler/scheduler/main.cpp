#include <cstddef>
#include <iostream>

#include "scheduler.hpp"
#include "tasks.hpp"
#include "config.hpp"
#include "tests.hpp"
#include "opencv_test.hpp"

using namespace std;

void test_hetero_logic() {
    size_t n_matrix = 80;

    task* task_array = init_hetero_tasks(n_matrix, Type::FLOAT);
    cout << "Task create \n";
    Logic l = Logic::CUDA_ONLY;
    for (int i=0; i<3; i++) {

        if (i == 1) {l = Logic::STATIC_HETERO_PARTITIONING; cout << "\nStatic partitioning \n";}
        if (i == 2) {l = Logic::DYNAMIC; cout << "\nStatic partitioning \n";}

        AMScheduler scheduler = AMScheduler(l);

        scheduler.do_tasks(task_array, n_matrix);
        scheduler.wait();

        /* test_compare_task(task_array, num_matrix); */
        scheduler.print_stats(task_array,n_matrix);

    }
    clean_tasks(task_array, n_matrix);
}

void test_dynamic() {
    size_t n_matrix = 500;
    int M = M_;
    int N = N_;
    int K = K_;
    Logic l = Logic::DYNAMIC;
    Type type = Type::FLOAT;

    task* task_array = init_tasks(n_matrix, M, N, K, type);
    AMScheduler scheduler = AMScheduler(l);
    scheduler.do_tasks(task_array, n_matrix);
    scheduler.wait();
    scheduler.print_stats(task_array,n_matrix);
    clean_tasks(task_array, n_matrix);
}

void test_scheduler_logics() {
    size_t n_matrix = N_MATRIX;
    int M = M_;
    int N = N_;
    int K = K_;

    Type type = Type::FLOAT;

    task* task_array = init_tasks(n_matrix, M, N, K, type);
    for (int w=0; w<1; w++) {
        if (w == 0) {cout << "\n--------------------  32BIT:  \n";}
        if (w == 1) {cout << "\n--------------------  16BIT:  \n"; type = Type::HALF;}

        /* cuda test without passing from scheduler */
        //test_cuda_streaming(M, N, K, n_matrix, type);

        Logic l;
        for (int i=0; i<3; i++) {
            if (i == 0) {l = Logic::CUDA_ONLY; cout << "\nCuda with scheduler \n";}
            if (i == 1) {l = Logic::STATIC_PARTITIONING; cout << "\nStatic partitioning \n";}
            if (i == 2) {l = Logic::DYNAMIC; cout << "\nDynamic \n";}

            AMScheduler scheduler = AMScheduler(l);
            scheduler.do_tasks(task_array, n_matrix);
            scheduler.wait();
            /* test_compare_task(task_array, num_matrix); */
            scheduler.print_stats(task_array,n_matrix);
        } 
    }
    clean_tasks(task_array, n_matrix);
}

void test_large_matrix_split() {
    cout << "\nLarge Matrix Split \n";
    Logic l = Logic::LARGE_MATRIX_SPLIT; 
    task* big_task = init_tasks(1, M_split,N_split,K_split,Type::FLOAT);
    AMScheduler scheduler = AMScheduler(l);
    scheduler.do_tasks(big_task, 1);
    scheduler.wait();
    scheduler.print_stats(nullptr,1);
    /* test_compare_task(big_task, 1); */
    clean_tasks(big_task, 1);
}

/************************ real benchmarks *****************************/

/* 23 GB */
const size_t RAM_LIMIT = 23ULL * 1024 * 1024 * 1024;

size_t check_mem(int n_tasks, int M, int N, int K) {
    size_t elem = (size_t)M * K + (size_t)K * N + (size_t)M * N;
    return n_tasks * elem * 4;
}

int get_max_tasks(int M, int N, int K) {
    size_t elem = (size_t)M * K + (size_t)K * N + (size_t)M * N;
    size_t task_size = elem * 4; 
    return (int)(RAM_LIMIT / task_size);
}

size_t actual_tasks_size(task* tasks, int n) {
    size_t total_bytes = 0;
    for (int i = 0; i < n; ++i) {
        size_t matrix_A = (size_t)tasks[i].M * tasks[i].K;
        size_t matrix_B = (size_t)tasks[i].K * tasks[i].N;
        size_t matrix_C = (size_t)tasks[i].M * tasks[i].N;
        total_bytes += (matrix_A + matrix_B + matrix_C) * 4;
    }
    return total_bytes;
}

void bench_static_partitioning() {
    int sizes[] = {1024, 2048, 4096};
    
    for (int s : sizes) {
        int step;
        if (s == 1024) step = 50;       
        else if (s == 2048) step = 20;   
        else step = 4;                  

        int max_possible = get_max_tasks(s, s, s);

        cout << "Allocating " << max_possible << " tasks for size " << s << "...\n";
        task* all_tasks = init_tasks(max_possible, s, s, s, Type::FLOAT);

        for (int b = 20; b <= 70; b += 10) {
            BATCH_SIZE = b;

            int t = step; 
            bool last_run = false;

            while (!last_run) {
                if (t >= max_possible) {
                    t = max_possible;
                    last_run = true;
                }

                csvname = "bin/csv/static_part_S" + to_string(s) + "_B" + to_string(b) + ".csv";
                
                if (check_mem(t, s, s, s) > RAM_LIMIT) {
                    break;
                }

                cout << "Static Part | Size: " << s << " | Batch: " << b << " | Tasks: " << t;
                if (last_run) cout << " (MAX RAM)";
                cout << endl;

                AMScheduler sched(Logic::STATIC_PARTITIONING);
                sched.do_tasks(all_tasks, t);
                sched.wait();
                sched.print_stats(all_tasks, t);

                t += step;
            }
        }
        clean_tasks(all_tasks, max_possible);
    }
}

void bench_dynamic_homo() {
    int sizes[] = {1024, 2048, 4096};

    cout << "\n=== START DYNAMIC & CUDA HOMOGENEOUS BENCHMARK ===\n";

    for (int s : sizes) {
        int step;
        if (s == 1024) step = 50;
        else if (s == 2048) step = 20;
        else step = 4;

        int max_possible = get_max_tasks(s, s, s);
        
        cout << "Allocating " << max_possible << " tasks for size " << s << "...\n";
        task* all_tasks = init_tasks(max_possible, s, s, s, Type::FLOAT);

        int t = step;
        bool last_run = false;

        while (!last_run) {
            if (t >= max_possible) {
                t = max_possible;
                last_run = true;
            }

            if (check_mem(t, s, s, s) > RAM_LIMIT) {
                break;
            }

            csvname = "bin/csv/dynamic_homo_S" + to_string(s) + ".csv";
            cout << "Dynamic Homo | Size: " << s << " | Tasks: " << t << endl;

            {
                AMScheduler sched_dyn(Logic::DYNAMIC);
                sched_dyn.do_tasks(all_tasks, t);
                sched_dyn.wait();
                sched_dyn.print_stats(all_tasks, t);
            }

            t += step;
        }

        t = step;
        last_run = false;

        while (!last_run) {
            if (t >= max_possible) {
                t = max_possible;
                last_run = true;
            }

            if (check_mem(t, s, s, s) > RAM_LIMIT) {
                break;
            }

            csvname = "bin/csv/cuda_only_S" + to_string(s) + ".csv";
            cout << "CUDA Only    | Size: " << s << " | Tasks: " << t;
            if (last_run) cout << " (MAX RAM)";
            cout << endl;

            {
                AMScheduler sched_cuda(Logic::CUDA_ONLY);
                sched_cuda.do_tasks(all_tasks, t);
                sched_cuda.wait();
                sched_cuda.print_stats(all_tasks, t);
            }

            t += step;
        }

        clean_tasks(all_tasks, max_possible);
    }
}

void bench_hetero_comparison() {
    cout << "\n=== START HETERO COMPARISON BENCHMARK ===\n";

    int max_allocatable = get_max_tasks(MAX_SIZE, MAX_SIZE, MAX_SIZE);
    
    task* shared_tasks = init_hetero_tasks(max_allocatable, Type::FLOAT);

    size_t current_mem = actual_tasks_size(shared_tasks, max_allocatable);
    cout << "Allocated " << max_allocatable << " tasks using " << (double)current_mem / (1024*1024*1024) << " GB of RAM.\n";

    int max_total_tasks = max_allocatable;
    int step = 5;

    for (int b = 20; b <= 70; b += 10) {
        BATCH_SIZE_HETERO = b;
        
        int t = step;
        bool last_run = false;

        while (!last_run) {
            if (t >= max_total_tasks) {
                t = max_total_tasks;
                last_run = true;
            }

            csvname = "bin/csv/static_hetero_B" + to_string(b) + ".csv";
            cout << "Static Hetero | Batch: " << b << " | Tasks: " << t;
            if (last_run) cout << " (MAX ALLOC)";
            cout << endl;

            AMScheduler sched(Logic::STATIC_HETERO_PARTITIONING);
            sched.do_tasks(shared_tasks, t); 
            sched.wait();
            sched.print_stats(shared_tasks, t);

            t += step;
        }
    }

    cout << "\n--- Dynamic Logic on Hetero ---\n";
    
    int t = step;
    bool last_run = false;

    while (!last_run) {
        if (t >= max_total_tasks) {
            t = max_total_tasks;
            last_run = true;
        }

        csvname = "bin/csv/dynamic_hetero.csv";
        cout << "Dynamic Hetero | Tasks: " << t;
        if (last_run) cout << " (MAX ALLOC)";
        cout << endl;

        AMScheduler sched(Logic::DYNAMIC);
        sched.do_tasks(shared_tasks, t);
        sched.wait();
        sched.print_stats(shared_tasks, t);

        t += step;
    }

    cout << "\n--- CUDA Only Logic on Hetero ---\n";

    t = step;
    last_run = false;

    while (!last_run) {
        if (t >= max_total_tasks) {
            t = max_total_tasks;
            last_run = true;
        }

        csvname = "bin/csv/cuda_only_hetero.csv";
        cout << "CUDA Only Hetero | Tasks: " << t;
        if (last_run) cout << " (MAX ALLOC)";
        cout << endl;

        AMScheduler sched(Logic::CUDA_ONLY);
        sched.do_tasks(shared_tasks, t);
        sched.wait();
        sched.print_stats(shared_tasks, t);

        t += step;
    }

    clean_tasks(shared_tasks, max_total_tasks);
}

void bench_large_matrix() {
    int N = 4000;
    int K = 4000;
    int start_M = 5000;
    

    for (int M = start_M; ; M += 1000) {
        if (check_mem(1, M, N, K) > RAM_LIMIT) {
            cout << "RAM limit reached (M=" << M << ")\n";
            break;
        }

        csvname = "bin/csv/large_matrix_cuda.csv";

        cout << "Large Matrix | M: " << M << " N: " << N << " K: " << K << endl;

        task* t = init_tasks(1, M, N, K, Type::FLOAT);
        AMScheduler sched(Logic::LARGE_MATRIX_SPLIT_CUDA);
        sched.do_tasks(t, 1);
        sched.wait();
        sched.print_stats(t, 1);
        clean_tasks(t, 1);
    }
}

int main() {
    test_accellerators();
    //test_scheduler_logics();
    //test_large_matrix_split();
    //test_hetero_logic();
    //test_dynamic();
    //test_jit_times();
    
    /* remove openvino */
    //test_video_filter(Logic::CUDA_ONLY, false);
    //test_video_filter(Logic::STATIC_PARTITIONING, false);
    //test_video_filter(Logic::DYNAMIC, true);
    
    //bench_static_partitioning();
    //bench_dynamic_homo();
    //bench_hetero_comparison();
    
    bench_large_matrix();
    return 0;
}
