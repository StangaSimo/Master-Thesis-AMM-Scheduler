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

            task* task_array = init_tasks(n_matrix, M, N, K, type);
            AMScheduler scheduler = AMScheduler(l);
            scheduler.do_tasks(task_array, n_matrix);
            scheduler.wait();
            /* test_compare_task(task_array, num_matrix); */
            scheduler.print_stats(task_array,n_matrix);
            clean_tasks(task_array, n_matrix);
        } 
    }
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

const size_t RAM_LIMIT = 20ULL * 1024 * 1024 * 1024;
const size_t RAM_LIMIT_SAFE = 18ULL * 1024 * 1024 * 1024;

size_t check_mem(int n_tasks, int M, int N, int K) {
    size_t elem = (size_t)M * K + (size_t)K * N + (size_t)M * N;
    return n_tasks * elem * 4;
}

void bench_static_partitioning() {
    int sizes[] = {512, 1024, 2048, 4096};
    
    for (int s : sizes) {
        for (int b = 10; b <= 50; b += 10) {
            BATCH_SIZE = b;

            for (int t = 10; ; t += 50) {
                if (check_mem(t, s, s, s) > RAM_LIMIT) {
                    cout << "RAM limit reached (" << t << " tasks)\n";
                    break;
                }

                csvname = "bin/csv/static_part_S" + to_string(s) + "_B" + to_string(b) + ".csv";
                cout << "Static Part | Size: " << s << " | Batch: " << b << " | Tasks: " << t << endl;

                task* tasks = init_tasks(t, s, s, s, Type::FLOAT);
                AMScheduler sched(Logic::STATIC_PARTITIONING);
                sched.do_tasks(tasks, t);
                sched.wait();
                sched.print_stats(tasks, t);
                clean_tasks(tasks, t);
            }
        }
    }
}

void bench_static_hetero() {
    for (int b = 10; b <= 50; b += 10) {
        BATCH_SIZE_HETERO = b;

        for (int t = 10; ; t += 50) {
            if (check_mem(t, MAX_SIZE, MAX_SIZE, MAX_SIZE) > RAM_LIMIT_SAFE) {
                cout << "Safe RAM limit reached (" << t << " tasks)\n";
                break;
            }

            csvname = "bin/csv/static_hetero_B" + to_string(b) + ".csv";
            cout << "Static Hetero | Batch: " << b << " | Tasks: " << t << endl;

            task* tasks = init_hetero_tasks(t, Type::FLOAT);
            AMScheduler sched(Logic::STATIC_HETERO_PARTITIONING);
            sched.do_tasks(tasks, t);
            sched.wait();
            sched.print_stats(tasks, t);
            clean_tasks(tasks, t);
        }
    }
}

void bench_dynamic_logic() {
    int sizes[] = {512, 1024, 2048, 4096};

    for (int s : sizes) {
        for (int t = 10; ; t += 50) {
            if (check_mem(t, s, s, s) > RAM_LIMIT) {
                cout << "RAM limit reached (" << t << " tasks)\n";
                break;
            }

            csvname = "bin/csv/dynamic_homo_S" + to_string(s) + ".csv";
            cout << "Dynamic Homo | Size: " << s << " | Tasks: " << t << endl;

            task* tasks = init_tasks(t, s, s, s, Type::FLOAT);
            AMScheduler sched(Logic::DYNAMIC);
            sched.do_tasks(tasks, t);
            sched.wait();
            sched.print_stats(tasks, t);
            clean_tasks(tasks, t);
        }
    }

    for (int t = 10; ; t += 50) {
        if (check_mem(t, MAX_SIZE, MAX_SIZE, MAX_SIZE) > RAM_LIMIT_SAFE) {
            cout << "Safe RAM limit reached (" << t << " tasks)\n";
            break;
        }

        csvname = "bin/csv/dynamic_hetero.csv";
        cout << "Dynamic Hetero | Tasks: " << t << endl;

        task* tasks = init_hetero_tasks(t, Type::FLOAT);
        AMScheduler sched(Logic::DYNAMIC);
        sched.do_tasks(tasks, t);
        sched.wait();
        sched.print_stats(tasks, t);
        clean_tasks(tasks, t);
    }
}

void bench_large_matrix() {
    int N = 4000;
    int K = 4000;
    int start_M = 5000;
    
    csvname = "bin/csv/large_matrix.csv";

    for (int M = start_M; ; M += 1000) {
        if (check_mem(1, M, N, K) > RAM_LIMIT) {
            cout << "RAM limit reached (M=" << M << ")\n";
            break;
        }

        cout << "Large Matrix | M: " << M << " N: " << N << " K: " << K << endl;

        task* t = init_tasks(1, M, N, K, Type::FLOAT);
        AMScheduler sched(Logic::LARGE_MATRIX_SPLIT);
        sched.do_tasks(t, 1);
        sched.wait();
        sched.print_stats(nullptr, 1);
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
    
    bench_static_partitioning();
    //bench_static_hetero();
    //bench_dynamic_logic();
    //bench_large_matrix();

    return 0;
}
