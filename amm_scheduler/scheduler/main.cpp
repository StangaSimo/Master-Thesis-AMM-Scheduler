#include <cstddef>
#include <iostream>

#include "scheduler.hpp"
#include "tasks.hpp"
#include "config.hpp"
#include "tests.hpp"
#include "opencv_test.hpp"

/*
 * TODO: MEMORY ORDERING
 *
 * test jit, prova senza blas, prova meno cpu a blas, benchamark aggiornati (sbagliati per openvino, funzioan solo se le matrici sono ordine), performance map aggiornata,  risolto
 *
 * test prima con openvino scheduler logic (sia matrici 2048 che 4096) e large matrix mul
 *
 *
 * poi ethero 
 * poi ethero senza openvino 
 *
 *
 * MA PERCHE SYCL VA PIU VELOCE CON DYNAMIC NEL TEST SCHEDULER LOGIC????????
 *
 *
 * Asymetric Matrix Multiplication Scheduler 
 *
 *  Design and Implementation of a Unified Heterogeneous Scheduler for CPU, GPU, and NPU Architectures
 *
 *  Development of a Cross-Platform Heterogeneous Scheduler: Bridging Proprietary and Open Standards in Modern Computing
 *
 * "AMM_SCHEDULER: Architecting a runtime scheduling System for Asymmetric Heterogeneous Computing"
 *
 * "AMM_SCHEDULER: Architecting a Scheduling System for Asymmetric Heterogeneous Workloads
 *
 *
 *
 * */

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

        //test_compare_task(task_array, num_matrix);
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
            //test_compare_task(task_array, num_matrix);
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
    //test_compare_task(big_task, 1);
    clean_tasks(big_task, 1);
}

int main() {
    test_accellerators();

    //test_scheduler_logics();
    //test_large_matrix_split();

    test_hetero_logic();

    //test_dynamic();


    //test_jit_times();
    /* remove openvino */
    //test_video_filter(Logic::CUDA_ONLY, false);
    //test_video_filter(Logic::STATIC_PARTITIONING, false);
    //test_video_filter(Logic::DYNAMIC, true);
    return 0;
}
