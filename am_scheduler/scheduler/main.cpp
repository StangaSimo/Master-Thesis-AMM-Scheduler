#include <cstddef>
#include <iostream>

#include "scheduler.hpp"
#include "tasks.hpp"
#include "config.hpp"
#include "tests.hpp"
#include "opencv_test.hpp"


/* appunti:
 * 
 * Piccolo scontro con la realtà, fare prima logica 32 bit, scrivere dei test 32 bit e vedere se siamo piu veloci
 * cosa che vedo molto strana.
 *
 * Fare implementazione di convoluzioni per ogni accelleratore,
 * trovare matrici filtro e capire come funzionano, 
 * implementare test per convoluzioni per vedere se funzionano,
 *
 * implementazione di matrici con dag e logica dello scheduler per gestirle.
 *
 * blocking ring buffer, se è vuoto si fa busy spin per un po' poi ci si mette a dormire 
 * veniamo svegliati dal main comunque.
 *
 * tabella matrici e ms, e modello di regressione per il fall back
 *
 * uint16_t per matrici a 16 bit, mentre float per 32.
 *
 * scrivere mille test veri, benchmark veri, e anche benchmark per calcolo su dag, 
 *
 * TODO: per ora, round robin, e solo cuda, pipeline per diverse operazioni (convoluzione e gemm), auto partitioning. 
 *
 * TODO: dag logic per grafi complessi? 
 *
 * TODO: simulare una rete neurale: quindi frame 1 per matrice 1 poi 2 e cosi via, poi frame 2 per matrice 1 poi 2 e cosi via per i test, quindi simil dag. 
 *                                                                       
 * TODO: big matrix multiplication 
 *
 * TODO: latenza e tempo totale per fare benchmark, anche watt consumati in generale e quindi anche una logica piu smart
 *
 * TODO: batch implementation per gli accelleratori che possono? 
 * 
 *
 *
 *
 * scheduler targeting different accellerators in matrix multplication 
 *
 *
 * TODO: streming di Matrici diverse 
 * TODO: aggiustare large matrix split con euristica.
 *
 *
 * test jit -> euristica risolta -
 *
 * */

/* BATCH_SIZE, SLEEP, DEBUG */

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
        for (int i=0; i<2; i++) {
            if (i == 0) {l = Logic::CUDA_ONLY; cout << "\nCuda with scheduler \n";}
            if (i == 1) {l = Logic::STATIC_PARTITIONING; cout << "\nStatic partitioning \n";}

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
    test_large_matrix_split();

    //test_jit_times();
    /* remove openvino */
    //test_video_filter(Logic::CUDA_ONLY, false);
    //test_video_filter(Logic::STATIC_PARTITIONING, true);
    return 0;
}
