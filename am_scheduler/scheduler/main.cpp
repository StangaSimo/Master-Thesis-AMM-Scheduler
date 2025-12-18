#include "scheduler.hpp"
#include <iostream>
#include <ostream>

#include "sharedbuffer.hpp"
#include "tests.hpp"

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
 * TODO: support for 16 bit and 8 bit and update tests. 
 *
 * */

int main() {

    /* testing purpuse */
    test_accellerators();

    std::cout << "[MAIN] Tests Done\n";
    std::cout << "[MAIN] Init Scheduler\n";

    size_t n_matrix = 100;
    int M = 512;
    int N = 512;
    int K = 256;

    Logic l;

    for (int i=0; i<2; i++) {

        cout << "\n-------------------- [MAIN] ------------------------- \n" ;

        if (i == 0) {l = Logic::CUDA_ONLY;}
        if (i == 1) {l = Logic::ROUND_ROBIN;}
        if (i == 2) {l = Logic::AUTO_PARTITIONING;}

        task* task_array = init_tasks(n_matrix, M, N, K, Type::FLOAT);

        AMScheduler scheduler = AMScheduler(l);

        scheduler.do_tasks(task_array, n_matrix);
        scheduler.wait();

        //test_compare_task_float(task_array, n_matrix);
        print_performance_stats(task_array, n_matrix);
        clean_tasks(task_array,n_matrix, Type::FLOAT);

    } 

    return 0;
}
