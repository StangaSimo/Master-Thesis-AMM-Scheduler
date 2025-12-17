#include "scheduler.hpp"
#include <iostream>
#include <ostream>

#include "tests.hpp"

/* appunti:
 * 
 * Piccolo scontro con la realtà, fare prima logica 32 bit, scrivere dei test 32 bit e vedere se siamo piu veloci
 * cosa che vedo molto strana.
 *
 *
 * Fare implementazione di convoluzioni per ogni accelleratore,
 * trovare matrici filtro e capire come funzionano, 
 * implementare test per convoluzioni per vedere se funzionano,
 * 
 *
 *
 * finire fi fare la logica dello scheduer per streaming nudo e crudo di matrici, 
 *
 * implementazione di matrici con dag e logica dello scheduler per gestirle.
 *
 *
 * blocking ring buffer, se è vuoto si fa busy spin per un po' poi ci si mette a dormire 
 * veniamo svegliati dal main comunque.
 *
 * tabella matrici e ms, e modello di regressione per il fall back
 *
 *
 * array statico std::array per i thread e le strutture dati.
 * 
 * uint16_t per matrici a 16 bit, mentre float per 32.
 *
 * scrivere mille test veri, benchmark veri, e anche benchmark per calcolo su dag, 
 *
 * TODO: latenza e tempo totale per fare benchmark 
 *
 * TODO: batch implementation for GPU, but not so easy with matrix with multiple ???????
 * TODO: dag imlementation for testing
 * TODO  video frame for filter application 
 * TODO  
 *
 * */

int main() {

    /* testing purpuse */
    test_accellerators();

    std::cout << "[MAIN] Tests Done\n";
    std::cout << "[MAIN] Init Scheduler\n";

    size_t n_matrix = 100;
    int M = 1024;
    int N = 1024;
    int K = 512;

    std::cout << "[MAIN] CUDA_ONLY ------------------------------------- \n";
    Logic l = Logic::CUDA_ONLY; 
    for (int i=0; i<2; i++) {
        if (i == 1) {l = Logic::ROUND_ROBIN;}
        task* task_array = init_tasks_float(n_matrix, M, N, K);
        AMScheduler scheduler = AMScheduler(l);
        scheduler.do_tasks(task_array, n_matrix);
        scheduler.wait();

        print_performance_stats(task_array, n_matrix);
        clean_tasks_float(task_array,n_matrix);

    } 

    std::cout << "[MAIN] -------------------------------------------------- \n\n";
    std::cout << "[MAIN] Closed Scheduler\n";
    return 0;
}
