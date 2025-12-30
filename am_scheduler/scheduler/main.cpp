#include "scheduler.hpp"
#include <iostream>
#include <ostream>

#include "sharedbuffer.hpp"
#include "tasks.hpp"
#include "config.hpp"
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
 *
 * FARE TIPI DIVERSI (TEST ecc...)
 * AGGIUNGERE LA CPU
 * 
 * LOGICA CON MATRICI DIVERSE.
 *
 *
 *
 * MATRICE ENORME 
 * 
 * ho finito l'implementazione dei benchmark, creato una cache per i modelli di open vino e infatti ora abbiamo delle prestazioni molto carine, ho provato un semplice modello di regressione ma non funziona bene, per cui ho usato un hashmap e giocando con la chiave riesco a fare delle query velocissime che posso anche cachare (visto che di solito le matrici sono sempre tutte uguali).
 *
 * */

/* BATCH_SIZE, SLEEP, DEBUG */




// TODO: vedere se ha senso fare l'implementazione a 8 bit, magari si convertono e via. 
//
//
//
//
//
//
//

int main() {

    //cout << "\n-------------------- [TESTS] ------------------------- \n" ;
    test_accellerators();

    cout << "\n-------------------- [MAIN] ------------------------- \n" ;

    size_t num_matrix = N_MATRIX;
    int M = M_;
    int N = N_;
    int K = K_;
    
    /* cuda test without passing from scheduler */
    cout << "\nCuda only \n";
    test_cuda_streaming(M,N,K,num_matrix,Type::FLOAT);

    Logic l;

    for (int i=0; i<3; i++) {

        if (i == 1) {continue;}
        cout << "\n-------------------- [SCHEDULER] " << i << " ------------------------- \n" ;
        if (i == 0) {l = Logic::CUDA_ONLY;cout << "\nCuda with scheduler \n";}
        if (i == 1) {l = Logic::ROUND_ROBIN;cout << "\nRound robin \n";}
        if (i == 2) {l = Logic::STATIC_PARTITIONING;cout << "\nStatic partitioning \n";}

        task* task_array = init_tasks(num_matrix, M, N, K, Type::FLOAT);

        AMScheduler scheduler = AMScheduler(l);

        scheduler.do_tasks(task_array, num_matrix);
        scheduler.wait();

        //test_compare_task(task_array, num_matrix, Type::FLOAT);
        print_performance_stats(task_array, num_matrix);
        clean_tasks(task_array,num_matrix, Type::FLOAT);
    } 

    return 0;
}
