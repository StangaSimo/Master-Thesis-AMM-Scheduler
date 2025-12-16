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

 * */

int main() {
    test_accellerators();

    std::cout << "[MAIN] Tests Done\n";

    std::cout << "[MAIN] Init Scheduler\n";

    size_t n = 10;
    task* task_array = init_tasks_float(n, 1024, 1024, 512);

    {
        AMScheduler scheduler = AMScheduler();
        scheduler.do_tasks(task_array, n);
        compare_task_float(task_array, n);
    }

    clean_tasks_float(task_array,n);

    std::cout << "[MAIN] Closed Scheduler\n";

    return 0;
}
