#include "scheduler.hpp"
#include <iostream>
#include <ostream>

#include "tests.hpp"

/* appunti:
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
 * */

int main() {
    tests();
    std::cout << "[MAIN] Tests Done\n";

    std::cout << "[MAIN] Init Scheduler\n";
    {
        AMScheduler scheduler = AMScheduler();
    }
    std::cout << "[MAIN] Closed Scheduler\n";

    return 0;
}
