//#include <stdio.h>
//#include <stdlib.h>

#include "scheduler.h"

int main() {
    init_scheduler();

    run_scheduler();

    free_scheduler();
    return 0;
}
