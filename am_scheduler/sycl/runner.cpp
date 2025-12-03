#include <sycl/sycl.hpp>
#include <iostream>
#include "sycl_wrapper.h"


//void run_sycl_code() {
//
//}

extern "C" {
    void sycl_init() {
        std::cout << "[SYCL-LIB] Inizializzazione OneAPI...\n";
    }

    void sycl_process(int N) {
        std::cout << "[SYCL-LIB] Processo " << N << " elementi...\n";
    }
}
