#include "sycl_wrapper.h"
#include <sycl/sycl.hpp>
#include <iostream>

// Funzione interna C++
void run_sycl_code() {
    try {
        sycl::queue q(sycl::gpu_selector_v);
        std::cout << "[SYCL-LIB] Device trovato: " 
                  << q.get_device().get_info<sycl::info::device::name>() << "\n";
        
        // Esempio banale di esecuzione
        q.submit([&](sycl::handler& h) {
            sycl::stream out(1024, 256, h);
            h.parallel_for(sycl::range<1>(1), [=](sycl::id<1> i) {
                out << "GODO dal Kernel SYCL!" << sycl::endl;
            });
        }).wait();

    } catch (sycl::exception const& e) {
        std::cout << "[SYCL-LIB] Errore: Nessuna GPU SYCL trovata o errore driver.\n";
        std::cout << "Dettagli: " << e.what() << "\n";
    }
}

extern "C" {
    void sycl_init() {
        std::cout << "[SYCL-LIB] Inizializzazione OneAPI...\n";
    }

    void sycl_process(int N) {
        std::cout << "[SYCL-LIB] Processo " << N << " elementi...\n";
        run_sycl_code();
    }
}
