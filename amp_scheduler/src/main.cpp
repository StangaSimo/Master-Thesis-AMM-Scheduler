#include <iostream>
#include "config.hpp"

#ifdef ENABLE_OPENVINO
#include "amp_scheduler/openvino_hello.hpp"
#endif

#ifdef ENABLE_CUDA
#include "amp_scheduler/cuda_hello.hpp"
#endif



int main() {
    std::cout << "--- Hello World Ibrido ---" << std::endl;

#ifdef ENABLE_OPENVINO
    /* --- OpenVINO --- */
    try {
        runOpenvinoHello();
    } catch (const std::exception& e) {
        std::cerr << "[OpenVINO] Errore: " << e.what() << std::endl;
        return 1;
    }
#endif

    std::cout << "--------------------------" << std::endl;

#ifdef ENABLE_CUDA
    /* --- CUDA --- */
    try {
        runCudaHello();
    } catch (const std::exception& e) {
        std::cerr << "[CUDA] Errore: " << e.what() << std::endl;
        return 1;
    }
#endif

    std::cout << "--- Esecuzione completata ---" << std::endl;
    return 0;
}