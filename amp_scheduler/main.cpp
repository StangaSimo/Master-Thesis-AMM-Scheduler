#include <iostream>
#include "cuda_hello.h"
#include "openvino_hello.h"

int main() {
    std::cout << "--- Hello World Ibrido ---" << std::endl;

    /* --- OpenVINO --- */
    try {
        runOpenvinoHello();
    } catch (const std::exception& e) {
        std::cerr << "[OpenVINO] Errore: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "--------------------------" << std::endl;

    /* --- CUDA --- */
    try {
        runCudaHello();
    } catch (const std::exception& e) {
        std::cerr << "[CUDA] Errore: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "--- Esecuzione completata ---" << std::endl;
    return 0;
}