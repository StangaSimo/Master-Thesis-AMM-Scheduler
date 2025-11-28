#include "ov_wrapper.h" // Il nostro header C
#include <openvino/openvino.hpp>
#include <openvino/op/matmul.hpp>
#include <iostream>

extern "C" {
    void ov_init() {
        // Magari stampiamo la versione di OpenVINO
        std::cout << "[OV-LIB] Init OpenVINO module.\n";
    }

    void ov_bench_device(const char* device_name, int M, int N, int K, int runs) {
        // Chiamiamo la funzione C++ convertendo la stringa C
        std::cout << "GODO \n";
    }
}
