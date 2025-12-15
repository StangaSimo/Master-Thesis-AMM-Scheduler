#include "ov_wrapper.h" // Il nostro header C
#include <openvino/openvino.hpp>
#include <openvino/op/matmul.hpp>
#include <cstdlib>
#include <iostream>

/* This library is intended for the Intel NPU present in the metheor Lake CPUs */

ov::Core core;

using namespace std;

void init_ov(){
    cout << "[OPENVINO] Init OpenVINO module.\n";

    bool available = false;

    for (auto &d : core.get_available_devices())
        if (d.find("NPU") != string::npos)
            available = true;

    if (!available) {
        cout << "[OPENVINO] NPU not present, aborting \n";
        exit(EXIT_FAILURE); /* shut down something is wrong */
    }

    cout << "[OPENVINO] NPU up\n";
}

void gemm_32bit_prova(float *A, float *B, float *C, int M, int N, int K) {
    // ---------------------------------------------------------
    // 1. DEBUG PRELIMINARE: I puntatori sono validi?
    // ---------------------------------------------------------
    if (A == nullptr || B == nullptr || C == nullptr) {
        cerr << "[OV ERROR] Puntatori nulli passati a gemm_32bit!\n";
        return;
    }
    // Stampiamo il primo valore "visto" dalla funzione
    cout << "[OV DEBUG] Input A[0]: " << A[0] << " | B[0]: " << B[0] << "\n";

    try {
        ov::Core core; // Creiamo l'istanza core localmente per pulizia

        // ---------------------------------------------------------
        // 2. Costruzione Grafo (Shapes Statiche)
        // ---------------------------------------------------------
        auto A_ov = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{(size_t)M, (size_t)K});
        auto B_ov = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{(size_t)K, (size_t)N});
        
        // Esplicitiamo transpose_a=false, transpose_b=false per sicurezza
        auto matmul = std::make_shared<ov::op::v0::MatMul>(A_ov, B_ov, false, false);
        
        auto result = std::make_shared<ov::op::v0::Result>(matmul);
        auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{A_ov, B_ov});

        // ---------------------------------------------------------
        // 3. Compilazione (Forziamo CPU per il test)
        // ---------------------------------------------------------
        // Cambia in "NPU" solo dopo che su "CPU" vedi i numeri giusti
        string device = "CPU"; 
        ov::CompiledModel compiled_model = core.compile_model(model, device);
        ov::InferRequest infer_request = compiled_model.create_infer_request();

        // ---------------------------------------------------------
        // 4. Wrapping Tensor e Controllo Dati
        // ---------------------------------------------------------
        ov::Tensor tensor_A(ov::element::f32, ov::Shape{(size_t)M, (size_t)K}, A);
        ov::Tensor tensor_B(ov::element::f32, ov::Shape{(size_t)K, (size_t)N}, B);

        // DEBUG: Verifichiamo se il tensore "vede" i dati
        float* ptr_check = tensor_A.data<float>();
        if (ptr_check[0] != A[0]) {
            cerr << "[OV ERROR] Il Tensor OpenVINO non sta leggendo la memoria host correttamente!\n";
        }

        // Assegnazione agli indici corretti (0 -> A, 1 -> B)
        infer_request.set_input_tensor(0, tensor_A);
        infer_request.set_input_tensor(1, tensor_B);

        // ---------------------------------------------------------
        // 5. Output con SAFE COPY
        // ---------------------------------------------------------
        // Lasciamo allocare a OV, poi copiamo. È il metodo più sicuro.
        infer_request.infer();

        ov::Tensor output_tensor = infer_request.get_output_tensor();
        const float* out_ptr = output_tensor.data<float>();

        // DEBUG: Stampiamo il risultato parziale PRIMA della memcpy
        cout << "[OV DEBUG] Output computed [0]: " << out_ptr[0] 
             << " [1]: " << out_ptr[1] << "\n";

        if (out_ptr[0] == 0.0f && out_ptr[1] == 0.0f && A[0] != 0.0f) {
             cerr << "[OV WARNING] L'output è zero nonostante input validi. Il modello non calcola.\n";
        }

        std::memcpy(C, out_ptr, M * N * sizeof(float));

    } catch (const std::exception& e) {
        cerr << "[OV EXCEPTION] " << e.what() << "\n";
    }
}

void gemm_32bit_prova2(float *A, float *B, float *C, int M, int N, int K){
    /* TODO: don't compile it every time make a method */
    auto A_ov = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{(size_t)M, (size_t)K});
    auto B_ov = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{(size_t)K, (size_t)N});

    auto matmul = std::make_shared<ov::op::v0::MatMul>(A_ov, B_ov);
    auto result = std::make_shared<ov::op::v0::Result>(matmul);

    auto model = std::make_shared<ov::Model>(result, ov::ParameterVector{A_ov, B_ov});

    ov::CompiledModel compiled_model = core.compile_model(model, "NPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    ov::Tensor tensor_A(ov::element::f32, ov::Shape{(size_t)M, (size_t)K}, A);
    ov::Tensor tensor_B(ov::element::f32, ov::Shape{(size_t)K, (size_t)N}, B);

    ov::Tensor tensor_C(ov::element::f32, ov::Shape{(size_t)M, (size_t)N}, C);

    infer_request.set_input_tensor(0, tensor_A);
    infer_request.set_input_tensor(1, tensor_B);
    
    infer_request.set_output_tensor(0, tensor_C);

    infer_request.infer();
}
void prova() {
        cout << "[OPENVINO] PROVA.\n";
}


extern "C" {
    void ov_init() {
        prova();
        cout << "[OPENVINO] Init.\n";
        init_ov();
        prova();
    }

    void ov_gemm_32bit(float *A, float *B, float *C, int M, int N, int K){
        gemm_32bit_prova2(A, B, C, M, N, K);
    }
}
