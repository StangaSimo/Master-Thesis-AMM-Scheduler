#include "ov_wrapper.h" // Il nostro header C
#include <openvino/openvino.hpp>
#include <openvino/op/matmul.hpp>
#include <cstdlib>
#include <iostream>

/* This library is intended for the Intel NPU present in the metheor Lake CPUs */

ov::Core core;

using namespace std;

void init(){
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


void gemm_32bit(float *A, float *B, float *C, int M, int N, int K){
    auto A_ov = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{(u_long)M, (u_long)K});
    auto B_ov = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{(u_long)K, (u_long)N});

    auto matmul = std::make_shared<ov::op::v0::MatMul>(A_ov, B_ov);
    auto result = std::make_shared<ov::op::v0::Result>(matmul);

    auto model = std::make_shared<ov::Model>(result, ov::ParameterVector{A_ov, B_ov});

    ov::CompiledModel compiled_model = core.compile_model(model, "NPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    ov::Tensor tensor_A(ov::element::f32, ov::Shape{(u_long)M, (u_long)K}, A);
    ov::Tensor tensor_B(ov::element::f32, ov::Shape{(u_long)K, (u_long)N}, B);

    /* memcpy */
    infer_request.set_input_tensor(0, tensor_A);
    infer_request.set_input_tensor(1, tensor_B);

    /* matmul */
    infer_request.infer();

    /* return result */
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    const float* output_data_f32 = output_tensor.data<float>();
    memcpy(C, output_data_f32, (size_t)M * (size_t)N * sizeof(float));
}



extern "C" {
    void ov_init() {
        init();
    }

    void ov_gemm_32bit(float *A, float *B, float *C, int M, int N, int K){
        gemm_32bit(A, B, C, M, N, K);
    }
}
