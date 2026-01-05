#include "ov_wrapper.h" // Il nostro header C
#include <openvino/openvino.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/convert.hpp>
#include <cstdlib>
#include <iostream>

/* This library is intended for the Intel NPU present in the metheor Lake CPUs */

ov::Core core;

using namespace std;

//TODO convert hashmap from tuple to bellissimo int 
/* cache for compiled models */
static std::map<tuple<int, int, int>, ov::InferRequest> request_cache_32bit;
static std::map<tuple<int, int, int>, ov::InferRequest> request_cache_16bit;
static std::map<tuple<int, int, int>, ov::InferRequest> request_cache_8bit;
const size_t MAX_MAP_SIZE = 20; 
size_t map_size_32bit=0;
size_t map_size_16bit=0;
size_t map_size_8bit=0;

void cache_32bit(tuple<int, int, int> key) {
    if (request_cache_32bit.find(key) == request_cache_32bit.end()) {
        map_size_32bit++;
        if (map_size_32bit >= MAX_MAP_SIZE) {
            request_cache_32bit.clear();
            map_size_32bit = 1;
        }
        auto A_ov = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{(size_t)get<0>(key), (size_t)get<2>(key)});
        auto B_ov = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{(size_t)get<2>(key), (size_t)get<1>(key)});

        auto matmul = std::make_shared<ov::op::v0::MatMul>(A_ov, B_ov);
        auto result = std::make_shared<ov::op::v0::Result>(matmul);

        auto model = std::make_shared<ov::Model>(result, ov::ParameterVector{A_ov, B_ov});

        ov::CompiledModel compiled_model = core.compile_model(model, "NPU");
        ov::InferRequest request = compiled_model.create_infer_request();

        request_cache_32bit[key] = request;
    }
}

void cache_16bit(tuple<int, int, int> key) {
    if (request_cache_16bit.find(key) == request_cache_16bit.end()) {
        map_size_16bit++;
        if (map_size_16bit >= MAX_MAP_SIZE) {
            request_cache_16bit.clear();
            map_size_16bit = 1;
        }
        auto A_ov = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{(size_t)get<0>(key), (size_t)get<2>(key)});
        auto B_ov = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{(size_t)get<2>(key), (size_t)get<1>(key)});

        auto matmul = std::make_shared<ov::op::v0::MatMul>(A_ov, B_ov);
        auto result = std::make_shared<ov::op::v0::Result>(matmul);

        auto model = std::make_shared<ov::Model>(result, ov::ParameterVector{A_ov, B_ov});

        ov::CompiledModel compiled_model = core.compile_model(model, "NPU");
        ov::InferRequest request = compiled_model.create_infer_request();

        request_cache_16bit[key] = request;
    }
}

void cache_8bit(tuple<int, int, int> key) {
    if (request_cache_8bit.find(key) == request_cache_8bit.end()) {
        map_size_8bit++;
        if (map_size_8bit >= MAX_MAP_SIZE) {
            request_cache_8bit.clear();
            map_size_8bit = 1;
        }
        auto A_ov = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::Shape{(size_t)get<0>(key), (size_t)get<2>(key)});
        auto B_ov = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::Shape{(size_t)get<2>(key), (size_t)get<1>(key)});

        /* convert inner tensor from 8bit to 16bit */
        auto A_conv = std::make_shared<ov::op::v0::Convert>(A_ov, ov::element::i16);
        auto B_conv = std::make_shared<ov::op::v0::Convert>(B_ov, ov::element::i16);

        auto matmul = std::make_shared<ov::op::v0::MatMul>(A_conv, B_conv);

        auto result_conv = std::make_shared<ov::op::v0::Convert>(matmul, ov::element::i32);
        auto result = std::make_shared<ov::op::v0::Result>(result_conv);

        auto model = std::make_shared<ov::Model>(result, ov::ParameterVector{A_ov, B_ov});

        ov::CompiledModel compiled_model = core.compile_model(model, "NPU");
        ov::InferRequest request = compiled_model.create_infer_request();

        request_cache_8bit[key] = request;
    }
}

void ov_init_p() {
    bool available = false;

    for (auto &d : core.get_available_devices())
        if (d.find("NPU") != string::npos)
            available = true;

    if (!available) {
        cout << "[OPENVINO] NPU not present, aborting \n";
        exit(EXIT_FAILURE); /* shut down something is wrong */
    }
}

/* gemm with cached openvino */
void ov_gemm_32bit_p(void *A, void *B, void *C, int M, int N, int K){
    auto key = std::make_tuple(M, N, K);
    cache_32bit(key); 

    ov::InferRequest& infer_request = request_cache_32bit[key];

    ov::Tensor tensor_A(ov::element::f32, ov::Shape{(size_t)M, (size_t)K}, A);
    ov::Tensor tensor_B(ov::element::f32, ov::Shape{(size_t)K, (size_t)N}, B);
    ov::Tensor tensor_C(ov::element::f32, ov::Shape{(size_t)M, (size_t)N}, C);

    infer_request.set_input_tensor(0, tensor_A);
    infer_request.set_input_tensor(1, tensor_B);
    infer_request.set_output_tensor(0, tensor_C);

    infer_request.infer();
}

void ov_gemm_16bit_p(void *A, void *B, void *C, int M, int N, int K){
    auto key = std::make_tuple(M, N, K);
    cache_16bit(key);

    ov::InferRequest& infer_request = request_cache_16bit[key];

    ov::Tensor tensor_A(ov::element::f16, ov::Shape{(size_t)M, (size_t)K}, A);
    ov::Tensor tensor_B(ov::element::f16, ov::Shape{(size_t)K, (size_t)N}, B);
    ov::Tensor tensor_C(ov::element::f16, ov::Shape{(size_t)M, (size_t)N}, C);

    infer_request.set_input_tensor(0, tensor_A);
    infer_request.set_input_tensor(1, tensor_B);
    infer_request.set_output_tensor(0, tensor_C);

    infer_request.infer();
}

void ov_gemm_8bit_p(void *A, void *B, void *C, int M, int N, int K){
    auto key = std::make_tuple(M, N, K);
    cache_8bit(key);

    ov::InferRequest& infer_request = request_cache_8bit[key];

    ov::Tensor tensor_A(ov::element::i8, ov::Shape{(size_t)M, (size_t)K}, A);
    ov::Tensor tensor_B(ov::element::i8, ov::Shape{(size_t)K, (size_t)N}, B);
    ov::Tensor tensor_C(ov::element::i32, ov::Shape{(size_t)M, (size_t)N}, C);

    infer_request.set_input_tensor(0, tensor_A);
    infer_request.set_input_tensor(1, tensor_B);
    infer_request.set_output_tensor(0, tensor_C);

    infer_request.infer();
}

extern "C" {
    void ov_init() {
        ov_init_p();
    }

    void ov_gemm_32bit(void *A, void *B, void *C, int M, int N, int K){
        ov_gemm_32bit_p(A, B, C, M, N, K);
    }

    void ov_gemm_16bit(void *A, void *B, void *C, int M, int N, int K){
        ov_gemm_16bit_p(A, B, C, M, N, K);
    }

    void ov_gemm_8bit(void *A, void *B, void *C, int M, int N, int K){
        ov_gemm_8bit_p(A, B, C, M, N, K);
    }
    void ov_free() {
    }
}
