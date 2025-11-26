#include "cuda_wrapper.h"
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void myKernel(const float* in, float* out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) out[i] = in[i] * 10.0f; // Moltiplica per 10
}

extern "C" {
    void gpu_init() {
        printf("[CUDA-LIB] Inizializzazione device...\n");
    }

    void gpu_do_work(const float* input, float* output, int size) {
        float *d_in, *d_out;
        size_t bytes = size * sizeof(float);

        cudaMalloc(&d_in, bytes);
        cudaMalloc(&d_out, bytes);

        cudaMemcpy(d_in, input, bytes, cudaMemcpyHostToDevice);
        myKernel<<<(size+255)/256, 256>>>(d_in, d_out, size);
        cudaMemcpy(output, d_out, bytes, cudaMemcpyDeviceToHost);

        cudaFree(d_in);
        cudaFree(d_out);
        
        printf("[CUDA-LIB] Calcolo finito sulla GPU.\n");
    }
}
