#include "scheduler.h"
#include "sycl_wrapper.h"
#include "cuda_wrapper.h"
#include "ov_wrapper.h"
#include <stdio.h>
#include <stdlib.h>

void init_scheduler();

void run_scheduler() {
    printf("[SCHEDULER] Running..\n");

    int M = 1024;
    int N = 1024;
    int K = 512;

    float *A, *B, *C; 

    A = malloc(M*N*sizeof(float));
    if (A == NULL) {
        fprintf(stderr, "[ERROR] malloc\n");
        exit(EXIT_FAILURE);
    }

    B = malloc(M*N*sizeof(float));
    if (B == NULL) {
        fprintf(stderr, "[ERROR] malloc\n");
        exit(EXIT_FAILURE);
    }   

    C = malloc(M*N*sizeof(float));
    if (C == NULL) {
        fprintf(stderr, "[ERROR] malloc\n");
        exit(EXIT_FAILURE);
    } 


#ifdef ENABLE_CUDA
    printf("\n>>> Attivazione Modulo CUDA <<<\n");
    gpu_init();
    run_cuda_32bit(A,B,C,M,N,K);
#else
    printf("\n[INFO] CUDA OFF \n");
#endif

#ifdef ENABLE_SYCL
    printf("\n>>> Attivazione Modulo SYCL <<<\n");
#else
    printf("\n[INFO] SYCL OFF \n");
#endif

#ifdef ENABLE_OPENVINO
    printf("\n>>> Attivazione Modulo OpenVINO <<<\n");
#else
printf("\n[INFO] OPENVINO OFF \n");
#endif
    
    
    free(A);
    free(B);
    free(C);
}

void free_scheduler() {

}
