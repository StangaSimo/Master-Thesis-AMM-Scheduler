#include <stdio.h>
#include <stdlib.h>

#include "sycl_wrapper.h"
#include "cuda_wrapper.h"
#include "ov_wrapper.h"

int main() {
    printf("[SCHEDULER] Partenza in C puro!\n");

    // 2. Preparo dati
    int N = 5;
    float* dati_in = (float*)malloc(N * sizeof(float));
    float* dati_out = (float*)malloc(N * sizeof(float));

    for(int i=0; i<N; i++) dati_in[i] = (float)i + 1;

#ifdef ENABLE_CUDA
    gpu_init();
    gpu_do_work(dati_in, dati_out, N);
    printf("[SCHEDULER] Risultato elemento 0: %.2f (Dovrebbe essere 10.0)\n", dati_out[0]);
#else
    printf("\n[INFO] CUDA OFF \n");
#endif

#ifdef ENABLE_SYCL
    printf("\n>>> Attivazione Modulo SYCL <<<\n");
    sycl_init();
    sycl_process(100);
#else
    printf("\n[INFO] SYCL OFF \n");
#endif

#ifdef ENABLE_OPENVINO
    printf("\n>>> Attivazione Modulo OpenVINO <<<\n");
    ov_init();
#else
printf("\n[INFO] OPENVINO OFF \n");
#endif

    free(dati_in);
    free(dati_out);
    return 0;
}
