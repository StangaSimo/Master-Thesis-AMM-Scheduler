#include <stdio.h>
#include <stdlib.h>
#include "cuda_wrapper.h" // Includiamo l'header della lib

#ifndef ENABLE_CUDA
#define ENABLE_CUDA 0
#endif

int main() {
    printf("[SCHEDULER] Partenza in C puro!\n");

#ifdef ENABLE_CUDA
    gpu_init();
#endif

    // 2. Preparo dati
    int N = 5;
    float* dati_in = (float*)malloc(N * sizeof(float));
    float* dati_out = (float*)malloc(N * sizeof(float));

    for(int i=0; i<N; i++) dati_in[i] = (float)i + 1;

    // 3. Passo i puntatori a CUDA
     
#ifdef ENABLE_CUDA
    gpu_do_work(dati_in, dati_out, N);
    printf("[SCHEDULER] Risultato elemento 0: %.2f (Dovrebbe essere 10.0)\n", dati_out[0]);
#endif

#ifndef ENABLE_CUDA
    printf("non ho fatto niente"\n);
#endif

    free(dati_in);
    free(dati_out);
    return 0;
}
