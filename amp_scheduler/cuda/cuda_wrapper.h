#ifndef WRAPPER_H
#define WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

    // Inizializza la GPU
    void gpu_init();

    // Fa i calcoli
    // input/output sono array float standard
    void gpu_do_work(const float* input, float* output, int size);

#ifdef __cplusplus
}
#endif

#endif
