#ifndef OV_WRAPPER_H
#define OV_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

    // Inizializza risorse (se serve)
    void ov_init();

    // Esegue il benchmark su un device specifico ("CPU", "GPU", "NPU")
    void ov_bench_device(const char* device_name, int M, int N, int K, int runs);

#ifdef __cplusplus
}
#endif

#endif
