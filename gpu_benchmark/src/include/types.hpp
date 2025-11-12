#ifndef TYPES_HPP
#define TYPES_HPP

/* single run results*/
struct TestResult {
    double ms_compute;
    double gflops;
    float ms_memcpy;
};

/* runs results */
struct RunData {
    double ms_compute;
    double ms_memcpy;
    double gflops;
    double avg_power;
    double min_power;
    double max_power;
};

enum class KernelType {
    NAIVE,
    TILED, 
    DENSE,
    COPY,
    ADD,
    SCALE,
    TRIAD,
    CUBLAS,
    TENSOR,
};

#endif