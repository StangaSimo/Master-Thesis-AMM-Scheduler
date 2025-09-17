#ifndef TYPES_HPP
#define TYPES_HPP

struct TestResult {
    double ms;
    double gflops;
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
};

#endif