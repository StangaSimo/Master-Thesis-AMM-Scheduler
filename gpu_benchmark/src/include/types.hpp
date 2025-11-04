#ifndef TYPES_HPP
#define TYPES_HPP

/* single run results*/
struct TestResult {
    double ms;
    double gflops;
};

/* runs results */
struct RunData {
    double ms;
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