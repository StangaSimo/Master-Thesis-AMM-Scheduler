#ifndef HELP_HPP
#define HELP_HPP


#define CHECK_CUDA(func) {				    \
    cudaError_t e = (func);			        \
    if(e != cudaSuccess)			        \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e)); \
}


#define CHECK_CUBLAS(func) {                      \
    cublasStatus_t e = (func);                    \
    if(e != CUBLAS_STATUS_SUCCESS)                \
        printf ("%s %d CuBlas: %s", __FILE__, __LINE__, _cuBlasGetErrorEnum(e)); \
}

#endif