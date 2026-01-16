cmake_minimum_required(VERSION 3.15)
project(CudaPart LANGUAGES CUDA CXX)

message("\n--------------- CUDA COMPILING \n")

find_package(CUDAToolkit REQUIRED)

add_library(cuda_lib SHARED runner.cu)

target_link_libraries(cuda_lib PRIVATE CUDA::cublas CUDA::cudart)

install(TARGETS cuda_lib DESTINATION lib)
