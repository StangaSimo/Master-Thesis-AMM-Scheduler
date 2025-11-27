cmake_minimum_required(VERSION 3.15)
project(CudaPart LANGUAGES CUDA CXX)

add_library(cuda_lib SHARED runner.cu)

install(TARGETS cuda_lib DESTINATION lib)
