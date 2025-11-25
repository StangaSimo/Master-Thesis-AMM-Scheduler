cmake_minimum_required(VERSION 3.20)
project(SyclMklProject)

find_package(MKL CONFIG REQUIRED)

add_executable(sycl_benchmark main.cpp)

target_compile_options(sycl_benchmark PRIVATE -fsycl)
target_link_options(sycl_benchmark PRIVATE -fsycl)

target_link_libraries(sycl_benchmark PRIVATE MKL::MKL_SYCL)
