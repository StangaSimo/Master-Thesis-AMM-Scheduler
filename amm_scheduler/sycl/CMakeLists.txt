cmake_minimum_required(VERSION 3.20)
project(SyclPart LANGUAGES CXX)

message("\n--------------- SYCL COMPILING \n")
find_package(MKL CONFIG REQUIRED)

add_library(sycl_lib SHARED runner.cpp)

#set_target_properties(sycl_lib PROPERTIES PUBLIC_HEADER sycl_wrapper.h)

target_compile_options(sycl_lib PRIVATE -fsycl)
target_link_options(sycl_lib PRIVATE -fsycl)

target_link_libraries(sycl_lib PRIVATE MKL::MKL_SYCL)

install(TARGETS sycl_lib 
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)
