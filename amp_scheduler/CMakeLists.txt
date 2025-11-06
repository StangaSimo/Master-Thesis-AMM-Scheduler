cmake_minimum_required(VERSION 3.18)

#set(CMAKE_CXX_COMPILER "g++-15")
#set(CMAKE_C_COMPILER "gcc-15")
#set(CMAKE_CXX_COMPILER "g++-13")
#set(CMAKE_C_COMPILER "gcc-13")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(STATUS "Compilatore C++ forzato a: ${CMAKE_CXX_COMPILER}")
message(STATUS "Compilatore C forzato a: ${CMAKE_C_COMPILER}")

project(HybridHello LANGUAGES CXX CUDA)

#find_package(OpenVINO REQUIRED)
#find_package(Threads REQUIRED)
#find_package(CUDA REQUIRED)

add_executable(mio_eseguibile
    main.cpp
    cuda_hello.cu
    openvino_hello.cpp
)

#target_include_directories(mio_eseguibile PRIVATE 
#    ${OpenVINO_INCLUDE_DIRS}
#    ${CUDA_INCLUDE_DIRS}
#)

#target_include_directories(mio_eseguibile PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(mio_eseguibile PRIVATE
    openvino

    cublas 

    nvidia-ml

    cudart
)