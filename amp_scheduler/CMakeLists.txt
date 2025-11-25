cmake_minimum_required(VERSION 3.18)

option(ENABLE_CUDA "Enable CUDA support" ON)
option(ENABLE_OPENVINO "Enable OpenVINO support" ON)

set(PROJECT_LANGUAGES CXX)
if(ENABLE_CUDA)
    list(APPEND PROJECT_LANGUAGES CUDA)
endif()

project(HybridHello LANGUAGES ${PROJECT_LANGUAGES})

configure_file(
    ${CMAKE_SOURCE_DIR}/config.hpp.in
    ${CMAKE_BINARY_DIR}/config.hpp
)

if (ENABLE_OPENVINO)
    set(CMAKE_CXX_COMPILER "clang++")
    set(CMAKE_C_COMPILER "clang")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message(STATUS "CUDA support: ${ENABLE_CUDA}")
message(STATUS "OpenVINO support: ${BUILD_WITH_OPENVINO}")
message(STATUS "File config.hpp generated at: ${CMAKE_BINARY_DIR}/config.hpp")

set(MY_SOURCES
    src/main.cpp
)

if(ENABLE_CUDA)
    list(APPEND MY_SOURCES src/cuda_hello.cu)
endif()

if(ENABLE_OPENVINO)
    list(APPEND MY_SOURCES src/openvino_hello.cpp)
endif()

add_executable(amp_scheduler ${MY_SOURCES})

target_include_directories(amp_scheduler PUBLIC
    ${CMAKE_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

if(ENABLE_CUDA)
    target_link_libraries(amp_scheduler PRIVATE
        cublas
        nvidia-ml
        cudart
    )
endif()

if(ENABLE_OPENVINO)
    target_link_libraries(amp_scheduler PRIVATE openvino)
endif()