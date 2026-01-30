cmake_minimum_required(VERSION 3.20)
project(OpenVinoPart LANGUAGES CXX)

message("\n--------------- OPENVINO COMPILING \n")

find_package(OpenVINO REQUIRED)

add_library(ov_lib SHARED runner.cpp)

set_target_properties(ov_lib PROPERTIES PUBLIC_HEADER ov_wrapper.h)

# C++17 needed OpenVINO
target_compile_features(ov_lib PRIVATE cxx_std_17)

target_link_libraries(ov_lib PRIVATE openvino::runtime)

install(TARGETS ov_lib 
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)
