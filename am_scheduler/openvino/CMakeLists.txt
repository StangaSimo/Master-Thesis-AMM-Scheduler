cmake_minimum_required(VERSION 3.20)
project(OpenVinoPart LANGUAGES CXX)

message("\n--------------- OPENVINO COMPILING \n")
# Trova OpenVINO
find_package(OpenVINO REQUIRED)

add_library(ov_lib SHARED runner.cpp)

# Esponi l'header
set_target_properties(ov_lib PROPERTIES PUBLIC_HEADER ov_wrapper.h)

# C++17 è richiesto da OpenVINO
target_compile_features(ov_lib PRIVATE cxx_std_17)

# Linka le librerie OpenVINO runtime
target_link_libraries(ov_lib PRIVATE openvino::runtime)

# Installazione
install(TARGETS ov_lib 
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)
