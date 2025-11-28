cmake_minimum_required(VERSION 3.20)
project(SyclPart LANGUAGES CXX)

message("\n--------------- SYCL COMPILING \n")
# Trova OneMKL (opzionale se fai solo hello world, ma tu lo vorrai dopo)
find_package(MKL CONFIG REQUIRED)

# Creiamo la libreria condivisa
add_library(sycl_lib SHARED runner.cpp)

# Header pubblico
#set_target_properties(sycl_lib PROPERTIES PUBLIC_HEADER sycl_wrapper.h)

# Opzioni di compilazione fondamentali per SYCL
target_compile_options(sycl_lib PRIVATE -fsycl)
target_link_options(sycl_lib PRIVATE -fsycl)

# Linkiamo MKL
target_link_libraries(sycl_lib PRIVATE MKL::MKL_SYCL)

# Installazione standard nella cartella libs del boss
install(TARGETS sycl_lib 
    LIBRARY DESTINATION lib
    PUBLIC_HEADER DESTINATION include
)
