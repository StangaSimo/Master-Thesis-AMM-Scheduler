cmake_minimum_required(VERSION 3.15)
project(SuperGasseProject)

include(ExternalProject)

set(LIB_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/libs)
file(MAKE_DIRECTORY ${LIB_OUTPUT_DIR})

ExternalProject_Add(build_cuda
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/cuda
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${LIB_OUTPUT_DIR} # Installa in /libs
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    BUILD_ALWAYS 1
)

ExternalProject_Add(build_scheduler
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/scheduler
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${LIB_OUTPUT_DIR} # Mettiamo anche l'exe in /libs? Facciamo di sì
        -DCMAKE_BUILD_TYPE=Release
        -DLIBS_DIR=${LIB_OUTPUT_DIR}             # Gli diciamo dove trovare le lib
        -DCUDA_SRC_DIR=${CMAKE_SOURCE_DIR}/cuda  # Per trovare l'header .h
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    DEPENDS build_cuda
    BUILD_ALWAYS 1
)
