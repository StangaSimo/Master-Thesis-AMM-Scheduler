cmake_minimum_required(VERSION 3.15)
project(amp_scheduler)

include(ExternalProject)

set(LIB_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${LIB_OUTPUT_DIR})

option(USE_CUDA "Abilita il backend CUDA" OFF) 
#option(USE_SYCL "Abilita il backend SYCL" OFF)

set(SCHEDULER_DEPS "") # depends variable 

# cuda
if(USE_CUDA)
    message(STATUS "CUDA Backend ON")
    ExternalProject_Add(build_cuda
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/cuda
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${LIB_OUTPUT_DIR}
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        BUILD_ALWAYS 1
    )

    list(APPEND SCHEDULER_DEPS build_cuda)
else()
    message(STATUS "CUDA Backend OFF")
endif()

# scheduler
ExternalProject_Add(build_scheduler
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/scheduler
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${LIB_OUTPUT_DIR} # exe in libs
        -DCMAKE_BUILD_TYPE=Release
        -DLIBS_DIR=${LIB_OUTPUT_DIR} # folder for the libs 
        -DCUDA_SRC_DIR=${CMAKE_SOURCE_DIR}/cuda
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON # for clangd

        -DENABLE_CUDA=${USE_CUDA} # USE_CUDA

    DEPENDS ${SCHEDULER_DEPS}
    BUILD_ALWAYS 1
)

add_custom_target(UpdateLSP
    ALL
    COMMAND ${CMAKE_COMMAND} -E echo "update LSP..."
    COMMAND python3 ${CMAKE_SOURCE_DIR}/merge_clangd.py
    COMMENT "merge compile_commands.json for clangd"
    DEPENDS build_scheduler 
)
