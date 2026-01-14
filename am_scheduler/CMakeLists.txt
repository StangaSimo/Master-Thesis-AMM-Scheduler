cmake_minimum_required(VERSION 3.15)
project(am_scheduler)

include(ExternalProject)

set(LIB_OUTPUT_DIR ${CMAKE_SOURCE_DIR}/bin)
file(MAKE_DIRECTORY ${LIB_OUTPUT_DIR})

# CUDA SYCL OPENVINO
option(USE_CUDA "CUDA ON" OFF) 
option(USE_SYCL "SYCL ON" OFF)
option(USE_OPENVINO "OpenVINO ON" OFF)
option(USE_OPENBLAS "OpenBLAS ON" OFF)
option(USE_OPENCV "OpenCV ON" OFF)

set(SCHEDULER_DEPS "") # depends variable 

set(OPT_FLAGS "-O2 -DNDEBUG")

# CUDA
if(USE_CUDA)
    message(STATUS "CUDA Backend ON")
    ExternalProject_Add(build_cuda
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/cuda
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${LIB_OUTPUT_DIR}
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
            -DCMAKE_CXX_FLAGS_RELEASE=${OPT_FLAGS}
            -DCMAKE_CUDA_FLAGS_RELEASE=${OPT_FLAGS}
            #BUILD_ALWAYS 1
    )

    list(APPEND SCHEDULER_DEPS build_cuda)
else()
    message(STATUS "CUDA Backend OFF")
endif()

# SYCL 
if(USE_SYCL)
    message(STATUS "SYCL Backend: ON")
    ExternalProject_Add(build_sycl
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/sycl
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${LIB_OUTPUT_DIR}
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
            -DCMAKE_CXX_COMPILER=icpx 
            -DCMAKE_CXX_FLAGS_RELEASE=${OPT_FLAGS}
            #BUILD_ALWAYS 1
    )
    list(APPEND SCHEDULER_DEPS build_sycl)
else()
    message(STATUS "SYCL Backend: OFF")
endif()

# OPENVINO
if(USE_OPENVINO)
    message(STATUS "OPENVINO Backend: ON")
    ExternalProject_Add(build_openvino
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/openvino
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${LIB_OUTPUT_DIR}
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
            -DCMAKE_CXX_FLAGS_RELEASE=${OPT_FLAGS}
            #BUILD_ALWAYS 1
    )
    list(APPEND SCHEDULER_DEPS build_openvino)
else()
    message(STATUS "OPENVINO Backend: OFF")
endif()

# scheduler
ExternalProject_Add(build_scheduler
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/scheduler
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${LIB_OUTPUT_DIR}
        -DCMAKE_BUILD_TYPE=Release
        -DLIBS_DIR=${LIB_OUTPUT_DIR}
        -DCMAKE_CXX_FLAGS_RELEASE=${OPT_FLAGS}

        -DCUDA_SRC_DIR=${CMAKE_SOURCE_DIR}/cuda
        -DSYCL_SRC_DIR=${CMAKE_SOURCE_DIR}/sycl
        -DOPEN_SRC_DIR=${CMAKE_SOURCE_DIR}/openvino

        -DENABLE_CUDA=${USE_CUDA} # USE_CUDA
        -DENABLE_SYCL=${USE_SYCL} # USE_SYCL
        -DENABLE_OPENVINO=${USE_OPENVINO} # USE_OPENVINO
        -DENABLE_OPENBLAS=${USE_OPENBLAS} # USE_OPENBLAS
        -DENABLE_OPENCV=${USE_OPENCV} # USE_OPENCV

        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON # for clangd
    DEPENDS ${SCHEDULER_DEPS}
    #BUILD_ALWAYS 1
)

#add_custom_target(UpdateLSP
#    ALL
#    COMMAND ${CMAKE_COMMAND} -E echo "update LSP..."
#    COMMAND python3 ${CMAKE_SOURCE_DIR}/merge_clangd.py
#    COMMENT "merge compile_commands.json for clangd"
#    DEPENDS build_scheduler 
#)
