cmake_minimum_required(VERSION 3.15)
project(Scheduler LANGUAGES C)
message("\n--------------- MAIN COMPILING \n")

# variable from the main CmakeList
if(NOT DEFINED LIBS_DIR)
    message(FATAL_ERROR "LIBS_DIR non definita!")
endif()

# CUDA
if(NOT DEFINED ENABLE_CUDA)
    message("[SCHEDULER] CUDA off on scheduler")
    set(ENABLE_CUDA OFF)
endif()

# SYCL 
if(NOT DEFINED ENABLE_SYCL) 
    message("[SCHEDULER] SYCL off on scheduler")
    set(ENABLE_SYCL OFF) 
endif()

# OPENVINO 
if(NOT DEFINED ENABLE_OPENVINO) 
    set(ENABLE_OPENVINO OFF) 
endif()

# cuda_wrapper.h
include_directories(${CUDA_SRC_DIR})
# sycl_wrapper.h
include_directories(${SYCL_SRC_DIR})
# openvino_wrapper.h
include_directories(${OPEN_SRC_DIR})

# where to find the .so
link_directories(${LIBS_DIR}/lib)

add_executable(scheduler main.c scheduler.c)

# CUDA 
if(ENABLE_CUDA)
    target_compile_definitions(scheduler PRIVATE ENABLE_CUDA)
    target_link_libraries(scheduler cuda_lib stdc++)
endif()

# SYCL 
if(ENABLE_SYCL)
    target_compile_definitions(scheduler PRIVATE ENABLE_SYCL)
    target_link_libraries(scheduler sycl_lib)
endif()

# OPENVINO 
if(ENABLE_OPENVINO)
    target_compile_definitions(scheduler PRIVATE ENABLE_OPENVINO)
    target_link_libraries(scheduler ov_lib)
endif()

# RPATH for libraries path, ORIGIN = exe path, so bin
set_target_properties(scheduler PROPERTIES 
    INSTALL_RPATH "$ORIGIN/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# /libs the exe
install(TARGETS scheduler DESTINATION .)
