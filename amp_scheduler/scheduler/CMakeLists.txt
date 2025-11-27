cmake_minimum_required(VERSION 3.15)
project(Scheduler LANGUAGES C)

# variable from the main CmakeList
if(NOT DEFINED LIBS_DIR)
    message(FATAL_ERROR "LIBS_DIR non definita!")
endif()

if(NOT DEFINED ENABLE_CUDA)
    message("CUDA off on scheduler")
    set(ENABLE_CUDA OFF)
endif()

# cuda_wrapper.h
include_directories(${CUDA_SRC_DIR})

# where to find the .so
link_directories(${LIBS_DIR}/lib)

# executable name
add_executable(scheduler main.c)

if(ENABLE_CUDA)
    # Equivale a scrivere: #define ENABLE_CUDA 1 nel codice
    target_compile_definitions(scheduler PRIVATE ENABLE_CUDA)
    # Linkiamo la nostra lib cuda E la lib standard C++ (necessaria perché cuda usa C++)
    target_link_libraries(scheduler cuda_lib stdc++)

    # RPATH for libraries path, ORIGIN = exe path, so bin
    set_target_properties(scheduler PROPERTIES 
        INSTALL_RPATH "$ORIGIN/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
else()
    # RPATH for libraries path, ORIGIN = exe path, so bin
    set_target_properties(scheduler PROPERTIES 
        INSTALL_RPATH "$ORIGIN/lib"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    # Equivale a: #define ENABLE_CUDA 0
endif()

# Installa l'eseguibile nella root di /libs
install(TARGETS scheduler DESTINATION .)
