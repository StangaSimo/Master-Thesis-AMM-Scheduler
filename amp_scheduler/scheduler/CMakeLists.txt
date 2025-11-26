cmake_minimum_required(VERSION 3.15)
project(SchedulerPart LANGUAGES C) # Solo C!

# Recuperiamo variabili passate dal Boss
if(NOT DEFINED LIBS_DIR)
    message(FATAL_ERROR "LIBS_DIR non definita!")
endif()

# Header: dove trovare wrapper.h
include_directories(${CUDA_SRC_DIR})

# Librerie: dove trovare i .so
link_directories(${LIBS_DIR}/lib)

add_executable(my_scheduler main.c)

# Linkiamo la nostra lib cuda E la lib standard C++ (necessaria perché cuda usa C++)
target_link_libraries(my_scheduler my_cuda_lib stdc++)

# RPATH: dice all'eseguibile di cercare le .so nella cartella ./lib vicino a se stesso
set_target_properties(my_scheduler PROPERTIES 
    INSTALL_RPATH "$ORIGIN/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Installa l'eseguibile nella root di /libs
install(TARGETS my_scheduler DESTINATION .)
