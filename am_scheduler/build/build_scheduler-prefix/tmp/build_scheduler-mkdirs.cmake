# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/simo/Documents/j/master-thesis/am_scheduler/scheduler")
  file(MAKE_DIRECTORY "/home/simo/Documents/j/master-thesis/am_scheduler/scheduler")
endif()
file(MAKE_DIRECTORY
  "/home/simo/Documents/j/master-thesis/am_scheduler/build/build_scheduler-prefix/src/build_scheduler-build"
  "/home/simo/Documents/j/master-thesis/am_scheduler/build/build_scheduler-prefix"
  "/home/simo/Documents/j/master-thesis/am_scheduler/build/build_scheduler-prefix/tmp"
  "/home/simo/Documents/j/master-thesis/am_scheduler/build/build_scheduler-prefix/src/build_scheduler-stamp"
  "/home/simo/Documents/j/master-thesis/am_scheduler/build/build_scheduler-prefix/src"
  "/home/simo/Documents/j/master-thesis/am_scheduler/build/build_scheduler-prefix/src/build_scheduler-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/simo/Documents/j/master-thesis/am_scheduler/build/build_scheduler-prefix/src/build_scheduler-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/simo/Documents/j/master-thesis/am_scheduler/build/build_scheduler-prefix/src/build_scheduler-stamp${cfgdir}") # cfgdir has leading slash
endif()
