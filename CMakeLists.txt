cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(gpu_planning LANGUAGES CXX CUDA)

option(BUILD_TESTS "Build tests" ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(LIBRARY_NAME "lib${PROJECT_NAME}")
set(EXECUTABLE_NAME "${PROJECT_NAME}")

add_subdirectory(src)

if(${BUILD_TESTS})
  enable_testing()
  add_subdirectory(test)
endif()
