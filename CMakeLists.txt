cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(gpu_planning LANGUAGES CXX CUDA)

add_executable(${PROJECT_NAME}
  src/main.cpp
)

set_property(
  TARGET ${PROJECT_NAME}
  PROPERTY CUDA_SEPARABLE_COMPILATION ON
)
set_property(
  TARGET ${PROJECT_NAME}
  PROPERTY CMAKE_CUDA_FLAGS "-arch=sm_30"
)
