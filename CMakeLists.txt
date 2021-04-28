cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(gpu_planning LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

add_definitions(-DBOOST_LOG_DYN_LINK)
find_package(Boost
  COMPONENTS
    log
    program_options
  REQUIRED
)

set(TARGET_NAME ${PROJECT_NAME})
add_executable(${TARGET_NAME}
  src/main.cpp
  src/logging.cpp
  src/cli.cpp
  src/cuda_device.cu
  src/map.cu
  src/robot.cu
  src/device_robot.cu
  src/collision_checker.cu
  src/debug.cpp
  src/device_2d_array.cu
  src/device_2d_array_handle.cu
  src/device_map.cu
  src/geometry.cpp
)

target_link_libraries(${TARGET_NAME}
  Boost::log_setup
  Boost::log
  Boost::program_options
)

set_property(
  TARGET ${TARGET_NAME}
  PROPERTY CUDA_SEPARABLE_COMPILATION ON
)
set_property(
  TARGET ${TARGET_NAME}
  PROPERTY CMAKE_CUDA_FLAGS "-arch=sm_30 -lineinfo"
)
