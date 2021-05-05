cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(gpu_planning LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

add_definitions(-DBOOST_LOG_DYN_LINK)
find_package(Boost
  COMPONENTS
    log
    program_options
  REQUIRED
)

#
# Set up library
#

set(LIBRARY_NAME "lib${PROJECT_NAME}")
add_library(${LIBRARY_NAME}
  src/${PROJECT_NAME}/cli.cpp
  src/${PROJECT_NAME}/collision_checker.cu
  src/${PROJECT_NAME}/cuda_device.cu
  src/${PROJECT_NAME}/debug.cpp
  src/${PROJECT_NAME}/logging.cpp
  src/${PROJECT_NAME}/map.cu
  src/${PROJECT_NAME}/robot.cu
  src/${PROJECT_NAME}/obstacle_manager.cu
)

target_include_directories(${LIBRARY_NAME}
  PUBLIC
    src
)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC
    CUDA::cudart
  PRIVATE
    Boost::log_setup
    Boost::log
    Boost::program_options
)

set_property(
  TARGET ${LIBRARY_NAME}
  PROPERTY CUDA_SEPARABLE_COMPILATION ON
)
set_property(
  TARGET ${LIBRARY_NAME}
  PROPERTY CMAKE_CUDA_FLAGS "-arch=sm_30 -lineinfo"
)

#
# Set up executable
#

set(EXECUTABLE_NAME "${PROJECT_NAME}")
add_executable(${EXECUTABLE_NAME}
  src/main.cpp
)

target_link_libraries(${EXECUTABLE_NAME}
  PRIVATE
    ${LIBRARY_NAME}
)

set_property(
  TARGET ${EXECUTABLE_NAME}
  PROPERTY CUDA_SEPARABLE_COMPILATION ON
)
set_property(
  TARGET ${EXECUTABLE_NAME}
  PROPERTY CMAKE_CUDA_FLAGS "-arch=sm_30 -lineinfo"
)
