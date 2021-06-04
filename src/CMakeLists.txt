find_package(CUDAToolkit REQUIRED)

add_definitions(-DBOOST_LOG_DYN_LINK)
find_package(Boost
  COMPONENTS
    log
    program_options
  REQUIRED
)



# Set up library

add_library(${LIBRARY_NAME}
  ${PROJECT_NAME}/cli.cpp
  ${PROJECT_NAME}/collision_checker.cu
  ${PROJECT_NAME}/cuda_device.cu
  ${PROJECT_NAME}/debug.cpp
  ${PROJECT_NAME}/logging.cpp
  ${PROJECT_NAME}/map.cu
  ${PROJECT_NAME}/robot.cu
  ${PROJECT_NAME}/obstacle_manager.cu
  ${PROJECT_NAME}/shape.cu
  ${PROJECT_NAME}/image.cpp
  ${PROJECT_NAME}/trajectory.cu
  ${PROJECT_NAME}/cuda_util.cu
  ${PROJECT_NAME}/thread_block.cu
)

target_include_directories(${LIBRARY_NAME}
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
)

target_link_libraries(${LIBRARY_NAME}
  PUBLIC
    CUDA::cudart
  PRIVATE
    ${LIBRARY_NAME}_tracepoints
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



# Set up tracing

find_package(LTTngUST REQUIRED)

add_library(${LIBRARY_NAME}_tracepoints
  ${PROJECT_NAME}_tracepoints/tracepoints.cpp
)

target_link_libraries(${LIBRARY_NAME}_tracepoints
  LTTng::UST
)

# Not sure why this is needed
target_include_directories(${LIBRARY_NAME}_tracepoints
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}_tracepoints
)



# Set up executable

add_executable(${EXECUTABLE_NAME}
  main.cpp
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

set_target_properties(${EXECUTABLE_NAME}
  PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
