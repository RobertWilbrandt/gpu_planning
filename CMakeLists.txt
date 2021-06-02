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

#
# Build tests
#

if(${BUILD_TESTS})
  enable_testing()
  add_subdirectory(test)
endif()

#
# Build documentation
#

find_package(Doxygen)

if (DOXYGEN_FOUND)

  set(DOXYGEN_IN "${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile.in")
  set(DOXYGEN_CONF "${CMAKE_CURRENT_BINARY_DIR}/doc/Doxyfile")

  configure_file(
    "${DOXYGEN_IN}"
    "${DOXYGEN_CONF}"
    @ONLY
  )

  add_custom_target(doc
    COMMAND ${DOXYGEN_EXECUTABLE} "${DOXYGEN_CONF}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Generate API documentation with Doxygen"
    VERBATIM
  )

else (DOXYGEN_FOUND)
  message(WARNING "Doxygen not found, cannot build documentation target")
endif()
