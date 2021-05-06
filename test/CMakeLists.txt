find_package(GTest REQUIRED)

set(UNIT_TEST_NAME "${PROJECT_NAME}_unit_tests")
add_executable(${UNIT_TEST_NAME}
  work_buffer.cu
)

target_link_libraries(${UNIT_TEST_NAME}
  PRIVATE
    ${LIBRARY_NAME}
    GTest::GTest
    GTest::Main
)

add_test(
  NAME
    ${UNIT_TEST_NAME}
  COMMAND
    ${UNIT_TEST_NAME}
)
