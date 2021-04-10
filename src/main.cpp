#include <iostream>

#include "logging.hpp"

using namespace gpu_planning;

int main(int argc, char *argv[]) {
  init_logging();

  logger log = create_logger();

  LOG_DEBUG(log) << "debug";
  LOG_INFO(log) << "info";
  LOG_WARN(log) << "warning";
  LOG_ERROR(log) << "error";

  return 0;
}
