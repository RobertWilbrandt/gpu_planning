#include <iostream>

#include "cli.hpp"
#include "cuda_device.hpp"
#include "logging.hpp"

using namespace gpu_planning;

int main(int argc, char *argv[]) {
  CliArgs args = parse_cli_args(argc, argv);

  init_logging(args.verbose);

  logger log = create_logger();

  if (args.list_devices) {
    cuda_list_devices(log);
  } else {
    LOG_INFO(log) << "Done";
  }

  return 0;
}
