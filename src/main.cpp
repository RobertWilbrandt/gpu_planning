#include <iostream>

#include "cli.hpp"
#include "configuration.hpp"
#include "cuda_device.hpp"
#include "logging.hpp"

using namespace gpu_planning;

int main(int argc, char* argv[]) {
  CliArgs args = parse_cli_args(argc, argv);

  init_logging(args.verbose);

  logger log = create_logger();

  try {
    if (args.device >= 0) {
      cuda_set_device(args.device, log);
    }

    if (args.list_devices) {
      cuda_list_devices(log);
    } else {
      LOG_INFO(log) << "Done";
    }
  } catch (std::runtime_error& ex) {
    std::cerr << ex.what() << std::endl;
  }

  Configuration test_conf(1.3245, 21.3456, 1.5);
  LOG_INFO(log) << "Test configuration: " << test_conf;

  return 0;
}
