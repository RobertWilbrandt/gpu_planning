#include "cli.hpp"

#include <boost/program_options.hpp>
#include <cstdlib>
#include <iostream>

namespace gpu_planning {

CliArgs::CliArgs() : verbose{false} {}

CliArgs::CliArgs(bool verbose, bool list_devices) : verbose{verbose} {}

CliArgs parse_cli_args(int argc, char* argv[]) {
  namespace po = boost::program_options;

  po::options_description generic_options("Generic options");
  generic_options.add_options()("help,h", "Print help message")(
      "verbose,v", "Print verbose output");

  po::options_description cuda_options("CUDA options");
  cuda_options.add_options()("list-devices,l", "List available CUDA devices");

  po::options_description desc("Available program options");
  desc.add(generic_options).add(cuda_options);

  po::variables_map vm;

  try {
    po::store(
        po::command_line_parser(argc, argv).options(desc).positional({}).run(),
        vm);
    po::notify(vm);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    exit(1);
  }

  if (vm.count("help")) {
    std::cout << desc << '\n';
    exit(0);
  }

  bool verbose = false;
  if (vm.count("verbose")) {
    verbose = true;
  }

  bool list_devices = false;
  if (vm.count("list-devices")) {
    list_devices = true;
  }

  return CliArgs(verbose, list_devices);
}

}  // namespace gpu_planning
