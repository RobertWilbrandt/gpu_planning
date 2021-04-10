#include "cli.hpp"

#include <boost/program_options.hpp>
#include <cstdlib>
#include <iostream>

namespace gpu_planning {

CliArgs::CliArgs() : verbose{false} {}

CliArgs::CliArgs(bool verbose) : verbose{verbose} {}

CliArgs parse_cli_args(int argc, char* argv[]) {
  namespace po = boost::program_options;

  po::options_description desc("Allowed options");
  desc.add_options()("help", "Print help message")("verbose,v",
                                                   "Print verbose output");

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

  return CliArgs(verbose);
}

}  // namespace gpu_planning
