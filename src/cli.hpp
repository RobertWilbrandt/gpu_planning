#pragma once

namespace gpu_planning {

struct CliArgs {
  CliArgs();
  CliArgs(bool verbose, bool list_devices);

  bool verbose;
  bool list_devices;
};

CliArgs parse_cli_args(int argc, char *argv[]);

}  // namespace gpu_planning
