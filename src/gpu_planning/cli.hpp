#pragma once

namespace gpu_planning {

struct CliArgs {
  CliArgs();
  CliArgs(bool verbose, bool list_devices, int device);

  bool verbose;
  bool list_devices;

  int device;
};

CliArgs parse_cli_args(int argc, char *argv[]);

}  // namespace gpu_planning
