#pragma once

namespace gpu_planning {

struct CliArgs {
  CliArgs();
  CliArgs(bool verbose);

  bool verbose;
};

CliArgs parse_cli_args(int argc, char *argv[]);

}  // namespace gpu_planning
