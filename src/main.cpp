#include <iostream>
#include <vector>

#include "bmp.hpp"
#include "cli.hpp"
#include "collision_checker.hpp"
#include "configuration.hpp"
#include "cuda_device.hpp"
#include "logging.hpp"
#include "map.hpp"
#include "robot.hpp"

using namespace gpu_planning;

int main(int argc, char* argv[]) {
  CliArgs args = parse_cli_args(argc, argv);

  init_logging(args.verbose);

  Logger log = create_logger();

  try {
    if (args.device >= 0) {
      cuda_set_device(args.device, &log);
    }

    if (args.list_devices) {
      cuda_list_devices(&log);
    } else {
      LOG_INFO(&log) << "Done";
    }
  } catch (std::runtime_error& ex) {
    std::cerr << ex.what() << std::endl;
  }

  Map map(15, 10, 10, &log);
  map.print_debug(40, 20);
  map.add_obstacle_circle(3, 2, 1);
  map.add_obstacle_circle(5, 5, 3);
  map.add_obstacle_rect(12, 7.5, 4, 2);
  map.print_debug(40, 20);
  Robot robot;
  CollisionChecker collision_checker(&map, &robot, &log);

  std::vector<Configuration> configurations;
  configurations.emplace_back(1, 1, 1);
  configurations.emplace_back(1, 2, 3);

  collision_checker.check(configurations);

  save_bmp(nullptr, 0, 0, "test.bmp", &log);

  return 0;
}
