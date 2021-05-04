#include <vector>

#include "cli.hpp"
#include "collision_checker.hpp"
#include "cuda_device.hpp"
#include "debug.hpp"
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

  const float map_width = 15;
  const size_t map_height = 10;
  const size_t map_resolution = 25;

  DeviceMap map(map_width, map_height, map_resolution, &log);
  map.add_obstacle_circle(map_width / 2, map_height / 2 + 5, 2);
  map.add_obstacle_rect(map_width / 2 + 2, map_height / 2, 1, 2);
  map.add_obstacle_rect(map_width / 2 + 6, map_height / 2, 4, 1.5);
  map.add_obstacle_circle(map_width / 4, map_height / 2 + 2, 2);
  map.add_obstacle_rect(map_width / 4, map_height / 2 - 2, 2, 2);

  debug_print_map(map, 40, 20, &log);

  DeviceRobot robot(
      Pose<float>((float)map_width / 2, (float)map_height / 2, M_PI / 2), 2.f,
      1.5f, 0.5f, 0.1f);
  CollisionChecker collision_checker(&map, &robot, &log);

  std::vector<Configuration> configurations;
  configurations.emplace_back(0, 0, 0);
  configurations.emplace_back(M_PI / 4, -M_PI / 2, 0);
  configurations.emplace_back(-M_PI / 2, 0, 0);
  configurations.emplace_back(-M_PI / 2, 0, M_PI / 2);
  configurations.emplace_back(-2, 2, 0);
  configurations.emplace_back(M_PI, 1, 0);

  collision_checker.check(configurations);

  debug_save_state(map, robot, configurations, map_width * 20, map_height * 20,
                   "test.bmp", &log);

  return 0;
}
