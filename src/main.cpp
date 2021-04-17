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

  const size_t map_width = 15;
  const size_t map_height = 10;
  const size_t map_resolution = 25;

  Map map(map_width, map_height, map_resolution, &log);
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

  const size_t img_max_width = map_width * 20;
  const size_t img_max_height = map_height * 20;

  float img_buf[img_max_width * img_max_height];
  size_t img_width;
  size_t img_height;
  map.get_data(img_buf, img_max_width, img_max_height, &img_width, &img_height);

  save_bmp(img_buf, img_width, img_height, "test.bmp", &log);

  return 0;
}
