#include <gpu_planning/cli.hpp>
#include <gpu_planning/collision_checker.hpp>
#include <gpu_planning/cuda_device.hpp>
#include <gpu_planning/debug.hpp>
#include <gpu_planning/logging.hpp>
#include <gpu_planning/map.hpp>
#include <gpu_planning/obstacle_manager.hpp>
#include <gpu_planning/robot.hpp>
#include <vector>

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

  uint8_t id_cnt = 0;

  Position<float> map_midpoint(map_width / 2, map_height / 2);

  ObstacleManager obstacle_manager;
  obstacle_manager.add_static_circle(map_midpoint + Translation<float>(0, 5), 2,
                                     "Top Circle");
  obstacle_manager.add_static_rectangle(map_midpoint + Translation<float>(2, 0),
                                        1, 2, "Close Rectangle");
  obstacle_manager.add_static_rectangle(map_midpoint + Translation<float>(6, 0),
                                        4, 1.5, "Far right");
  obstacle_manager.add_static_circle(
      map_midpoint + Translation<float>(-map_width / 4, 2), 2, "Big Circle");
  obstacle_manager.add_static_rectangle(
      map_midpoint + Translation<float>(-map_width / 4, -2), 2, 2,
      "Left Rectangle");

  DeviceMap map(map_width, map_height, map_resolution, &log);

  obstacle_manager.insert_in_map(map);

  DeviceRobot robot(
      Pose<float>((float)map_width / 2, (float)map_height / 2, M_PI / 2), 2.f,
      1.5f, 0.5f, 0.1f);
  CollisionChecker collision_checker(&map, &robot, &obstacle_manager, &log);

  debug_print_map(map, 40, 20, &log);

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
