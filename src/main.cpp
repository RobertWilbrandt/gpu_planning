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

  const Pose<float> map_midpoint(map_width / 2, map_height / 2, 0.f);

  DeviceMap map(map_width * map_resolution, map_height * map_resolution,
                map_resolution, &log);

  DeviceRobot robot(
      Pose<float>((float)map_width / 2, (float)map_height / 2, M_PI / 2), 2.f,
      1.5f, Rectangle(0.5f, 1.0f));

  ObstacleManager obstacle_manager;
  DeviceCollisionChecker collision_checker(&map, &robot, &obstacle_manager,
                                           &log);

  // Create obstacles
  obstacle_manager.add_static_circle(Transform<float>(0, 5, 0) * map_midpoint,
                                     2, "Top Circle");
  obstacle_manager.add_static_rectangle(
      Transform<float>(2, 0, 0) * map_midpoint, 1, 2, "Close Rectangle");
  obstacle_manager.add_static_rectangle(
      Transform<float>(6, 0, 0) * map_midpoint, 4, 1.5, "Far Right");
  obstacle_manager.add_static_circle(
      Transform<float>(-map_width / 4, 2, 0) * map_midpoint, 2, "Big Circle");

  obstacle_manager.add_static_rectangle(
      Transform<float>(-6, -3, 0.1) * map_midpoint, 2, 2, "Slightly rotated");
  obstacle_manager.add_static_rectangle(
      Transform<float>(-3, -3, 0.3) * map_midpoint, 2, 2, "More rotated");
  obstacle_manager.add_static_rectangle(
      Transform<float>(0, -3, 0.5) * map_midpoint, 2, 2, "Most rotated");

  obstacle_manager.insert_in_map(map);

  // Print map
  debug_print_map(map, 40, 20, &log);

  // Create and check configurations
  std::vector<Configuration> configurations;
  configurations.emplace_back(0, 0, 0);
  configurations.emplace_back(M_PI / 4, -M_PI / 2, 0);
  configurations.emplace_back(-M_PI / 2, 0, 0);
  configurations.emplace_back(-M_PI / 2, 0, M_PI / 2);
  configurations.emplace_back(-2, 2, 0);
  configurations.emplace_back(M_PI, 1, -1);
  configurations.emplace_back(M_PI / 4, 0, 0);
  configurations.emplace_back(M_PI / 4, -0.7, 0);
  configurations.emplace_back(M_PI / 4, -0.7, -M_PI / 4 - 0.4);

  collision_checker.check(configurations);

  // Save image of map to file
  debug_save_state(map, robot, configurations, "test.bmp", &log);

  return 0;
}
