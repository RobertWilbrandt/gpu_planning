#include <gpu_planning/cli.hpp>
#include <gpu_planning/collision_checker.hpp>
#include <gpu_planning/cuda_device.hpp>
#include <gpu_planning/cuda_util.hpp>
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

  LOG_INFO(&log) << args.list_devices;

  try {
    if (args.device >= 0) {
      cuda_set_device(args.device, &log);
    }

    if (args.list_devices) {
      cuda_list_devices(&log);
      return 0;
    } else {
      LOG_INFO(&log) << "Done";
    }
  } catch (std::runtime_error& ex) {
    std::cerr << ex.what() << std::endl;
    return 1;
  }

  // Create CUDA streams
  Stream collision_stream = Stream::create();

  // Set up map
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
  DeviceCollisionChecker collision_checker(&map, &robot, &log);

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
  configurations.emplace_back(-M_PI / 2, 0, 0);
  configurations.emplace_back(-M_PI / 2, 0, M_PI / 2);
  configurations.emplace_back(-2, 2, 0);
  configurations.emplace_back(M_PI, 1, -1);
  configurations.emplace_back(M_PI / 4, 0, 0);
  configurations.emplace_back(M_PI / 4, -0.7, 0);
  configurations.emplace_back(M_PI / 4, -0.7, -M_PI / 4 - 0.4);

  // Create segments
  std::vector<TrajectorySegment> segments;
  Configuration conf_basic(0, 0, 0);
  Configuration conf_seg_start(M_PI / 4, -M_PI / 2, 0);
  Configuration conf_seg_end(M_PI / 8, -M_PI / 2, M_PI / 4);
  configurations.push_back(conf_basic);
  configurations.push_back(conf_seg_start);
  configurations.push_back(conf_seg_end);
  segments.emplace_back(conf_seg_start, conf_basic);
  segments.emplace_back(conf_seg_start, conf_seg_end);

  std::vector<CollisionCheckResult> conf_check_results =
      collision_checker.check_async(configurations, collision_stream);
  std::vector<CollisionCheckResult> seg_check_results =
      collision_checker.check_async(segments, collision_stream);

  // Save image of map to file
  debug_save_state(map, robot, configurations, segments, "test.bmp", &log);

  // Print collision check results, sync before because we checked
  // asynchronously
  collision_stream.sync();
  for (size_t i = 0; i < conf_check_results.size(); ++i) {
    if (conf_check_results[i].result) {
      const std::string obst_name =
          obstacle_manager.get_obstacle_name(conf_check_results[i].obstacle_id);
      LOG_DEBUG(&log) << "Configuration " << i << ": X   (" << obst_name << ")";
    } else {
      LOG_DEBUG(&log) << "Configuration " << i << ":   X";
    }
  }

  for (size_t i = 0; i < seg_check_results.size(); ++i) {
    if (seg_check_results[i].result) {
      const std::string obst_name =
          obstacle_manager.get_obstacle_name(seg_check_results[i].obstacle_id);
      LOG_DEBUG(&log) << "Segment " << i << ": X   (" << obst_name << ")";
    } else {
      LOG_DEBUG(&log) << "Segment " << i << ":   X";
    }
  }

  return 0;
}
