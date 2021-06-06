#include <gpu_planning/cli.hpp>
#include <gpu_planning/collision_checker.hpp>
#include <gpu_planning/cuda_device.hpp>
#include <gpu_planning/cuda_util.hpp>
#include <gpu_planning/debug.hpp>
#include <gpu_planning/graph.hpp>
#include <gpu_planning/logging.hpp>
#include <gpu_planning/map.hpp>
#include <gpu_planning/obstacle_manager.hpp>
#include <gpu_planning/robot.hpp>
#include <gpu_planning_tracepoints/tracepoints.hpp>
#include <vector>

using namespace gpu_planning;

int main(int argc, char* argv[]) {
  tracepoint(gpu_planning, my_first_tracepoint, 23, "hi there!");
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

  // Create configuration graph
  Graph<Configuration, CollisionChecker::Result<TrajectorySegment>> conf_graph;
  conf_graph.add_node(Configuration(-M_PI / 2, 0, 0));
  conf_graph.add_node(Configuration(-M_PI / 2, 0, M_PI / 2));
  conf_graph.add_node(Configuration(-2, 2, 0));
  conf_graph.add_node(Configuration(M_PI, 1, -1));
  conf_graph.add_node(Configuration(M_PI / 4, 0, 0));
  conf_graph.add_node(Configuration(M_PI / 4, -0.7, 0));
  conf_graph.add_node(Configuration(M_PI / 4, -0.7, -M_PI / 4 - 0.4));

  const size_t conf_basic = conf_graph.add_node(Configuration(0, 0, 0));
  const size_t conf_seg_start =
      conf_graph.add_node(Configuration(M_PI / 4, -M_PI / 2, 0));
  const size_t conf_seg_end =
      conf_graph.add_node(Configuration(M_PI / 8, -M_PI / 2, M_PI / 4));

  std::vector<Configuration> check_confs;
  for (size_t i = 0; i < conf_graph.num_nodes(); ++i) {
    check_confs.push_back(conf_graph.node(i));
  }

  std::vector<TrajectorySegment> check_segs;
  check_segs.emplace_back(conf_graph.node(conf_seg_start),
                          conf_graph.node(conf_basic));
  check_segs.emplace_back(conf_graph.node(conf_seg_start),
                          conf_graph.node(conf_seg_end));

  // Check configurations and segments
  std::vector<CollisionChecker::Result<Configuration>> conf_check_results =
      collision_checker.check_async(check_confs, collision_stream);
  std::vector<CollisionChecker::Result<TrajectorySegment>> seg_check_results =
      collision_checker.check_async(check_segs, collision_stream);

  // Save image of map to file
  debug_save_state(map, robot, check_confs, check_segs, "test.bmp", &log);

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

  tracepoint(gpu_planning, my_first_tracepoint, 23, "done");

  return 0;
}
