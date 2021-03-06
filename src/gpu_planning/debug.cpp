#include "debug.hpp"

#include <stdint.h>
#include <string.h>

#include <fstream>
#include <memory>
#include <unordered_map>

#include "geometry.hpp"
#include "image.hpp"

namespace gpu_planning {

void debug_print_map(DeviceMap& map, size_t max_width, size_t max_height,
                     Logger* log) {
  std::unique_ptr<float[]> buf(new float[max_width * max_height]);
  size_t width;
  size_t height;
  map.get_data(buf.get(), max_width, max_height, &width, &height);

  LOG_DEBUG(log) << "--- " << map.width() << "x" << map.height()
                 << " with resolution " << map.resolution() << " (shown as "
                 << width << "x" << height << ") ---";
  for (size_t y = 0; y < height; ++y) {
    std::string line = "|";
    for (size_t x = 0; x < width; ++x) {
      float val = buf[(height - y - 1) * width + x];

      if (val < 0.5) {
        line += ' ';
      } else if (val < 1.0) {
        line += 'X';
      } else {
        line += '#';
      }
    }
    LOG_DEBUG(log) << line << "|";
  }
  LOG_DEBUG(log) << "---";
}

void debug_save_state(
    DeviceMap& map, DeviceRobot& robot,
    const Graph<Configuration, CollisionChecker::Result<TrajectorySegment>>&
        conf_graph,
    const std::string& path, Logger* log) {
  const HostMap host_map = map.load_to_host();
  const Box<size_t> map_area = host_map.data()->area();

  // Draw map data
  Image img(map_area.width(), map_area.height());
  for (size_t y = 0; y < map_area.height(); ++y) {
    for (size_t x = 0; x < map_area.width(); ++x) {
      const Position<size_t> pos(x, y);
      const char scaled_value =
          static_cast<char>(host_map.data()->at(x, y).value * 255);

      img.pixel(pos) = Color(255 - scaled_value, 255, 255 - scaled_value);
    }
  }

  // draw ee geometries
  for (size_t i = 0; i < conf_graph.num_nodes(); ++i) {
    const Configuration& conf = conf_graph.node(i);

    const Pose<float> ee = robot.fk_ee(conf);
    const Rectangle ee_rect = robot.robot().ee();

    Array2d<Color> img_as_array = img.as_array();
    shape_insert_into<Rectangle, Color>(
        ee_rect, ee, img_as_array, host_map.resolution(), Color(180, 180, 180),
        ThreadBlock3d::host().slice_z());
  }

  const Box<size_t> img_area = img.area();
  const Position<size_t> img_base = host_map.to_index(robot.base().position);

  // draw FK markers and segments
  for (size_t i = 0; i < conf_graph.num_edges(); ++i) {
    const Configuration& seg_start = conf_graph.node(conf_graph.edge_from(i));
    const Configuration& seg_end = conf_graph.node(conf_graph.edge_to(i));
    const CollisionChecker::Result<TrajectorySegment> seg_result =
        conf_graph.edge(i);

    const Position<size_t> img_start =
        host_map.to_index(robot.fk_ee(seg_start).position);
    const Position<size_t> img_end =
        host_map.to_index(robot.fk_ee(seg_end).position);

    Color line_color(0, 0, 200);
    if (seg_result.result) {
      line_color = Color::RED;
    }

    img.draw_line(img_start, img_end, line_color, true);
  }

  for (size_t i = 0; i < conf_graph.num_nodes(); ++i) {
    const Configuration& conf = conf_graph.node(i);

    const Position<size_t> img_elbow =
        host_map.to_index(robot.fk_elbow(conf).position);
    const Position<size_t> img_ee =
        host_map.to_index(robot.fk_ee(conf).position);

    img.draw_line(img_base, img_elbow, Color(100, 100, 100));
    img.draw_line(img_elbow, img_ee, Color(150, 150, 150));

    img.draw_marker(img_elbow, Color::YELLOW);
    img.draw_marker(img_ee, Color::RED);
  }

  img.draw_marker(img_base, Color::BLUE);

  // Save image to file
  img.save_bmp(path);
}

}  // namespace gpu_planning
