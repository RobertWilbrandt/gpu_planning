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

void debug_save_state(DeviceMap& map, DeviceRobot& robot,
                      const std::vector<Configuration>& configurations,
                      const std::string& path, Logger* log) {
  const HostMap host_map = map.load_to_host();
  const Map& raw_map = host_map.map();
  const Box<size_t> map_area = raw_map.data()->area();

  // Draw map data
  Image img(map_area.width(), map_area.height());
  for (size_t y = 0; y < map_area.height(); ++y) {
    for (size_t x = 0; x < map_area.width(); ++x) {
      const Position<size_t> pos(x, y);
      const char scaled_value =
          static_cast<char>(host_map.map().data()->at(x, y).value * 255);

      img.pixel(pos) = Color(255 - scaled_value, 255, 255 - scaled_value);
    }
  }

  // draw ee geometries
  for (const Configuration& conf : configurations) {
    const Pose<float> ee = robot.fk_ee(conf);
    const Rectangle ee_rect = robot.robot().ee();

    const Box<float> ee_bb = ee_rect.bounding_box(ee.orientation)
                                 .translate(ee.position.from_origin());
    const Box<size_t> ee_mask(
        map_area.clamp(raw_map.to_index(ee_bb.lower_left)),
        map_area.clamp(raw_map.to_index(ee_bb.upper_right)));

    for (size_t y = ee_mask.lower_left.y; y <= ee_mask.upper_right.y; ++y) {
      for (size_t x = ee_mask.lower_left.x; x <= ee_mask.upper_right.x; ++x) {
        const Position<size_t> pos(x, y);
        const Position<float> norm_pos =
            (Pose<float>(raw_map.from_index(pos), 0).from_origin() *
             ee.from_origin().inverse() * Pose<float>())
                .position;

        if (ee_rect.is_inside(norm_pos)) {
          img.pixel(pos) = Color(180, 180, 180);
        }
      }
    }
  }

  // draw robot base
  const Box<size_t> img_area = img.area();
  const Position<size_t> img_base = raw_map.to_index(robot.base().position);
  img.draw_marker(img_base, Color::BLUE);

  // draw FK markers and segments
  for (const Configuration& conf : configurations) {
    const Position<size_t> img_elbow =
        raw_map.to_index(robot.fk_elbow(conf).position);
    const Position<size_t> img_ee =
        raw_map.to_index(robot.fk_ee(conf).position);

    img.draw_line(img_base, img_elbow, Color(100, 100, 100));
    img.draw_line(img_elbow, img_ee, Color(150, 150, 150));

    img.draw_marker(img_elbow, Color::YELLOW);
    img.draw_marker(img_ee, Color::RED);
  }

  // Save image to file
  img.save_bmp(path);
}

}  // namespace gpu_planning
