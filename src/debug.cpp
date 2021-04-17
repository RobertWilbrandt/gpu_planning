#include "debug.hpp"

#include <memory>

#include "bmp.hpp"

namespace gpu_planning {

void debug_print_map(Map& map, size_t max_width, size_t max_height,
                     Logger* log) {
  std::unique_ptr<float[]> buf(new float[max_width * max_height]);
  size_t width;
  size_t height;
  map.get_data(buf.get(), max_width, max_height, &width, &height);

  LOG_DEBUG(log) << "--- " << map.width() / map.resolution() << "x"
                 << map.height() / map.resolution() << " with resolution "
                 << map.resolution() << " (shown as " << width << "x" << height
                 << ") ---";
  for (size_t y = 0; y < height; ++y) {
    std::string line = "|";
    for (size_t x = 0; x < width; ++x) {
      float val = buf[y * width + x];

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

void debug_save_map(Map& map, size_t max_width, size_t max_height,
                    const std::string& path, Logger* log) {
  std::unique_ptr<float[]> buf(new float[max_width * max_height]);
  size_t img_width;
  size_t img_height;
  map.get_data(buf.get(), max_width, max_height, &img_width, &img_height);

  save_bmp(buf.get(), img_width, img_height, path, log);
}

}  // namespace gpu_planning
