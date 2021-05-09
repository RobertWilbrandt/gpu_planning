#pragma once

#include <string>
#include <vector>

#include "geometry.hpp"
#include "logging.hpp"
#include "map.hpp"
#include "robot.hpp"

namespace gpu_planning {

class Overlay {
 public:
  enum class OverlayClass { NONE = 0, BASE, ELBOW, EE, S1, S2, EE_RECT };

  Overlay();
  Overlay(size_t width, size_t height);

  void draw_point(const Position<size_t>& pos, OverlayClass cls);
  void draw_marker(const Position<size_t>& pos, OverlayClass cls);
  void draw_line(const Position<size_t>& from, const Position<size_t>& to,
                 OverlayClass cls);

  OverlayClass get(const Position<size_t>& pos) const;

 private:
  void try_draw_point(const Position<size_t>& pos, OverlayClass cls);

  size_t width_;
  size_t height_;
  std::unique_ptr<OverlayClass[]> data_;
};

void debug_print_map(DeviceMap& map, size_t max_width, size_t max_height,
                     Logger* log);
void debug_save_state(DeviceMap& map, DeviceRobot& robot,
                      const std::vector<Configuration>& configurations,
                      const std::string& path, Logger* log);

}  // namespace gpu_planning
