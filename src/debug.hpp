#pragma once

#include <string>
#include <vector>

#include "logging.hpp"
#include "map.hpp"
#include "robot.hpp"

namespace gpu_planning {

void debug_print_map(Map& map, size_t max_width, size_t max_height,
                     Logger* log);
void debug_save_state(Map& map, Robot& robot,
                      const std::vector<Configuration>& configurations,
                      size_t max_width, size_t max_height,
                      const std::string& path, Logger* log);

}  // namespace gpu_planning
