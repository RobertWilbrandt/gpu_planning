#pragma once

#include <memory>
#include <vector>

#include "logging.hpp"
#include "robot.hpp"

namespace gpu_planning {

class Robot;
class Map;

class CollisionChecker {
 public:
  CollisionChecker();
  CollisionChecker(Map* map, Robot* robot, Logger* log);

  ~CollisionChecker();

  void check(const std::vector<Configuration>& configurations);

 private:
  size_t conf_buf_len_;

  void* conf_buf_;
  void* dev_conf_buf_;

  Map* map_;
  Robot* robot_;

  Logger* log_;
};

}  // namespace gpu_planning
