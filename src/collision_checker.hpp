#pragma once

#include <vector>

#include "configuration.hpp"
#include "logging.hpp"

namespace gpu_planning {

class Robot;
class Map;

class CollisionChecker {
 public:
  CollisionChecker();
  CollisionChecker(Map* map, Robot* robot, Logger* log);

  void check(const std::vector<Configuration>& configurations);

 private:
  Map* map_;
  Robot* robot_;

  Logger* log_;
};

}  // namespace gpu_planning
