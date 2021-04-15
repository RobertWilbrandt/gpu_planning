#include "collision_checker.hpp"

#include "configuration.hpp"
#include "robot.hpp"

namespace gpu_planning {

CollisionChecker::CollisionChecker()
    : map_{nullptr}, robot_{nullptr}, log_{nullptr} {}

CollisionChecker::CollisionChecker(Map* map, Robot* robot, Logger* log)
    : map_{map}, robot_{robot}, log_{log} {}

void CollisionChecker::check(const std::vector<Configuration>& configurations) {
  LOG_INFO(log_) << "Checking " << configurations.size() << " configurations";
}

}  // namespace gpu_planning
