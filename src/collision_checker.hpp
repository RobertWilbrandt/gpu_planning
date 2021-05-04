#pragma once

#include <memory>
#include <vector>

#include "array.hpp"
#include "logging.hpp"
#include "robot.hpp"

namespace gpu_planning {

class Robot;
class DeviceMap;

struct CollisionCheckResult {
  __host__ __device__ CollisionCheckResult();
  __host__ __device__ CollisionCheckResult(bool result);

  bool result;
};

class CollisionChecker {
 public:
  CollisionChecker();
  CollisionChecker(DeviceMap* map, DeviceRobot* robot, Logger* log);

  void check(const std::vector<Configuration>& configurations);

 private:
  size_t check_block_size_;
  DeviceArray<Configuration> device_configuration_buf_;
  DeviceArray<CollisionCheckResult> device_result_buf_;

  DeviceMap* map_;
  DeviceRobot* robot_;

  Logger* log_;
};

}  // namespace gpu_planning
