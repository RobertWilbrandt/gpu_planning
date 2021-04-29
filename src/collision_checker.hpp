#pragma once

#include <memory>
#include <vector>

#include "device_array.hpp"
#include "device_robot.cuh"
#include "logging.hpp"
#include "robot.hpp"

namespace gpu_planning {

class Robot;
class Map;

struct CollisionCheckResult {
  __host__ __device__ CollisionCheckResult();
  __host__ __device__ CollisionCheckResult(bool result);

  bool result;
};

class CollisionChecker {
 public:
  CollisionChecker();
  CollisionChecker(Map* map, Robot* robot, Logger* log);

  void check(const std::vector<Configuration>& configurations);

 private:
  size_t conf_buf_len_;

  std::vector<DeviceConfiguration> configuration_buf_;
  std::vector<CollisionCheckResult> result_buf_;

  DeviceArray<DeviceConfiguration> device_configuration_buf_;
  DeviceArray<CollisionCheckResult> device_result_buf_;

  Map* map_;
  Robot* robot_;

  Logger* log_;
};

}  // namespace gpu_planning
