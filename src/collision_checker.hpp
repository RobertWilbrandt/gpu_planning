#pragma once

#include <memory>
#include <vector>

#include "array.hpp"
#include "block_buffer.hpp"
#include "logging.hpp"
#include "robot.hpp"

namespace gpu_planning {

class Robot;
class DeviceMap;
class ObstacleManager;

struct CollisionCheckResult {
  __host__ __device__ CollisionCheckResult();
  __host__ __device__ CollisionCheckResult(bool result, uint8_t obstacle_id);

  bool result;
  uint8_t obstacle_id;
};

class CollisionChecker {
 public:
  CollisionChecker();
  CollisionChecker(DeviceMap* map, DeviceRobot* robot,
                   ObstacleManager* obstacle_manager, Logger* log);

  void check(const std::vector<Configuration>& configurations);

 private:
  size_t check_block_size_;

  WriteBlockBuffer<Configuration> device_configuration_buf_;
  DeviceArray<CollisionCheckResult> device_result_buf_;

  DeviceMap* map_;
  DeviceRobot* robot_;
  ObstacleManager* obstacle_manager_;

  Logger* log_;
};

}  // namespace gpu_planning
