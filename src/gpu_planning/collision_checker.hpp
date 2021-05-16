#pragma once

#include <memory>
#include <vector>

#include "array.hpp"
#include "logging.hpp"
#include "map.hpp"
#include "robot.hpp"
#include "work_buffer.hpp"

namespace gpu_planning {

class Robot;
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
  WorkBuffer<Configuration, CollisionCheckResult> device_work_buf_;

  std::vector<DeviceMap> mask_bufs_;
  DeviceArray<Map*> mask_buf_handles_;

  DeviceMap* map_;
  DeviceRobot* robot_;
  ObstacleManager* obstacle_manager_;

  Logger* log_;
};

}  // namespace gpu_planning
