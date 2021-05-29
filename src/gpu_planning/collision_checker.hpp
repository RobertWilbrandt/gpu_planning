#pragma once

#include <memory>
#include <vector>

#include "array.hpp"
#include "logging.hpp"
#include "map.hpp"
#include "robot.hpp"
#include "work_buffer.hpp"
#include "work_layout.hpp"

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
  __host__ __device__ CollisionChecker();
  __host__ __device__ CollisionChecker(Map* map, Robot* robot,
                                       Array<Map*>* mask_bufs);

  /** Check a set of configurations for collision with the map.
   *
   * @param work Set of input configurations and output results, size needs to
   *        be a multiple of work_layout.stride_z
   * @param shared_buf Shared buffer for threads, has to have size of at least
   *        work_layout.stride_z * work_layout.stride_y * work_layout.stride_x *
   *        sizeof(CollisionCheckResult)
   * @param work_layout Thread block layout, work_layout.stride_y and
   *        work_layout.x have to be powers of two
   */
  __device__ void check_configurations(
      WorkBlock<Configuration, CollisionCheckResult>& work, void* shared_buf,
      const WorkLayout3d& work_layout);

 private:
  Map* map_;
  Robot* robot_;

  Array<Map*>* mask_bufs_;
};

class DeviceCollisionChecker {
 public:
  DeviceCollisionChecker();
  DeviceCollisionChecker(DeviceMap* map, DeviceRobot* robot,
                         ObstacleManager* obstacle_manager, Logger* log);

  DeviceCollisionChecker(const DeviceCollisionChecker& other) = delete;
  DeviceCollisionChecker& operator=(const DeviceCollisionChecker& other) =
      delete;

  void check(const std::vector<Configuration>& configurations,
             cudaStream_t stream);

 private:
  DeviceHandle<CollisionChecker> collision_checker_;

  WorkBuffer<Configuration, CollisionCheckResult> device_work_buf_;

  std::vector<DeviceMap> mask_bufs_;
  DeviceArray<Map*> mask_buf_handles_;

  DeviceMap* map_;
  DeviceRobot* robot_;
  ObstacleManager* obstacle_manager_;

  Logger* log_;
};

}  // namespace gpu_planning
