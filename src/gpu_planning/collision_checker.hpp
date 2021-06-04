#pragma once

#include <memory>
#include <vector>

#include "array.hpp"
#include "logging.hpp"
#include "map.hpp"
#include "robot.hpp"
#include "thread_block.hpp"
#include "trajectory.hpp"
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
  __host__ __device__ CollisionChecker();
  __host__ __device__ CollisionChecker(Map* map, Robot* robot,
                                       Array<Map*>* mask_bufs);

  /** Check a set of configurations for collision with the map.
   *
   * This is done by "rendering" the robot collision geometry for each
   * configuration into the provided submaps and then comparing them to the
   * global map.
   *
   * This will use `thread_block.dim_x() * thread_block.dim_y()` thraeds to
   * check single configurations, with `thread_block.dim_z()` configurations
   * getting checked concurrently. All per-thread results are stored in
   * `shared_buf` and reduced to the final results in parallel.
   *
   * @param work Set of input configurations and output results
   * @pre   \code{.cu}
   *          work->size() % thread_block.dim_z() == 0
   *        \endcode
   * @param shared_buf Shared buffer for thread result merging
   * @pre   shared_buf must be larger than \code{.cu}
   *          thread_block.dim_z() * thread_block.dim_y() * thread_block.dim_x()
   *          * sizeof(CollisionCheckResult)
   *        \endcode
   * @param thread_block Thread block layout
   * @pre   `thread_block.dim_x()` has to be a power of 2
   * @pre   `thread_block.dim_y()` has to be a power of 2
   */
  __host__ __device__ void check_configurations(
      WorkBlock<Configuration, CollisionCheckResult>& work, void* shared_buf,
      const ThreadBlock3d& thread_block);

 private:
  Map* map_;
  Robot* robot_;

  Array<Map*>* mask_bufs_;
};

/** Use a CollisionChecker to check configuration collisions
 *
 * @param collision_checker CollisionChecker to use
 * @param work Collisions to check and result buffer
 *
 * @pre Needs to be invoked with at shared buffer of at least size \code{.cu}
 *        blockDim.x * blockDim.y * blockDim.z * sizeof(CollisionCheckResult)
 *      \endcode
 * @pre blockDim.x has to be a power of 2
 * @pre blockDim.y has to be a power of 2
 */
__global__ void collision_checker_check_conf_collision(
    CollisionChecker* collision_checker,
    WorkBlock<Configuration, CollisionCheckResult>* work);

/** Use a CollisionChecker to check segment collisions
 *
 * @param collision_checker CollisionChecker to use
 * @param segments Segments to check and return buffer
 * @param conf_work Work block that can be used as buffer space
 *
 * @pre Needs to be invoked with a shared buffer of at least size \code{.cu}
 *        blockDim.x * blockDim.y * blockDim.z * sizeof(CollisionCheckResult)
 *      \endcode
 * @pre blockDim.x has to be a power of 2
 * @pre blockDim.y has to be a power of 2
 */
__global__ void collision_checker_check_seg_collision(
    CollisionChecker* collision_checker,
    WorkBlock<TrajectorySegment, CollisionCheckResult>* segments,
    WorkBlock<Configuration, CollisionCheckResult>* conf_work);

class DeviceCollisionChecker {
 public:
  DeviceCollisionChecker();
  DeviceCollisionChecker(DeviceMap* map, DeviceRobot* robot, Logger* log);

  DeviceCollisionChecker(const DeviceCollisionChecker& other) = delete;
  DeviceCollisionChecker& operator=(const DeviceCollisionChecker& other) =
      delete;

  /** Check a set of robot configurations for collisions
   *
   * @param configurations Configuration%s to test
   * @param stream CUDA Stream to use for device interactions
   * @return CollisionCheckResult%s for each configuration
   *
   * @note As this function works asynchronously using the provided `stream`,
   *       you have to synchronize `stream` before accessing the result
   */
  std::vector<CollisionCheckResult> check_async(
      const std::vector<Configuration>& configurations, const Stream& stream);

  /** Check a set of robot trajectory segments for collisions
   *
   * @param segments TrajectorySegment%s to test
   * @param stream CUDA Stream to use for device interactions
   * @return CollisionCheckResult%s for each trajectory segment
   *
   * @note As this function works asynchronously using the provided `stream`,
   *       you have to synchronize `stream` before accessing the result
   */
  std::vector<CollisionCheckResult> check_async(
      const std::vector<TrajectorySegment>& segments, const Stream& stream);

 private:
  DeviceHandle<CollisionChecker> collision_checker_;

  WorkBuffer<Configuration, CollisionCheckResult> device_conf_work_buf_;
  WorkBuffer<TrajectorySegment, CollisionCheckResult> device_seg_work_buf_;

  std::vector<DeviceMap> mask_bufs_;
  DeviceArray<Map*> mask_buf_handles_;

  DeviceMap* map_;
  DeviceRobot* robot_;

  Logger* log_;
};

}  // namespace gpu_planning
