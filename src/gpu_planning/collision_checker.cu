#include <cstring>

#include "array_2d.hpp"
#include "collision_checker.hpp"
#include "cuda_util.hpp"
#include "map.hpp"
#include "obstacle_manager.hpp"

namespace gpu_planning {

CollisionCheckResult::CollisionCheckResult() : result{false} {}

CollisionCheckResult::CollisionCheckResult(bool result, uint8_t obstacle_id)
    : result{result}, obstacle_id{obstacle_id} {}

CollisionChecker::CollisionChecker()
    : device_work_buf_{},
      map_{nullptr},
      robot_{nullptr},
      obstacle_manager_{nullptr},
      log_{nullptr} {}

CollisionChecker::CollisionChecker(DeviceMap* map, DeviceRobot* robot,
                                   ObstacleManager* obstacle_manager,
                                   Logger* log)
    : device_work_buf_{100},
      map_{map},
      robot_{robot},
      obstacle_manager_{obstacle_manager},
      log_{log} {}

__global__ void check_collisions(
    Map* map, Robot* robot,
    WorkBlock<Configuration, CollisionCheckResult>* work) {
  for (size_t i = threadIdx.x; i < work->size(); i += blockDim.x) {
    const Pose<float> ee = robot->fk_ee(work->data(i));
    const Cell& cell = map->get(ee.position);

    work->result(i) = CollisionCheckResult(cell.value >= 1.f, cell.id);
  }
}

void CollisionChecker::check(const std::vector<Configuration>& configurations) {
  LOG_DEBUG(log_) << "Checking " << configurations.size()
                  << " configurations for collisions in blocks of "
                  << device_work_buf_.block_size();
  std::vector<CollisionCheckResult> result;
  result.resize(configurations.size());

  device_work_buf_.set_work(configurations.size(), configurations.data(),
                            result.data());

  while (!device_work_buf_.done()) {
    DeviceWorkHandle<Configuration, CollisionCheckResult> work =
        device_work_buf_.next_work_block();
    check_collisions<<<1, 32>>>(map_->device_map(), robot_->device_handle(),
                                work.device_handle());
  }

  for (size_t i = 0; i < result.size(); ++i) {
    if (result[i].result) {
      const std::string obst_name =
          obstacle_manager_->get_obstacle_name(result[i].obstacle_id);
      LOG_DEBUG(log_) << "Configuration " << i << ": X   (" << obst_name << ")";
    } else {
      LOG_DEBUG(log_) << "Configuration " << i << ":   X";
    }
  }
}

}  // namespace gpu_planning
