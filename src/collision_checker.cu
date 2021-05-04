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
    : check_block_size_{0},
      device_configuration_buf_{},
      device_result_buf_{},
      map_{nullptr},
      robot_{nullptr},
      obstacle_manager_{nullptr},
      log_{nullptr} {}

CollisionChecker::CollisionChecker(DeviceMap* map, DeviceRobot* robot,
                                   ObstacleManager* obstacle_manager,
                                   Logger* log)
    : check_block_size_{100},
      device_configuration_buf_{check_block_size_},
      device_result_buf_{check_block_size_},
      map_{map},
      robot_{robot},
      obstacle_manager_{obstacle_manager},
      log_{log} {}

__global__ void check_collisions(Map* map, Robot* robot,
                                 Array<Configuration>* configurations,
                                 Array<CollisionCheckResult>* results,
                                 size_t num_checks) {
  for (size_t i = threadIdx.x; i < num_checks; i += blockDim.x) {
    const Pose<float> ee = robot->fk_ee((*configurations)[i]);
    const Cell& cell = map->get(ee.position);

    (*results)[i] = CollisionCheckResult(cell.value >= 1.f, cell.id);
  }
}

void CollisionChecker::check(const std::vector<Configuration>& configurations) {
  LOG_DEBUG(log_) << "Checking " << configurations.size()
                  << " configurations for collisions in blocks of "
                  << check_block_size_;
  std::vector<CollisionCheckResult> result;
  result.resize(configurations.size());

  size_t num_iterations = (configurations.size() - 1) / check_block_size_ + 1;
  for (size_t i = 0; i < num_iterations; ++i) {
    size_t block_remaining =
        min(check_block_size_, configurations.size() - i * check_block_size_);

    device_configuration_buf_.memcpy_set(configurations, i * check_block_size_,
                                         block_remaining);
    check_collisions<<<1, 32>>>(map_->device_map(), robot_->device_handle(),
                                device_configuration_buf_.device_handle(),
                                device_result_buf_.device_handle(),
                                block_remaining);
    device_result_buf_.memcpy_get(result, i * check_block_size_,
                                  block_remaining);
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
