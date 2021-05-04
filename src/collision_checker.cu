#include <cstring>

#include "array_2d.hpp"
#include "collision_checker.hpp"
#include "cuda_util.hpp"
#include "map.hpp"

namespace gpu_planning {

CollisionCheckResult::CollisionCheckResult() : result{false} {}

CollisionCheckResult::CollisionCheckResult(bool result) : result{result} {}

CollisionChecker::CollisionChecker()
    : check_block_size_{0},
      device_configuration_buf_{},
      device_result_buf_{},
      map_{nullptr},
      robot_{nullptr},
      log_{nullptr} {}

CollisionChecker::CollisionChecker(DeviceMap* map, DeviceRobot* robot,
                                   Logger* log)
    : check_block_size_{100},
      device_configuration_buf_{check_block_size_},
      device_result_buf_{check_block_size_},
      map_{map},
      robot_{robot},
      log_{log} {}

__global__ void check_collisions(Map* map, Robot* robot,
                                 Array<Configuration>* configurations,
                                 Array<CollisionCheckResult>* results,
                                 size_t num_checks) {
  for (size_t i = threadIdx.x; i < num_checks; i += blockDim.x) {
    const Pose<float> ee = robot->fk_ee((*configurations)[i]);
    (*results)[i] = map->get(ee.position).value >= 1.f;
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

    Array<const Configuration> configuration_block(
        &configurations[i * check_block_size_], block_remaining);
    device_configuration_buf_.memcpy_set(configuration_block);
    check_collisions<<<1, 32>>>(map_->device_map(), robot_->device_handle(),
                                device_configuration_buf_.device_handle(),
                                device_result_buf_.device_handle(),
                                block_remaining);

    Array<CollisionCheckResult> result_block(&result[i * check_block_size_],
                                             block_remaining);
    device_result_buf_.memcpy_get(result_block);
  }

  for (size_t i = 0; i < result.size(); ++i) {
    if (result[i].result) {
      LOG_DEBUG(log_) << "Configuration " << i << ": X";
    } else {
      LOG_DEBUG(log_) << "Configuration " << i << ":   X";
    }
  }
}

}  // namespace gpu_planning
