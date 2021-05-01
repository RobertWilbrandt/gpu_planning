#include <cstring>

#include "collision_checker.hpp"
#include "cuda_util.hpp"
#include "device_2d_array.hpp"
#include "device_map.hpp"
#include "device_robot.cuh"
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

CollisionChecker::CollisionChecker(Map* map, Robot* robot, Logger* log)
    : check_block_size_{100},
      device_configuration_buf_{check_block_size_},
      device_result_buf_{check_block_size_},
      map_{map},
      robot_{robot},
      log_{log} {}

__global__ void check_collisions(
    DeviceMap* map, DeviceRobot* robot,
    DeviceArrayHandle<Configuration>* configurations,
    DeviceArrayHandle<CollisionCheckResult>* results, size_t num_checks) {
  const size_t resolution = map->resolution();
  const Device2dArrayHandle<float>* map_data = map->data();

  for (size_t i = threadIdx.x; i < num_checks; i += blockDim.x) {
    Pose<float> ee = robot->fk_ee((*configurations)[i]);

    size_t x = ee.position.x * resolution;
    size_t y = ee.position.y * resolution;

    (*results)[i] = map_data->get(x, y) >= 1.f;
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

    device_configuration_buf_.memcpy_set(&configurations[i * check_block_size_],
                                         block_remaining);
    check_collisions<<<1, 32>>>(map_->device_map(), robot_->device_robot(),
                                device_configuration_buf_.device_handle(),
                                device_result_buf_.device_handle(),
                                block_remaining);

    device_result_buf_.memcpy_get(&result[i * check_block_size_],
                                  block_remaining);
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
