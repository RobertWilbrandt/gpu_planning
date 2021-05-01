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
    : conf_buf_len_{0},
      configuration_buf_{},
      result_buf_{},
      device_configuration_buf_{},
      device_result_buf_{},
      map_{nullptr},
      robot_{nullptr},
      log_{nullptr} {}

CollisionChecker::CollisionChecker(Map* map, Robot* robot, Logger* log)
    : conf_buf_len_{100},
      configuration_buf_{},
      result_buf_{},
      device_configuration_buf_{conf_buf_len_},
      device_result_buf_{conf_buf_len_},
      map_{map},
      robot_{robot},
      log_{log} {
  configuration_buf_.resize(conf_buf_len_);
  result_buf_.resize(conf_buf_len_);
}

__global__ void check_collisions(
    DeviceMap* map, DeviceRobot* robot,
    DeviceArrayHandle<DeviceConfiguration>* configurations,
    DeviceArrayHandle<CollisionCheckResult>* results) {
  const size_t resolution = map->resolution();
  const Device2dArrayHandle<float>* map_data = map->data();

  for (size_t i = threadIdx.x; i < configurations->size(); i += blockDim.x) {
    DevicePose ee = robot->fk_ee(&(*configurations)[i]);

    size_t x = ee.x * resolution;
    size_t y = ee.y * resolution;

    (*results)[i] = map_data->get(x, y) >= 1.f;
  }
}

void CollisionChecker::check(const std::vector<Configuration>& configurations) {
  LOG_DEBUG(log_) << "Checking " << configurations.size()
                  << " configurations for collisions in blocks of "
                  << conf_buf_len_;

  size_t num_iterations = (configurations.size() - 1) / conf_buf_len_ + 1;
  for (size_t i = 0; i < num_iterations; ++i) {
    size_t block_remaining =
        min(conf_buf_len_, configurations.size() - i * conf_buf_len_);

    for (size_t j = 0; j < block_remaining; ++j) {
      const Configuration& conf = configurations[i * conf_buf_len_ + j];
      configuration_buf_[j] =
          DeviceConfiguration(conf.joints[0], conf.joints[1], conf.joints[2]);
    }

    device_configuration_buf_.memcpy_set(configuration_buf_.data());
    check_collisions<<<1, 32>>>(map_->device_map(), robot_->device_robot(),
                                device_configuration_buf_.device_handle(),
                                device_result_buf_.device_handle());

    device_result_buf_.memcpy_get(result_buf_.data());

    for (size_t j = 0; j < block_remaining; ++j) {
      if (result_buf_[j].result) {
        LOG_DEBUG(log_) << "Configuration " << i * conf_buf_len_ + j << ": X";
      } else {
        LOG_DEBUG(log_) << "Configuration " << i * conf_buf_len_ + j << ":   X";
      }
    }
  }
}

}  // namespace gpu_planning
