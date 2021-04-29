#include <cstring>

#include "collision_checker.hpp"
#include "cuda_util.hpp"
#include "device_2d_array.cuh"
#include "device_map.cuh"
#include "device_robot.cuh"
#include "map.hpp"

namespace gpu_planning {

CollisionChecker::CollisionChecker()
    : conf_buf_len_{0},
      conf_buf_{nullptr},
      result_buf_{nullptr},
      dev_conf_buf_{nullptr},
      dev_result_buf_{nullptr},
      map_{nullptr},
      robot_{nullptr},
      log_{nullptr} {}

CollisionChecker::CollisionChecker(Map* map, Robot* robot, Logger* log)
    : conf_buf_len_{100},
      conf_buf_{new DeviceConfiguration[conf_buf_len_]},
      result_buf_{new bool[conf_buf_len_]},
      dev_conf_buf_{nullptr},
      dev_result_buf_{nullptr},
      map_{map},
      robot_{robot},
      log_{log} {
  CHECK_CUDA(
      cudaMalloc(&dev_conf_buf_, conf_buf_len_ * sizeof(DeviceConfiguration)),
      "Could not allocate configuration buffer for collision checker");
  CHECK_CUDA(cudaMalloc(&dev_result_buf_, conf_buf_len_ * sizeof(bool)),
             "Could not allocate result buffer for collision checker");
}

CollisionChecker::~CollisionChecker() {
  if (conf_buf_ != nullptr) {
    delete[](DeviceConfiguration*) conf_buf_;
  }
  if (result_buf_ != nullptr) {
    delete[](bool*) result_buf_;
  }
  if (dev_conf_buf_ != nullptr) {
    CHECK_CUDA(cudaFree(dev_conf_buf_),
               "Could not free configuration buffer of collision checker");
  }
  if (dev_result_buf_ != nullptr) {
    CHECK_CUDA(cudaFree(dev_result_buf_),
               "Could not free result buffer of collision checker");
  }
}

__global__ void check_collisions(DeviceMap* map, DeviceRobot* robot,
                                 DeviceConfiguration* confs, bool* results,
                                 size_t conf_count) {
  const size_t resolution = map->resolution();
  const Device2dArray* map_data = map->data();

  for (size_t i = threadIdx.x; i < conf_count; i += blockDim.x) {
    DevicePose ee = robot->fk_ee(&confs[i]);

    size_t x = ee.x * resolution;
    size_t y = ee.y * resolution;

    float cell = *((float*)map_data->get(x, y));

    results[i] = cell >= 1.f;
  }
}

void CollisionChecker::check(const std::vector<Configuration>& configurations) {
  LOG_DEBUG(log_) << "Checking " << configurations.size()
                  << " configurations for collisions in blocks of "
                  << conf_buf_len_;

  DeviceConfiguration* host_buf = (DeviceConfiguration*)conf_buf_;

  size_t num_iterations = (configurations.size() - 1) / conf_buf_len_ + 1;
  for (size_t i = 0; i < num_iterations; ++i) {
    size_t block_remaining =
        min(conf_buf_len_, configurations.size() - i * conf_buf_len_);

    for (size_t j = 0; j < block_remaining; ++j) {
      const Configuration& conf = configurations[i * conf_buf_len_ + j];
      host_buf[j] =
          DeviceConfiguration(conf.joints[0], conf.joints[1], conf.joints[2]);
    }

    CHECK_CUDA(cudaMemcpy(dev_conf_buf_, conf_buf_,
                          conf_buf_len_ * sizeof(DeviceConfiguration),
                          cudaMemcpyHostToDevice),
               "Could not move configurations to device");
    check_collisions<<<1, 32>>>(map_->device_map(), robot_->device_robot(),
                                (DeviceConfiguration*)dev_conf_buf_,
                                (bool*)dev_result_buf_, block_remaining);
    CHECK_CUDA(cudaMemcpy(result_buf_, dev_result_buf_,
                          conf_buf_len_ * sizeof(bool), cudaMemcpyDeviceToHost),
               "Could not memcpy collision check result to host");

    for (size_t j = 0; j < block_remaining; ++j) {
      if (((bool*)result_buf_)[j]) {
        LOG_DEBUG(log_) << "Configuration " << i * conf_buf_len_ + j << ": X";
      } else {
        LOG_DEBUG(log_) << "Configuration " << i * conf_buf_len_ + j << ":   X";
      }
    }
  }
}

}  // namespace gpu_planning
