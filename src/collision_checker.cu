#include <cstring>

#include "collision_checker.hpp"
#include "cuda_util.hpp"
#include "device_robot.cuh"

namespace gpu_planning {

CollisionChecker::CollisionChecker()
    : conf_buf_len_{0},
      conf_buf_{nullptr},
      dev_conf_buf_{nullptr},
      map_{nullptr},
      robot_{nullptr},
      log_{nullptr} {}

CollisionChecker::CollisionChecker(Map* map, Robot* robot, Logger* log)
    : conf_buf_len_{100},
      conf_buf_{new DeviceConfiguration[conf_buf_len_]},
      dev_conf_buf_{nullptr},
      map_{map},
      robot_{robot},
      log_{log} {
  CHECK_CUDA(
      cudaMalloc(&dev_conf_buf_, conf_buf_len_ * sizeof(DeviceConfiguration)),
      "Could not allocate configuration buffer for collision checker");
}

CollisionChecker::~CollisionChecker() {
  if (conf_buf_ != nullptr) {
    delete[](DeviceConfiguration*) conf_buf_;
  }
  if (dev_conf_buf_ != nullptr) {
    CHECK_CUDA(cudaFree(dev_conf_buf_),
               "Could not free configuration buffer or collision checker");
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
  }
}

}  // namespace gpu_planning
