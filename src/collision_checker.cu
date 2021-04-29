#include "collision_checker.hpp"
#include "cuda_util.hpp"

namespace gpu_planning {

CollisionChecker::CollisionChecker()
    : conf_buf_size_{0},
      conf_buf_{nullptr},
      dev_conf_buf_{nullptr},
      map_{nullptr},
      robot_{nullptr},
      log_{nullptr} {}

CollisionChecker::CollisionChecker(Map* map, Robot* robot, Logger* log)
    : conf_buf_size_{100},
      conf_buf_{new float[conf_buf_size_ * 3]},
      dev_conf_buf_{nullptr},
      map_{map},
      robot_{robot},
      log_{log} {
  CHECK_CUDA(cudaMalloc(&dev_conf_buf_, conf_buf_size_ * 3 * sizeof(float)),
             "Could not allocate configuration buffer for collision checker");
}

CollisionChecker::~CollisionChecker() {
  if (conf_buf_ != nullptr) {
    CHECK_CUDA(cudaFree(dev_conf_buf_),
               "Could not free configuration buffer or collision checker");
  }
}

void CollisionChecker::check(const std::vector<Configuration>& configurations) {
  LOG_DEBUG(log_) << "Checking " << configurations.size()
                  << " configurations for collisions in blocks of "
                  << conf_buf_size_;
  size_t num_iterations = (configurations.size() - 1) / conf_buf_size_ + 1;
  for (size_t i = 0; i < num_iterations; ++i) {
    size_t block_remaining =
        min(conf_buf_size_, configurations.size() - i * conf_buf_size_);

    for (size_t j = 0; j < block_remaining; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        conf_buf_[3 * j + k] = configurations[i * conf_buf_size_ + j].joints[k];
      }
    }

    CHECK_CUDA(
        cudaMemcpy(dev_conf_buf_, conf_buf_.get(),
                   conf_buf_size_ * 3 * sizeof(float), cudaMemcpyHostToDevice),
        "Could not move configurations to device");
  }
}

}  // namespace gpu_planning
