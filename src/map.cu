#include <iostream>
#include <string>

#include "map.hpp"

#define CHECK_CUDA(fun, mes)                             \
  {                                                      \
    cudaError_t err = fun;                               \
    if (err != cudaSuccess) {                            \
      throw std::runtime_error{std::string(mes) + ": " + \
                               cudaGetErrorString(err)}; \
    }                                                    \
  }

namespace gpu_planning {

Map::Map()
    : extent_{nullptr}, pitched_ptr_{nullptr}, resolution_{0}, log_{nullptr} {}

Map::Map(float width, float height, size_t resolution, Logger* log)
    : extent_{nullptr},
      pitched_ptr_{nullptr},
      resolution_{resolution},
      log_{log} {
  extent_ = new cudaExtent(make_cudaExtent(width * resolution * sizeof(float),
                                           height * resolution * sizeof(float),
                                           sizeof(float)));
  pitched_ptr_ = new cudaPitchedPtr();

  CHECK_CUDA(cudaMalloc3D(pitched_ptr_, *extent_),
             "Could not allocate map memory");
  CHECK_CUDA(cudaMemset3D(*pitched_ptr_, 0, *extent_),
             "Could not clear map memory");
}

Map::~Map() {
  if (extent_ != nullptr) {
    free(extent_);
  }
  if (pitched_ptr_ != nullptr) {
    CHECK_CUDA(cudaFree(pitched_ptr_->ptr), "Could not free map memory");
    free(pitched_ptr_);
  }
}

void Map::print_debug() {
  float buf[extent_->width * extent_->height];
  CHECK_CUDA(
      cudaMemcpy2D(buf, extent_->width, pitched_ptr_->ptr, pitched_ptr_->pitch,
                   extent_->width, extent_->height, cudaMemcpyDeviceToHost),
      "Could not copy map memory from device to host for debug printing");

  const size_t width = extent_->width / extent_->depth;
  const size_t height = extent_->height;

  for (size_t y = 0; y < height; ++y) {
    std::string line = "";
    for (size_t x = 0; x < width; ++x) {
      if (buf[y * width + x] >= 1.0) {
        line += '#';
      } else {
        line += ' ';
      }
    }
    LOG_DEBUG(log_) << line;
  }
}

}  // namespace gpu_planning
