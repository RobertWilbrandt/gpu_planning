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
                                           height * resolution, sizeof(float)));
  pitched_ptr_ = new cudaPitchedPtr();

  CHECK_CUDA(cudaMalloc3D(pitched_ptr_, *extent_),
             "Could not allocate map memory");
  CHECK_CUDA(cudaMemset3D(*pitched_ptr_, 0, *extent_),
             "Could not clear map memory");

  LOG_DEBUG(log) << "Created 3d device array of size " << pitched_ptr_->xsize
                 << "x" << pitched_ptr_->ysize
                 << " (pitch: " << pitched_ptr_->pitch << ") for map of size "
                 << width << "x" << height << " (resolution: " << resolution
                 << ")";
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

  LOG_DEBUG(log_) << "--- " << width << "x" << height << " ---";
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
  LOG_DEBUG(log_) << "---";
}

__global__ void device_add_obstacle_circle(void* map, size_t pitch,
                                           size_t resolution, float cx,
                                           float cy, float crad) {
  size_t block_size = 2 * crad * resolution;

  size_t first_row = (cy - crad) * resolution;
  size_t first_col = (cx - crad) * resolution;

  for (size_t j = threadIdx.x; j < block_size; j += blockDim.x) {
    size_t y = first_row + j;
    float* row_base = (float*)((unsigned char*)map + y * pitch);

    for (size_t i = threadIdx.y; i < block_size; i += blockDim.y) {
      size_t x = first_col + i;

      double dx = (float)x / resolution - cx;
      double dy = (float)y / resolution - cy;

      if (dx * dx + dy * dy < crad * crad) {
        row_base[x] = 1.0;
      }
    }
  }
}

void Map::add_obstacle_circle(float x, float y, float radius) {
  device_add_obstacle_circle<<<2, 16>>>(pitched_ptr_->ptr, pitched_ptr_->pitch,
                                        resolution_, x, y, radius);
  cudaDeviceSynchronize();
}

}  // namespace gpu_planning
