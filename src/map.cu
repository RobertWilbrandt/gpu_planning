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

size_t Map::width() const { return extent_->width / extent_->depth; }

size_t Map::height() const { return extent_->height; }

size_t Map::resolution() const { return resolution_; }

__global__ void device_consolidate_data(float* map, size_t map_pitch,
                                        size_t map_width, size_t map_height,
                                        float* dest, size_t dest_pitch,
                                        size_t x_fact, size_t y_fact) {
  const size_t sub_width = map_width / x_fact;
  const size_t sub_height = map_height / y_fact;

  for (size_t j = threadIdx.y; j < sub_height; j += blockDim.y) {
    for (size_t i = threadIdx.x; i < sub_width; i += blockDim.x) {
      float sum = 0.f;

      for (size_t cy = 0; cy < y_fact; ++cy) {
        float* map_row =
            (float*)((unsigned char*)map + (j * y_fact + cy) * map_pitch);
        for (size_t cx = 0; cx < x_fact; ++cx) {
          float entry = map_row[i * x_fact + cx];
          sum += entry;
        }
      }

      float* dest_cell =
          (float*)((unsigned char*)dest + j * dest_pitch + i * sizeof(float));
      *dest_cell = sum / (x_fact * y_fact);
    }
  }
}

void Map::get_data(float* dest, size_t max_width, size_t max_height,
                   size_t* result_width, size_t* result_height) {
  const size_t map_width = width();
  const size_t map_height = height();

  const size_t x_fact = map_width / (max_width + 1) + 1;
  const size_t y_fact = map_height / (max_height + 1) + 1;

  const size_t sub_width = map_width / x_fact;
  const size_t sub_height = map_height / y_fact;

  cudaExtent sub_extent =
      make_cudaExtent(sub_width * sizeof(float), sub_height, sizeof(float));
  cudaPitchedPtr sub_pitched_ptr;
  CHECK_CUDA(cudaMalloc3D(&sub_pitched_ptr, sub_extent),
             "Could not allocate map data consolidation buffer");

  device_consolidate_data<<<1, 32>>>(
      (float*)pitched_ptr_->ptr, pitched_ptr_->pitch, map_width, map_height,
      (float*)sub_pitched_ptr.ptr, sub_pitched_ptr.pitch, x_fact, y_fact);

  CHECK_CUDA(cudaMemcpy2D(dest, sub_width * sizeof(float), sub_pitched_ptr.ptr,
                          sub_pitched_ptr.pitch, sub_width * sizeof(float),
                          sub_height, cudaMemcpyDeviceToHost),
             "Could not copy map data consolidation buffer to host");

  CHECK_CUDA(cudaFree(sub_pitched_ptr.ptr),
             "Could not free map data consolidation buffer");

  *result_width = sub_width;
  *result_height = sub_height;
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
  device_add_obstacle_circle<<<1, 32>>>(pitched_ptr_->ptr, pitched_ptr_->pitch,
                                        resolution_, x, y, radius);
  cudaDeviceSynchronize();
}

__global__ void device_add_obstacle_rect(void* map, size_t pitch,
                                         size_t resolution, float cx, float cy,
                                         float width, float height) {
  const size_t cell_width = width * resolution;
  const size_t cell_height = height * resolution;

  for (size_t j = threadIdx.y; j < cell_height; j += blockDim.y) {
    const size_t y = j + (cy - 0.5 * height) * resolution;
    float* row_base = (float*)((unsigned char*)map + y * pitch);

    for (size_t i = threadIdx.x; i < cell_width; i += blockDim.x) {
      const size_t x = i + (cx - 0.5 * width) * resolution;
      row_base[x] = 1.0;
    }
  }
}

void Map::add_obstacle_rect(float x, float y, float width, float height) {
  device_add_obstacle_rect<<<1, 32>>>(pitched_ptr_->ptr, pitched_ptr_->pitch,
                                      resolution_, x, y, width, height);
  cudaDeviceSynchronize();
}

}  // namespace gpu_planning
