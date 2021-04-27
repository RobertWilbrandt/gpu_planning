#include <iostream>
#include <string>

#include "cuda_util.hpp"
#include "device_2d_array.cuh"
#include "map.hpp"

namespace gpu_planning {

Map::Map() : data_{}, resolution_{}, log_{nullptr} {}

Map::Map(float width, float height, size_t resolution, Logger* log)
    : data_{static_cast<size_t>(width * resolution),
            static_cast<size_t>(height * resolution), sizeof(float), log},
      resolution_{resolution},
      log_{log} {
  data_.clear();

  LOG_DEBUG(log_) << "Created map of size " << width << "x" << height
                  << " and resolution " << resolution_;
}

float Map::width() const { return (float)data_.width() / resolution_; }

float Map::height() const { return (float)data_.height() / resolution_; }

size_t Map::resolution() const { return resolution_; }

__global__ void device_consolidate_data(Device2dArray* map,
                                        Device2dArray* dest) {
  const size_t x_fact = map->width() / dest->width();
  const size_t y_fact = map->height() / dest->height();

  for (size_t j = threadIdx.y; j < dest->height(); j += blockDim.y) {
    for (size_t i = threadIdx.x; i < dest->width(); i += blockDim.x) {
      float sum = 0.f;

      for (size_t cy = 0; cy < y_fact; ++cy) {
        for (size_t cx = 0; cx < x_fact; ++cx) {
          sum += *((float*)map->get(i * x_fact + cx, j * y_fact + cy));
        }
      }

      *((float*)dest->get(i, j)) = sum / (x_fact * y_fact);
    }
  }
}

void Map::get_data(float* dest, size_t max_width, size_t max_height,
                   size_t* result_width, size_t* result_height) {
  const size_t map_width = data_.width();
  const size_t map_height = data_.height();

  const size_t x_fact = map_width / (max_width + 1) + 1;
  const size_t y_fact = map_height / (max_height + 1) + 1;

  const size_t sub_width = map_width / x_fact;
  const size_t sub_height = map_height / y_fact;

  Device2dArrayHandle sub(sub_width, sub_height, data_.depth(), log_);

  device_consolidate_data<<<1, 32>>>(data_.device_array(), sub.device_array());

  sub.get_data(dest);

  *result_width = sub_width;
  *result_height = sub_height;
}

__global__ void device_add_obstacle_circle(Device2dArray* map,
                                           size_t resolution, float cx,
                                           float cy, float crad) {
  size_t block_size = 2 * crad * resolution;

  size_t first_row = (cy - crad) * resolution;
  size_t first_col = (cx - crad) * resolution;

  for (size_t j = threadIdx.x; j < block_size; j += blockDim.x) {
    for (size_t i = threadIdx.y; i < block_size; i += blockDim.y) {
      size_t x = first_col + i;
      size_t y = first_row + j;

      double dx = (float)x / resolution - cx;
      double dy = (float)y / resolution - cy;

      if (dx * dx + dy * dy < crad * crad) {
        *((float*)map->get(x, y)) = 1.0;
      }
    }
  }
}

void Map::add_obstacle_circle(float x, float y, float radius) {
  device_add_obstacle_circle<<<1, 32>>>(data_.device_array(), resolution_, x, y,
                                        radius);
}

__global__ void device_add_obstacle_rect(Device2dArray* map, size_t resolution,
                                         float cx, float cy, float width,
                                         float height) {
  const size_t cell_width = width * resolution;
  const size_t cell_height = height * resolution;

  for (size_t j = threadIdx.y; j < cell_height; j += blockDim.y) {
    for (size_t i = threadIdx.x; i < cell_width; i += blockDim.x) {
      const size_t x = i + (cx - 0.5 * width) * resolution;
      const size_t y = j + (cy - 0.5 * height) * resolution;

      *((float*)map->get(x, y)) = 1.0;
    }
  }
}

void Map::add_obstacle_rect(float x, float y, float width, float height) {
  device_add_obstacle_rect<<<1, 32>>>(data_.device_array(), resolution_, x, y,
                                      width, height);
}

}  // namespace gpu_planning
