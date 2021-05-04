#pragma once

#include "array_2d.hpp"
#include "cuda_runtime_api.h"
#include "logging.hpp"

namespace gpu_planning {

class Map {
 public:
  __host__ __device__ Map();
  __host__ __device__ Map(Array2d<float>* data, size_t resolution);

  __device__ float width() const;
  __device__ float height() const;
  __device__ size_t resolution() const;

  __device__ Array2d<float>* data() const;

 private:
  Array2d<float>* data_;
  size_t resolution_;
};

class DeviceMap {
 public:
  DeviceMap();
  DeviceMap(float width, float height, size_t resolution, Logger* log);

  ~DeviceMap();

  float width() const;
  float height() const;
  size_t resolution() const;

  Map* device_map() const;

  void get_data(float* dest, size_t max_width, size_t max_height,
                size_t* result_width, size_t* result_height);

  void add_obstacle_circle(float x, float y, float radius);
  void add_obstacle_rect(float x, float y, float width, float height);

 private:
  Map* map_;
  DeviceArray2d<float> data_;
  size_t resolution_;

  Logger* log_;
};

}  // namespace gpu_planning
