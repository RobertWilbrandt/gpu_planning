#pragma once

#include "array_2d.hpp"
#include "cuda_runtime_api.h"
#include "geometry.hpp"
#include "logging.hpp"
#include "stdint.h"

namespace gpu_planning {

struct Cell {
  __host__ __device__ Cell();
  __host__ __device__ Cell(float value, uint8_t mask);

  float value;
  uint8_t id;
};

class Map {
 public:
  __host__ __device__ Map();
  __host__ __device__ Map(Array2d<Cell>* data, size_t resolution);

  __device__ float width() const;
  __device__ float height() const;
  __device__ size_t resolution() const;
  __device__ Array2d<Cell>* data() const;

  __device__ Position<size_t> to_index(const Position<float>& position) const;
  __device__ Pose<size_t> to_index(const Pose<float>& pose) const;

  __device__ Position<float> from_index(const Position<size_t>& index) const;
  __device__ Pose<float> from_index(const Pose<size_t>& index) const;

  __device__ const Cell& get(const Position<float>& position);

 private:
  Array2d<Cell>* data_;
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

 private:
  Map* map_;
  DeviceArray2d<Cell> data_;
  size_t resolution_;

  Logger* log_;
};

}  // namespace gpu_planning
