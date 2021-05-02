#pragma once

#include "cuda_runtime_api.h"

namespace gpu_planning {

template <typename T>
class Array2d;

class DeviceMap {
 public:
  __host__ __device__ DeviceMap();
  __host__ __device__ DeviceMap(Array2d<float>* data, size_t resolution);

  __device__ float width() const;
  __device__ float height() const;
  __device__ size_t resolution() const;

  __device__ Array2d<float>* data() const;

 private:
  Array2d<float>* data_;
  size_t resolution_;
};

}  // namespace gpu_planning
