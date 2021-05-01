#pragma once

#include "cuda_runtime_api.h"

namespace gpu_planning {

template <typename T>
class Device2dArrayHandle;

class DeviceMap {
 public:
  __host__ __device__ DeviceMap();
  __host__ __device__ DeviceMap(Device2dArrayHandle<float>* data,
                                size_t resolution);

  __device__ float width() const;
  __device__ float height() const;
  __device__ size_t resolution() const;

  __device__ Device2dArrayHandle<float>* data() const;

 private:
  Device2dArrayHandle<float>* data_;
  size_t resolution_;
};

}  // namespace gpu_planning
