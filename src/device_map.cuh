#pragma once

namespace gpu_planning {

class Device2dArray;

class DeviceMap {
 public:
  __host__ __device__ DeviceMap();
  __host__ __device__ DeviceMap(Device2dArray* data, size_t resolution);

  __device__ float width() const;
  __device__ float height() const;
  __device__ size_t resolution() const;

  __device__ Device2dArray* data() const;

 private:
  Device2dArray* data_;
  size_t resolution_;
};

}  // namespace gpu_planning
