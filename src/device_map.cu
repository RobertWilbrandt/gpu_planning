#include "device_map.hpp"

#include "device_2d_array.hpp"

namespace gpu_planning {

DeviceMap::DeviceMap() : data_{nullptr}, resolution_{0} {}

DeviceMap::DeviceMap(Device2dArrayHandle<float>* data, size_t resolution)
    : data_{data}, resolution_{resolution} {}

__device__ float DeviceMap::width() const {
  return (float)data_->width() / resolution_;
}

__device__ float DeviceMap::height() const {
  return (float)data_->height() / resolution_;
}

__device__ size_t DeviceMap::resolution() const { return resolution_; }

__device__ Device2dArrayHandle<float>* DeviceMap::data() const { return data_; }

}  // namespace gpu_planning
