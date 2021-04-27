#include "device_2d_array.cuh"

namespace gpu_planning {

Device2dArray::Device2dArray()
    : width_{0}, height_{0}, depth_{0}, pitch_{0}, data_{nullptr} {}

Device2dArray::Device2dArray(size_t width, size_t height, size_t depth,
                             size_t pitch, void* data)
    : width_{width},
      height_{height},
      depth_{depth},
      pitch_{pitch},
      data_{data} {}

__device__ size_t Device2dArray::width() const { return width_; }

__device__ size_t Device2dArray::height() const { return height_; }

__device__ size_t Device2dArray::depth() const { return depth_; }

__device__ void* Device2dArray::get(size_t x, size_t y) const {
  return (unsigned char*)data_ + y * pitch_ + x * depth_;
}

}  // namespace gpu_planning
