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

}  // namespace gpu_planning
