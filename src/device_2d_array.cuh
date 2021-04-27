#pragma once

namespace gpu_planning {

  class Device2dArray{
    public:
      __device__ __host__ Device2dArray();
      __device__ __host__ Device2dArray(size_t width, size_t height, size_t depth, size_t pitch, void* data);
    private:
      size_t width_;
      size_t height_;
      size_t depth_;
      size_t pitch_;

      void* data_;
  };

}
