#include "cuda_util.hpp"
#include "device_2d_array.cuh"
#include "device_2d_array_handle.hpp"

namespace gpu_planning {

Device2dArrayHandle::Device2dArrayHandle()
    : device_array_{nullptr},
      width_{0},
      height_{0},
      depth_{0},
      pitch_{0},
      data_{nullptr},
      log_{nullptr} {}

Device2dArrayHandle::Device2dArrayHandle(size_t width, size_t height,
                                         size_t depth, Logger* log)
    : device_array_{nullptr},
      width_{width},
      height_{height},
      depth_{depth},
      pitch_{0},
      data_{nullptr},
      log_{log} {
  // Create 2d array on device
  cudaExtent extent = make_cudaExtent(width * depth, height, depth);
  cudaPitchedPtr pitched_ptr;
  CHECK_CUDA(cudaMalloc3D(&pitched_ptr, extent),
             "Could not allocate device 2d array");
  pitch_ = pitched_ptr.pitch;
  data_ = pitched_ptr.ptr;

  // Create device memory for management class
  CHECK_CUDA(cudaMalloc(&device_array_, sizeof(Device2dArray)),
             "Could not allocate memory for device 2d array");

  // Move array management class to device
  Device2dArray device_array(width_, height_, depth_, pitch_, data_);
  CHECK_CUDA(cudaMemcpy(device_array_, &device_array, sizeof(Device2dArray),
                        cudaMemcpyHostToDevice),
             "Could not memcpy device 2d array class to device memory");

  LOG_DEBUG(log_) << "Created device 2d array of size " << width_ << "x"
                  << height_ << " (depth: " << depth_ << ", pitch: " << pitch_
                  << ")";
}

Device2dArrayHandle::~Device2dArrayHandle() {
  if (device_array_ != nullptr) {
    CHECK_CUDA(cudaFree(device_array_), "Could not free device 2d array");
  }

  if (data_ != nullptr) {
    CHECK_CUDA(cudaFree(data_), "Could not free device 2d array data");
  }
}

size_t Device2dArrayHandle::width() const { return width_; }

size_t Device2dArrayHandle::height() const { return height_; }

size_t Device2dArrayHandle::depth() const { return depth_; }

Device2dArray* Device2dArrayHandle::device_array() const {
  return device_array_;
}

void Device2dArrayHandle::clear() {
  CHECK_CUDA(cudaMemset2D(data_, pitch_, 0, width_ * depth_, height_),
             "Could not memset devie 2d array to clear all contents");
}

void Device2dArrayHandle::get_data(void* dest) {
  CHECK_CUDA(cudaMemcpy2D(dest, width_ * depth_, data_, pitch_, width_ * depth_,
                          height_, cudaMemcpyDeviceToHost),
             "Could not memcpy 2d array to host");
}

}  // namespace gpu_planning
