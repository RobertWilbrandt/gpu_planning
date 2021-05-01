#pragma once

#include "cuda_runtime.h"
#include "cuda_util.hpp"

namespace gpu_planning {

template <typename T>
class Device2dArrayHandle {
 public:
  __host__ __device__ Device2dArrayHandle();
  __host__ __device__ Device2dArrayHandle(T* data, size_t width, size_t height,
                                          size_t pitch);

  __device__ T* data() const;
  __device__ size_t width() const;
  __device__ size_t height() const;
  __device__ size_t pitch() const;

  __device__ T& get(size_t x, size_t y);
  __device__ const T& get(size_t x, size_t y) const;

 private:
  T* data_;
  size_t width_;
  size_t height_;
  size_t pitch_;
};

template <typename T>
class Device2dArray {
 public:
  Device2dArray();
  Device2dArray(size_t width, size_t height);

  ~Device2dArray();

  Device2dArrayHandle<T>* device_handle() const;
  size_t width() const;
  size_t height() const;
  size_t pitch() const;

  void memset(int value);

  void memcpy_get(T* dest);

 private:
  Device2dArrayHandle<T>* handle_;

  T* data_;
  size_t width_;
  size_t height_;
  size_t pitch_;
};

template <typename T>
__host__ __device__ Device2dArrayHandle<T>::Device2dArrayHandle()
    : data_{nullptr}, width_{0}, height_{0}, pitch_{0} {}

template <typename T>
__host__ __device__ Device2dArrayHandle<T>::Device2dArrayHandle(T* data,
                                                                size_t width,
                                                                size_t height,
                                                                size_t pitch)
    : data_{data}, width_{width}, height_{height}, pitch_{pitch} {}

template <typename T>
__device__ T* Device2dArrayHandle<T>::data() const {
  return data_;
}

template <typename T>
__device__ size_t Device2dArrayHandle<T>::width() const {
  return width_;
}

template <typename T>
__device__ size_t Device2dArrayHandle<T>::height() const {
  return height_;
}

template <typename T>
__device__ size_t Device2dArrayHandle<T>::pitch() const {
  return pitch_;
}

template <typename T>
__device__ T& Device2dArrayHandle<T>::get(size_t x, size_t y) {
  unsigned char* row = reinterpret_cast<unsigned char*>(data_) + y * pitch_;
  return reinterpret_cast<T*>(row)[x];
}

template <typename T>
__device__ const T& Device2dArrayHandle<T>::get(size_t x, size_t y) const {
  unsigned char* row = reinterpret_cast<unsigned char*>(data_) + y * pitch_;
  return reinterpret_cast<T*>(row)[x];
}

template <typename T>
Device2dArray<T>::Device2dArray()
    : handle_{nullptr}, data_{nullptr}, width_{0}, height_{0}, pitch_{0} {}

template <typename T>
Device2dArray<T>::Device2dArray(size_t width, size_t height)
    : handle_{nullptr},
      data_{nullptr},
      width_{width},
      height_{height},
      pitch_{0} {
  cudaExtent data_extent =
      make_cudaExtent(width_ * sizeof(T), height, sizeof(T));
  cudaPitchedPtr data_pitched_ptr;
  CHECK_CUDA(cudaMalloc3D(&data_pitched_ptr, data_extent),
             "Could not allocate device 2d array data memory");
  data_ = static_cast<T*>(data_pitched_ptr.ptr);
  pitch_ = data_pitched_ptr.pitch;

  CHECK_CUDA(cudaMalloc(&handle_, sizeof(Device2dArrayHandle<T>)),
             "Could not allocate device 2d array handle memory");
  Device2dArrayHandle<T> host_handle(data_, width_, height_, pitch_);
  CHECK_CUDA(cudaMemcpy(handle_, &host_handle, sizeof(Device2dArrayHandle<T>),
                        cudaMemcpyHostToDevice),
             "Could not memcpy device 2d array handle memory to device");
}

template <typename T>
Device2dArray<T>::~Device2dArray() {
  SAFE_CUDA_FREE(data_, "Could not free device 2d array data memory");
  SAFE_CUDA_FREE(handle_, "Could not free device 2d array handle memory");
}

template <typename T>
Device2dArrayHandle<T>* Device2dArray<T>::device_handle() const {
  return handle_;
}

template <typename T>
size_t Device2dArray<T>::width() const {
  return width_;
}

template <typename T>
size_t Device2dArray<T>::height() const {
  return height_;
}

template <typename T>
size_t Device2dArray<T>::pitch() const {
  return pitch_;
}

template <typename T>
void Device2dArray<T>::memset(int value) {
  CHECK_CUDA(cudaMemset2D(data_, pitch_, value, width_ * sizeof(T), height_),
             "Could not memset data on device 2d array");
}

template <typename T>
void Device2dArray<T>::memcpy_get(T* dest) {
  CHECK_CUDA(cudaMemcpy2D(dest, width_ * sizeof(T), data_, pitch_,
                          width_ * sizeof(T), height_, cudaMemcpyDeviceToHost),
             "Could not memcpy data from device 2d array to host");
}

}  // namespace gpu_planning
