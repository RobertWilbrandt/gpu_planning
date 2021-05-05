#pragma once

#include "cuda_runtime.h"
#include "cuda_util.hpp"
#include "geometry.hpp"

namespace gpu_planning {

template <typename T>
class Array2d {
 public:
  __host__ __device__ Array2d();
  __host__ __device__ Array2d(T* data, size_t width, size_t height,
                              size_t pitch);

  __host__ __device__ T* data() const;
  __host__ __device__ size_t width() const;
  __host__ __device__ size_t height() const;
  __host__ __device__ size_t pitch() const;

  __device__ T& at(size_t x, size_t y);
  __device__ T& at(const Position<size_t>& position);

  __device__ const T& at(size_t x, size_t y) const;
  __device__ const T& at(const Position<size_t>& position) const;

  __device__ Position<size_t> clamp_index(
      const Position<size_t>& position) const;

 private:
  T* data_;
  size_t width_;
  size_t height_;
  size_t pitch_;
};

template <typename T>
class DeviceArray2d {
 public:
  DeviceArray2d();
  DeviceArray2d(size_t width, size_t height);

  ~DeviceArray2d();

  Array2d<T>* device_handle() const;

  size_t width() const;
  size_t height() const;
  size_t pitch() const;

  void memset(int value);
  void memcpy_set(const Array2d<const T>& data);
  void memcpy_get(const Array2d<T>& dest);

 private:
  Array2d<T> array_;
  Array2d<T>* device_handle_;
};

template <typename T>
__host__ __device__ Array2d<T>::Array2d()
    : data_{nullptr}, width_{0}, height_{0}, pitch_{0} {}

template <typename T>
__host__ __device__ Array2d<T>::Array2d(T* data, size_t width, size_t height,
                                        size_t pitch)
    : data_{data}, width_{width}, height_{height}, pitch_{pitch} {}

template <typename T>
__host__ __device__ T* Array2d<T>::data() const {
  return data_;
}

template <typename T>
__host__ __device__ size_t Array2d<T>::width() const {
  return width_;
}

template <typename T>
__host__ __device__ size_t Array2d<T>::height() const {
  return height_;
}

template <typename T>
__host__ __device__ size_t Array2d<T>::pitch() const {
  return pitch_;
}

template <typename T>
__device__ T& Array2d<T>::at(size_t x, size_t y) {
  return at(Position<size_t>(x, y));
}

template <typename T>
__device__ T& Array2d<T>::at(const Position<size_t>& position) {
  unsigned char* row =
      reinterpret_cast<unsigned char*>(data_) + position.y * pitch_;
  return reinterpret_cast<T*>(row)[position.x];
}

template <typename T>
__device__ const T& Array2d<T>::at(size_t x, size_t y) const {
  return at(Position<size_t>(x, y));
}

template <typename T>
__device__ const T& Array2d<T>::at(const Position<size_t>& position) const {
  unsigned char* row =
      reinterpret_cast<unsigned char*>(data_) + position.y * pitch_;
  return reinterpret_cast<T*>(row)[position.x];
}

template <typename T>
__device__ Position<size_t> Array2d<T>::clamp_index(
    const Position<size_t>& position) const {
  return position.clamp(Position<size_t>(0, 0),
                        Position<size_t>(width_, height_));
}

template <typename T>
DeviceArray2d<T>::DeviceArray2d() : array_{}, device_handle_{nullptr} {}

template <typename T>
DeviceArray2d<T>::DeviceArray2d(size_t width, size_t height)
    : array_{}, device_handle_{nullptr} {
  cudaExtent data_extent =
      make_cudaExtent(width * sizeof(T), height, sizeof(T));
  cudaPitchedPtr data_pitched_ptr;
  CHECK_CUDA(cudaMalloc3D(&data_pitched_ptr, data_extent),
             "Could not allocate device 2d array data memory");
  array_ = Array2d<T>(static_cast<T*>(data_pitched_ptr.ptr), width, height,
                      data_pitched_ptr.pitch);

  CHECK_CUDA(cudaMalloc(&device_handle_, sizeof(Array2d<T>)),
             "Could not allocate device 2d array handle memory");
  CHECK_CUDA(cudaMemcpy(device_handle_, &array_, sizeof(Array2d<T>),
                        cudaMemcpyHostToDevice),
             "Could not memcpy device 2d array handle to device");
}

template <typename T>
DeviceArray2d<T>::~DeviceArray2d() {
  SAFE_CUDA_FREE(array_.data(), "Could not free device 2d array data memory");
  SAFE_CUDA_FREE(device_handle_,
                 "Could not free device 2d array handle memory");
}

template <typename T>
Array2d<T>* DeviceArray2d<T>::device_handle() const {
  return device_handle_;
}

template <typename T>
size_t DeviceArray2d<T>::width() const {
  return array_.width();
}

template <typename T>
size_t DeviceArray2d<T>::height() const {
  return array_.height();
}

template <typename T>
size_t DeviceArray2d<T>::pitch() const {
  return array_.pitch();
}

template <typename T>
void DeviceArray2d<T>::memset(int value) {
  CHECK_CUDA(cudaMemset2D(array_.data(), array_.pitch(), value,
                          array_.width() * sizeof(T), array_.height()),
             "Could not memset data on device 2d array");
}

template <typename T>
void DeviceArray2d<T>::memcpy_set(const Array2d<const T>& data) {
  CHECK_CUDA(cudaMemcpy2D(array_.data(), array_.pitch(), data.data(),
                          data.pitch(), data.width() * sizeof(T), data.height(),
                          cudaMemcpyHostToDevice),
             "Could not memcpy data to 2d device array");
}

template <typename T>
void DeviceArray2d<T>::memcpy_get(const Array2d<T>& dest) {
  CHECK_CUDA(cudaMemcpy2D(dest.data(), dest.pitch(), array_.data(),
                          array_.pitch(), dest.width() * sizeof(T),
                          dest.height(), cudaMemcpyDeviceToHost),
             "Could not memcpy data from 2d device array");
}

}  // namespace gpu_planning
