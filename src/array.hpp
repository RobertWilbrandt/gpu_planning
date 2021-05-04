#pragma once

#include <vector>

#include "cuda_runtime_api.h"
#include "cuda_util.hpp"

namespace gpu_planning {

template <typename T>
class Array {
 public:
  __host__ __device__ Array();
  __host__ __device__ Array(T* data, size_t size);

  __host__ __device__ T* data() const;
  __host__ __device__ size_t size() const;

  __host__ __device__ T& operator[](size_t i);
  __host__ __device__ const T& operator[](size_t i) const;

 private:
  T* data_;
  size_t size_;
};

template <typename T>
class DeviceArray {
 public:
  DeviceArray();
  DeviceArray(size_t size);

  ~DeviceArray();

  Array<T>* device_handle() const;
  size_t size() const;

  void memcpy_set(const Array<const T>& data);
  void memcpy_set(const std::vector<T>& data);
  void memcpy_set(const std::vector<T>& data, size_t start, size_t length);

  void memcpy_get(const Array<T>& dest);
  void memcpy_get(std::vector<T>& dest);
  void memcpy_get(std::vector<T>& dest, size_t start, size_t length);

 private:
  Array<T> array_;
  Array<T>* device_handle_;
};

template <typename T>
__host__ __device__ Array<T>::Array() : data_{nullptr}, size_{0} {}

template <typename T>
__host__ __device__ Array<T>::Array(T* data, size_t size)
    : data_{data}, size_{size} {}

template <typename T>
__host__ __device__ T* Array<T>::data() const {
  return data_;
}

template <typename T>
__host__ __device__ size_t Array<T>::size() const {
  return size_;
}

template <typename T>
__host__ __device__ T& Array<T>::operator[](size_t i) {
  return data_[i];
}

template <typename T>
__host__ __device__ const T& Array<T>::operator[](size_t i) const {
  return data_[i];
}

template <typename T>
DeviceArray<T>::DeviceArray() : array_{}, device_handle_{nullptr} {}

template <typename T>
DeviceArray<T>::DeviceArray(size_t size) : array_{}, device_handle_{nullptr} {
  T* data;
  CHECK_CUDA(cudaMalloc(&data, size * sizeof(T)),
             "Could not allocate device array data memory");
  array_ = Array<T>(data, size);

  CHECK_CUDA(cudaMalloc(&device_handle_, sizeof(Array<T>)),
             "Could not allocate device array handle memory");
  CHECK_CUDA(cudaMemcpy(device_handle_, &array_, sizeof(Array<T>),
                        cudaMemcpyHostToDevice),
             "Could not memcpy device array handle to device");
}

template <typename T>
DeviceArray<T>::~DeviceArray() {
  SAFE_CUDA_FREE(array_.data(), "Could not free device array data memory");
  SAFE_CUDA_FREE(device_handle_, "Could not free device array handle memory");
}

template <typename T>
Array<T>* DeviceArray<T>::device_handle() const {
  return device_handle_;
}

template <typename T>
size_t DeviceArray<T>::size() const {
  return array_.size();
}

template <typename T>
void DeviceArray<T>::memcpy_set(const Array<const T>& data) {
  CHECK_CUDA(cudaMemcpy(array_.data(), data.data(), data.size() * sizeof(T),
                        cudaMemcpyHostToDevice),
             "Could not memcpy to device array");
}

template <typename T>
void DeviceArray<T>::memcpy_set(const std::vector<T>& data) {
  const Array<const T> access(data.data(), data.size());
  memcpy_set(access);
}

template <typename T>
void DeviceArray<T>::memcpy_set(const std::vector<T>& data, size_t start,
                                size_t length) {
  const Array<const T> access(&data[start], length);
  memcpy_set(access);
}

template <typename T>
void DeviceArray<T>::memcpy_get(const Array<T>& dest) {
  CHECK_CUDA(cudaMemcpy(dest.data(), array_.data(), dest.size() * sizeof(T),
                        cudaMemcpyDeviceToHost),
             "Could not memcpy from device array");
}

template <typename T>
void DeviceArray<T>::memcpy_get(std::vector<T>& dest) {
  const Array<T> access(dest.data(), dest.size());
  memcpy_get(access);
}

template <typename T>
void DeviceArray<T>::memcpy_get(std::vector<T>& dest, size_t start,
                                size_t length) {
  const Array<T> access(&dest[start], length);
  memcpy_get(access);
}

}  // namespace gpu_planning
