#pragma once

#include "cuda_runtime_api.h"
#include "cuda_util.hpp"

namespace gpu_planning {

template <typename T>
class DeviceArrayHandle {
 public:
  __host__ __device__ DeviceArrayHandle();
  __host__ __device__ DeviceArrayHandle(T* data, size_t size);

  __device__ T* data() const;
  __device__ size_t size() const;

  __device__ T& operator[](size_t i);
  __device__ const T& operator[](size_t i) const;

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

  DeviceArrayHandle<T>* device_handle() const;
  size_t size() const;

  void memcpy_set(T* data);
  void memcpy_set(T* data, size_t size);
  void memcpy_get(T* dest);
  void memcpy_get(T* dest, size_t size);

 private:
  DeviceArrayHandle<T>* handle_;

  T* data_;
  size_t size_;
};

template <typename T>
__host__ __device__ DeviceArrayHandle<T>::DeviceArrayHandle()
    : data_{nullptr}, size_{0} {}

template <typename T>
__host__ __device__ DeviceArrayHandle<T>::DeviceArrayHandle(T* data,
                                                            size_t size)
    : data_{data}, size_{size} {}

template <typename T>
__device__ T* DeviceArrayHandle<T>::data() const {
  return data_;
}

template <typename T>
__device__ size_t DeviceArrayHandle<T>::size() const {
  return size_;
}

template <typename T>
__device__ T& DeviceArrayHandle<T>::operator[](size_t i) {
  return data_[i];
}

template <typename T>
__device__ const T& DeviceArrayHandle<T>::operator[](size_t i) const {
  return data_[i];
}

template <typename T>
DeviceArray<T>::DeviceArray() : handle_{nullptr}, data_{nullptr}, size_{0} {}

template <typename T>
DeviceArray<T>::DeviceArray(size_t size)
    : handle_{nullptr}, data_{nullptr}, size_{size} {
  CHECK_CUDA(cudaMalloc(&data_, size_ * sizeof(T)),
             "Could not allocate device array data memory");

  DeviceArrayHandle<T> handle(data_, size_);
  CHECK_CUDA(cudaMalloc(&handle_, sizeof(DeviceArrayHandle<T>)),
             "Could not allocate device array handle memory");
  CHECK_CUDA(cudaMemcpy(handle_, &handle, sizeof(DeviceArrayHandle<T>),
                        cudaMemcpyHostToDevice),
             "Could not memcpy device array handle to device");
}

template <typename T>
DeviceArray<T>::~DeviceArray() {
  SAFE_CUDA_FREE(data_, "Could not free device array data memory");
  SAFE_CUDA_FREE(handle_, "Could not free device array handle memory");
}

template <typename T>
DeviceArrayHandle<T>* DeviceArray<T>::device_handle() const {
  return handle_;
}

template <typename T>
size_t DeviceArray<T>::size() const {
  return size_;
}

template <typename T>
void DeviceArray<T>::memcpy_set(T* data) {
  memcpy_set(data, size_);
}

template <typename T>
void DeviceArray<T>::memcpy_set(T* data, size_t size) {
  CHECK_CUDA(cudaMemcpy(data_, data, size * sizeof(T), cudaMemcpyHostToDevice),
             "Could not memcpy to device array");
}

template <typename T>
void DeviceArray<T>::memcpy_get(T* data) {
  memcpy_get(data, size_);
}

template <typename T>
void DeviceArray<T>::memcpy_get(T* data, size_t size) {
  CHECK_CUDA(cudaMemcpy(data, data_, size * sizeof(T), cudaMemcpyDeviceToHost),
             "Could not memcpy from device array");
}

}  // namespace gpu_planning
