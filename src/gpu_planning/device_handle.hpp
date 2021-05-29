#pragma once

#include "cuda_runtime_api.h"
#include "cuda_util.hpp"

namespace gpu_planning {

template <typename T>
class DeviceHandle {
 public:
  DeviceHandle();
  DeviceHandle(nullptr_t do_not_init);

  DeviceHandle(const DeviceHandle& other) = delete;
  DeviceHandle& operator=(const DeviceHandle& other) = delete;

  DeviceHandle(DeviceHandle&& other) noexcept;
  DeviceHandle& operator=(DeviceHandle&& other) noexcept;

  ~DeviceHandle();

  T* device_handle() const;

  void memcpy_set(const T* value);
  void memcpy_get(T* dest);

  void memcpy_set_async(const T* value, cudaStream_t stream = 0);
  void memcpy_get_async(T* dest, cudaStream_t stream = 0);

 private:
  T* device_handle_;
};

template <typename T>
DeviceHandle<T>::DeviceHandle() : device_handle_{nullptr} {
  CHECK_CUDA(cudaMalloc(&device_handle_, sizeof(T)),
             "Could not allocate memory for device handle");
}

template <typename T>
DeviceHandle<T>::DeviceHandle(nullptr_t do_not_init)
    : device_handle_{nullptr} {}

template <typename T>
DeviceHandle<T>::DeviceHandle(DeviceHandle&& other) noexcept
    : device_handle_{other.device_handle_} {
  other.device_handle_ = nullptr;
}

template <typename T>
DeviceHandle<T>& DeviceHandle<T>::operator=(DeviceHandle&& other) noexcept {
  if (this != &other) {
    SAFE_CUDA_FREE(device_handle_, "Could not free memory of device handle");

    device_handle_ = other.device_handle_;
    other.device_handle_ = nullptr;
  }

  return *this;
}

template <typename T>
DeviceHandle<T>::~DeviceHandle() {
  SAFE_CUDA_FREE(device_handle_, "Could not free memory of device handle");
}

template <typename T>
T* DeviceHandle<T>::device_handle() const {
  return device_handle_;
}

template <typename T>
void DeviceHandle<T>::memcpy_set(const T* value) {
  CHECK_CUDA(
      cudaMemcpy(device_handle_, value, sizeof(T), cudaMemcpyHostToDevice),
      "Could not set value of device handle using memcpy");
}

template <typename T>
void DeviceHandle<T>::memcpy_get(T* dest) {
  CHECK_CUDA(
      cudaMemcpy(dest, device_handle_, sizeof(T), cudaMemcpyDeviceToHost),
      "Could not get value from device handle using memcpy");
}

template <typename T>
void DeviceHandle<T>::memcpy_set_async(const T* value, cudaStream_t stream) {
  CHECK_CUDA(cudaMemcpyAsync(device_handle_, value, sizeof(T),
                             cudaMemcpyHostToDevice, stream),
             "Could not set value of device handle using memcpy");
}

template <typename T>
void DeviceHandle<T>::memcpy_get_async(T* dest, cudaStream_t stream) {
  CHECK_CUDA(cudaMemcpyAsync(dest, device_handle_, sizeof(T),
                             cudaMemcpyDeviceToHost, stream),
             "Could not set value of device handle using memcpy");
}

}  // namespace gpu_planning
