#pragma once

#include "cuda_runtime.h"
#include "cuda_util.hpp"
#include "device_handle.hpp"
#include "geometry.hpp"
#include "thread_block.hpp"

namespace gpu_planning {

template <typename T>
class Array2d {
 public:
  __host__ __device__ Array2d();
  __host__ __device__ Array2d(T* data, size_t width, size_t height);
  __host__ __device__ Array2d(T* data, size_t width, size_t height,
                              size_t pitch);

  __host__ __device__ T* data() const;
  __host__ __device__ size_t width() const;
  __host__ __device__ size_t height() const;
  __host__ __device__ size_t pitch() const;

  __host__ __device__ T& at(size_t x, size_t y);
  __host__ __device__ T& at(const Position<size_t>& position);

  __host__ __device__ const T& at(size_t x, size_t y) const;
  __host__ __device__ const T& at(const Position<size_t>& position) const;

  __host__ __device__ Box<size_t> area() const;

  /** Reduce this array in parallel
   *
   * \tparam Reducer Implementation of the specific reduction function
   * \pre    `Reducer` must provide a static function \code{.hpp}
   *           void reduce(T& v1, const T& v2)
   *         \endcode
   * \param  thread_block Thread block layout
   * \pre    \code{.cu}
   *           thread_block.dim_x() == width()
   *         \endcode
   * \pre    \code{.cu}
   *           thread_block.dim_y() == height()
   *         \endcode
   * \pre    `thread_block.dim_x()` and `thread_block.dim_y()`
   *
   * \note This will modify the array contents in the process
   * \note This will synchronize using `thread_block` multiple times, so be sure
   *       to call this on all threads of the full thread block
   */
  template <typename Reducer>
  __host__ __device__ T reduce(const ThreadBlock2d& thread_block);

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

  DeviceArray2d(const DeviceArray2d& other) = delete;
  DeviceArray2d& operator=(const DeviceArray2d& other) = delete;

  DeviceArray2d(DeviceArray2d&& other) noexcept;
  DeviceArray2d& operator=(DeviceArray2d&& other) noexcept;

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
  DeviceHandle<Array2d<T>> device_handle_;
};

template <typename T>
__host__ __device__ Array2d<T>::Array2d()
    : data_{nullptr}, width_{0}, height_{0}, pitch_{0} {}

template <typename T>
__host__ __device__ Array2d<T>::Array2d(T* data, size_t width, size_t height)
    : Array2d<T>{data, width, height, width * sizeof(T)} {}

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
__host__ __device__ Box<size_t> Array2d<T>::area() const {
  return Box<size_t>(0, width_ - 1, 0, height_ - 1);
}

template <typename T>
template <typename Reducer>
__host__ __device__ T Array2d<T>::reduce(const ThreadBlock2d& thread_block) {
  size_t cur_width = width();
  size_t cur_height = height();

  while ((cur_width > 1) || (cur_height > 1)) {
    const size_t x_fact = cur_width > 1 ? 2 : 1;
    const size_t y_fact = cur_height > 1 ? 2 : 1;

    const size_t next_width = cur_width / x_fact;
    const size_t next_height = cur_height / y_fact;

    if ((thread_block.x() < next_width) && (thread_block.y() < next_height)) {
      for (size_t iy = 0; iy < y_fact; ++iy) {
        for (size_t ix = 0; ix < x_fact; ++ix) {
          Reducer::reduce(at(thread_block.x(), thread_block.y()),
                          at(thread_block.x() + ix * next_width,
                             thread_block.y() + iy * next_height));
        }
      }
    }

    thread_block.sync();

    cur_width = next_width;
    cur_height = next_height;
  }

  return at(0, 0);
}

template <typename T>
DeviceArray2d<T>::DeviceArray2d() : array_{}, device_handle_{nullptr} {}

template <typename T>
DeviceArray2d<T>::DeviceArray2d(size_t width, size_t height)
    : array_{}, device_handle_{} {
  cudaExtent data_extent =
      make_cudaExtent(width * sizeof(T), height, sizeof(T));
  cudaPitchedPtr data_pitched_ptr;
  CHECK_CUDA(cudaMalloc3D(&data_pitched_ptr, data_extent),
             "Could not allocate device 2d array data memory");
  array_ = Array2d<T>(static_cast<T*>(data_pitched_ptr.ptr), width, height,
                      data_pitched_ptr.pitch);

  device_handle_.memcpy_set(&array_);
}

template <typename T>
DeviceArray2d<T>::DeviceArray2d(DeviceArray2d&& other) noexcept
    : array_{std::move(other.array_)},
      device_handle_{std::move(other.device_handle_)} {
  other.array_ = Array2d<T>();
}

template <typename T>
DeviceArray2d<T>& DeviceArray2d<T>::operator=(DeviceArray2d&& other) noexcept {
  if (this != &other) {
    SAFE_CUDA_FREE(array_.data(), "Could not free device 2d array data memory");
    array_ = std::move(other.array_);
    device_handle_ = std::move(other.device_handle_);

    other.array_ = Array2d<T>();
  }

  return *this;
}

template <typename T>
DeviceArray2d<T>::~DeviceArray2d() {
  SAFE_CUDA_FREE(array_.data(), "Could not free device 2d array data memory");
}

template <typename T>
Array2d<T>* DeviceArray2d<T>::device_handle() const {
  return device_handle_.device_handle();
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
