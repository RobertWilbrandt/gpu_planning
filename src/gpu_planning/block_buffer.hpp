#pragma once

#include "array.hpp"
#include "cuda_runtime.h"
#include "cuda_util.hpp"

namespace gpu_planning {

template <typename T>
class WriteBlockBuffer {
 public:
  WriteBlockBuffer();
  WriteBlockBuffer(size_t block_size);

  void set_data_source(const T* data_source, size_t size);

  bool done();
  Array<T>* next_device_handle();

  size_t block_size() const;
  size_t current_index_offset() const;

 private:
  const T* data_source_;
  size_t data_remaining_;
  size_t index_offset_;

  Array<T> device_buffer_;
  Array<T>* device_handle_;
};

template <typename T>
WriteBlockBuffer<T>::WriteBlockBuffer()
    : data_source_{nullptr},
      data_remaining_{0},
      index_offset_{0},
      device_buffer_{},
      device_handle_{nullptr} {}

template <typename T>
WriteBlockBuffer<T>::WriteBlockBuffer(size_t block_size)
    : data_source_{nullptr},
      data_remaining_{0},
      index_offset_{0},
      device_buffer_{},
      device_handle_{nullptr} {
  T* buffer;
  CHECK_CUDA(cudaMalloc(&buffer, block_size * sizeof(T)),
             "Could not allocate block buffer memory");
  device_buffer_ = Array<T>(buffer, block_size);

  CHECK_CUDA(cudaMalloc(&device_handle_, sizeof(Array<T>)),
             "Could not allocate block buffer handle");
}

template <typename T>
void WriteBlockBuffer<T>::set_data_source(const T* data_source, size_t size) {
  data_source_ = data_source;
  data_remaining_ = size;
  index_offset_ = 0;
}

template <typename T>
size_t WriteBlockBuffer<T>::block_size() const {
  return device_buffer_.size();
}

template <typename T>
size_t WriteBlockBuffer<T>::current_index_offset() const {
  return index_offset_;
}

template <typename T>
bool WriteBlockBuffer<T>::done() {
  if (data_remaining_ == 0) {
    return true;
  } else {
    return false;
  }
}

template <typename T>
Array<T>* WriteBlockBuffer<T>::next_device_handle() {
  const size_t remaining_block = min(data_remaining_, device_buffer_.size());

  // Update block handle
  const Array<T> updated_device_handle(device_buffer_.data(), remaining_block);
  CHECK_CUDA(cudaMemcpy(device_handle_, &updated_device_handle,
                        sizeof(Array<T>), cudaMemcpyHostToDevice),
             "Could not update block buffer handle with memcpy");

  // Move next block of data over
  CHECK_CUDA(cudaMemcpy(device_buffer_.data(), data_source_,
                        remaining_block * sizeof(T), cudaMemcpyHostToDevice),
             "Could not move next block of block buffer to device with memcpy");

  data_source_ += remaining_block;
  data_remaining_ -= remaining_block;
  index_offset_ += remaining_block;

  return device_handle_;
}

}  // namespace gpu_planning
