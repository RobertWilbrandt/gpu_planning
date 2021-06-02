#pragma once

#include "array.hpp"
#include "cuda_runtime.h"
#include "cuda_util.hpp"
#include "device_handle.hpp"

namespace gpu_planning {

template <typename Data, typename Result>
class WorkBlock {
 public:
  __host__ __device__ WorkBlock();
  __host__ __device__ WorkBlock(size_t size, Data* data, Result* result,
                                size_t offset);

  __host__ __device__ size_t size() const;
  __host__ __device__ size_t offset() const;

  __host__ __device__ Data& data(size_t i);
  __host__ __device__ const Data& data(size_t i) const;
  __host__ __device__ Result& result(size_t i);
  __host__ __device__ const Result& result(size_t i) const;

 private:
  size_t size_;
  Data* data_;
  Result* result_;

  size_t offset_;
};

template <typename Data, typename Result>
class DeviceWorkHandle {
 public:
  DeviceWorkHandle();
  DeviceWorkHandle(WorkBlock<Data, Result>* device_work_block,
                   size_t block_size, Data* device_data_buf,
                   const Data* data_source, Result* device_result_buf,
                   Result* result_destination, const Stream* stream);

  ~DeviceWorkHandle();

  WorkBlock<Data, Result>* device_handle() const;

 private:
  WorkBlock<Data, Result>* device_work_block_;

  size_t block_size_;
  Data* device_data_buf_;
  const Data* data_source_;
  Result* device_result_buf_;
  Result* result_destination_;

  const Stream* stream_;
};

template <typename Data, typename Result>
class WorkBuffer {
 public:
  WorkBuffer();
  WorkBuffer(size_t block_size);

  ~WorkBuffer();

  size_t block_size() const;
  size_t work_remaining() const;

  void set_work(size_t size, const Data* data, Result* result,
                const Stream* stream = nullptr);

  bool done() const;
  DeviceWorkHandle<Data, Result> next_work_block();

  WorkBlock<Data, Result>* device_full_block();

  void sync_result();

 private:
  size_t block_size_;
  Data* data_buf_;
  Result* result_buf_;

  size_t work_offset_;
  DeviceHandle<WorkBlock<Data, Result>> block_handle_;

  size_t work_remaining_;
  const Data* data_source_;
  Result* result_destination_;

  const Stream* stream_;
};

template <typename Data, typename Result>
__host__ __device__ WorkBlock<Data, Result>::WorkBlock()
    : size_{0}, data_{nullptr}, result_{nullptr}, offset_{0} {}

template <typename Data, typename Result>
__host__ __device__ WorkBlock<Data, Result>::WorkBlock(size_t size, Data* data,
                                                       Result* result,
                                                       size_t offset)
    : size_{size}, data_{data}, result_{result}, offset_{offset} {}

template <typename Data, typename Result>
__host__ __device__ size_t WorkBlock<Data, Result>::size() const {
  return size_;
}

template <typename Data, typename Result>
__host__ __device__ size_t WorkBlock<Data, Result>::offset() const {
  return offset_;
}

template <typename Data, typename Result>
__host__ __device__ Data& WorkBlock<Data, Result>::data(size_t i) {
  return data_[i];
}

template <typename Data, typename Result>
__host__ __device__ const Data& WorkBlock<Data, Result>::data(size_t i) const {
  return data_[i];
}

template <typename Data, typename Result>
__host__ __device__ Result& WorkBlock<Data, Result>::result(size_t i) {
  return result_[i];
}

template <typename Data, typename Result>
__host__ __device__ const Result& WorkBlock<Data, Result>::result(
    size_t i) const {
  return result_[i];
}

template <typename Data, typename Result>
DeviceWorkHandle<Data, Result>::DeviceWorkHandle()
    : device_work_block_{nullptr},
      block_size_{0},
      device_data_buf_{nullptr},
      data_source_{nullptr},
      device_result_buf_{nullptr},
      result_destination_{nullptr},
      stream_{nullptr} {}

template <typename Data, typename Result>
DeviceWorkHandle<Data, Result>::DeviceWorkHandle(
    WorkBlock<Data, Result>* device_work_block, size_t block_size,
    Data* device_data_buf, const Data* data_source, Result* device_result_buf,
    Result* result_destination, const Stream* stream)
    : device_work_block_{device_work_block},
      block_size_{block_size},
      device_data_buf_{device_data_buf},
      data_source_{data_source},
      device_result_buf_{device_result_buf},
      result_destination_{result_destination},
      stream_{stream} {}

template <typename Data, typename Result>
DeviceWorkHandle<Data, Result>::~DeviceWorkHandle() {
  CHECK_CUDA(
      cudaMemcpyAsync(result_destination_, device_result_buf_,
                      block_size_ * sizeof(Result), cudaMemcpyDeviceToHost,
                      stream_->stream),
      "Could not retreive result of work buffer from device with memcpy");
}

template <typename Data, typename Result>
WorkBlock<Data, Result>* DeviceWorkHandle<Data, Result>::device_handle() const {
  CHECK_CUDA(cudaMemcpyAsync(device_data_buf_, data_source_,
                             block_size_ * sizeof(Data), cudaMemcpyHostToDevice,
                             stream_->stream),
             "Could not update work buffer data with memcpy");

  return device_work_block_;
}

template <typename Data, typename Result>
WorkBuffer<Data, Result>::WorkBuffer()
    : block_size_{0},
      data_buf_{0},
      result_buf_{0},
      work_offset_{0},
      block_handle_{nullptr},
      work_remaining_{0},
      data_source_{nullptr},
      result_destination_{0},
      stream_{nullptr} {}

template <typename Data, typename Result>
WorkBuffer<Data, Result>::WorkBuffer(size_t block_size)
    : block_size_{block_size},
      data_buf_{nullptr},
      result_buf_{nullptr},
      work_offset_{0},
      block_handle_{},
      work_remaining_{0},
      data_source_{nullptr},
      result_destination_{nullptr},
      stream_{nullptr} {
  CHECK_CUDA(cudaMalloc(&data_buf_, block_size_ * sizeof(Data)),
             "Could not allocate data buffer for work buffer");
  CHECK_CUDA(cudaMalloc(&result_buf_, block_size_ * sizeof(Result)),
             "Could not allocate result buffer for work buffer");
}

template <typename Data, typename Result>
WorkBuffer<Data, Result>::~WorkBuffer() {
  SAFE_CUDA_FREE(data_buf_, "Could not free work buffer of data buffer");
  SAFE_CUDA_FREE(result_buf_, "Could not free result buffer of data buffer");
}

template <typename Data, typename Result>
size_t WorkBuffer<Data, Result>::block_size() const {
  return block_size_;
}

template <typename Data, typename Result>
size_t WorkBuffer<Data, Result>::work_remaining() const {
  return work_remaining_;
}

template <typename Data, typename Result>
void WorkBuffer<Data, Result>::set_work(size_t size, const Data* data,
                                        Result* result, const Stream* stream) {
  work_remaining_ = size;
  work_offset_ = 0;
  data_source_ = data;
  result_destination_ = result;

  stream_ = stream;
}

template <typename Data, typename Result>
bool WorkBuffer<Data, Result>::done() const {
  return work_remaining_ == 0;
}

template <typename Data, typename Result>
DeviceWorkHandle<Data, Result> WorkBuffer<Data, Result>::next_work_block() {
  size_t cur_block_size = min(block_size_, work_remaining_);

  Stream default_stream = Stream::default_stream();
  const Stream* used_stream = stream_;
  if (used_stream == nullptr) {
    used_stream = &default_stream;
  }

  // Update device work handle
  WorkBlock<Data, Result> cur_work_block(cur_block_size, data_buf_, result_buf_,
                                         work_offset_);
  block_handle_.memcpy_set_async(&cur_work_block, *used_stream);

  // Create RAII wrapper for memcpys
  DeviceWorkHandle<Data, Result> device_work_handle(
      block_handle_.device_handle(), cur_block_size, data_buf_, data_source_,
      result_buf_, result_destination_, used_stream);

  // Bookkeeping, advance to next block
  work_remaining_ -= cur_block_size;
  work_offset_ += cur_block_size;
  data_source_ += cur_block_size;
  result_destination_ += cur_block_size;

  return device_work_handle;
}

template <typename Data, typename Result>
WorkBlock<Data, Result>* WorkBuffer<Data, Result>::device_full_block() {
  Stream default_stream = Stream::default_stream();
  const Stream* used_stream = stream_;
  if (used_stream == nullptr) {
    used_stream = &default_stream;
  }

  WorkBlock<Data, Result> full_block(block_size_, data_buf_, result_buf_, 0);
  block_handle_.memcpy_set_async(&full_block, *used_stream);

  return block_handle_.device_handle();
}

template <typename Data, typename Result>
void WorkBuffer<Data, Result>::sync_result() {
  if (stream_ != nullptr) {
    stream_->sync();
  } else {
    Stream::default_stream().sync();
  }
}

}  // namespace gpu_planning
