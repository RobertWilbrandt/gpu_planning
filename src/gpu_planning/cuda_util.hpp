#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>

#define CHECK_CUDA(fun, mes)                             \
  {                                                      \
    cudaError_t err = fun;                               \
    if (err != cudaSuccess) {                            \
      throw std::runtime_error{std::string(mes) + ": " + \
                               cudaGetErrorString(err)}; \
    }                                                    \
  }

#define SAFE_CUDA_FREE(ptr, mes)      \
  {                                   \
    if (ptr != nullptr) {             \
      CHECK_CUDA(cudaFree(ptr), mes); \
    }                                 \
  }

namespace gpu_planning {

struct Stream {
 public:
  Stream();
  Stream(cudaStream_t stream);

  static Stream default_stream();
  static Stream create();

  Stream(const Stream& other) = delete;
  Stream& operator=(const Stream& other) = delete;

  Stream(Stream&& other) noexcept;
  Stream& operator=(Stream&& other) noexcept;

  ~Stream();

  void sync() const;

  cudaStream_t stream;
};

#ifndef __CUDACC__
template <typename T>
const T& min(const T& t1, const T& t2) {
  return t1 < t2 ? t1 : t2;
}

template <typename T>
const T& max(const T& t1, const T& t2) {
  return t1 > t2 ? t1 : t2;
}
#endif

}  // namespace gpu_planning
