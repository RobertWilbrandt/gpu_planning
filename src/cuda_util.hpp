#pragma once

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
