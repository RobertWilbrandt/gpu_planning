#include <iostream>
#include <string>

#include "map.hpp"

#define CHECK_CUDA(fun, mes)                                              \
  cudaError_t err = fun;                                                  \
  if (err != cudaSuccess) {                                               \
    throw std::runtime_error{std::string(mes) + cudaGetErrorString(err)}; \
  }

namespace gpu_planning {

DeviceArray2D::DeviceArray2D() : extent_{nullptr}, pitched_ptr_{nullptr} {}

DeviceArray2D::DeviceArray2D(size_t width, size_t height, size_t mem_size)
    : extent_{nullptr}, pitched_ptr_{nullptr} {
  extent_ = new cudaExtent(make_cudaExtent(width * mem_size, height, mem_size));
  pitched_ptr_ = new cudaPitchedPtr();

  cudaExtent* extent = (cudaExtent*)extent_;
  cudaPitchedPtr* pitched_ptr = (cudaPitchedPtr*)pitched_ptr_;

  CHECK_CUDA(cudaMalloc3D(pitched_ptr, *extent),
             "Could not allocate 2D device array: ");
}

DeviceArray2D::~DeviceArray2D() {
  if (extent_ != nullptr) {
    free(extent_);
  }
  if (pitched_ptr_ != nullptr) {
    cudaFree(((cudaPitchedPtr*)pitched_ptr_)->ptr);
    free(pitched_ptr_);
  }
}

size_t DeviceArray2D::width() const {
  cudaExtent* extent = (cudaExtent*)extent_;
  return extent->width / extent->depth;
}

size_t DeviceArray2D::height() const {
  cudaExtent* extent = (cudaExtent*)extent_;
  return extent->height;
}

void DeviceArray2D::clear() {
  cudaExtent* extent = (cudaExtent*)extent_;
  cudaPitchedPtr* pitched_ptr = (cudaPitchedPtr*)pitched_ptr_;

  CHECK_CUDA(cudaMemset3D(*pitched_ptr, 0, *extent),
             "Could not clear 2D device array: ");
}

void DeviceArray2D::read(size_t x, size_t y, size_t w, size_t h, void* dest) {
  cudaExtent* extent = (cudaExtent*)extent_;
  cudaPitchedPtr* pitched_ptr = (cudaPitchedPtr*)pitched_ptr_;

  size_t dpitch = w * extent->depth;
  void* src = (unsigned char*)pitched_ptr->ptr + x * extent->depth +
              y * pitched_ptr->pitch;
  size_t spitch = pitched_ptr->pitch;
  size_t width = w * extent->depth;
  size_t height = h;

  CHECK_CUDA(cudaMemcpy2D(dest, dpitch, src, spitch, width, height,
                          cudaMemcpyDeviceToHost),
             "Could not read array from device: ");
}

void DeviceArray2D::write(size_t x, size_t y, size_t w, size_t h, void* src) {
  cudaExtent* extent = (cudaExtent*)extent_;
  cudaPitchedPtr* pitched_ptr = (cudaPitchedPtr*)pitched_ptr_;

  void* dst = (unsigned char*)pitched_ptr->ptr + x * extent->depth +
              y * pitched_ptr->pitch;
  size_t dpitch = pitched_ptr->pitch;
  size_t spitch = w * extent->depth;
  size_t width = w * extent->depth;
  size_t height = h;

  CHECK_CUDA(cudaMemcpy2D(dst, dpitch, src, spitch, width, height,
                          cudaMemcpyHostToDevice),
             "Could not write data to device array: ");
}

Map::Map() : map_(), resolution_{0}, log_{nullptr} {}

Map::Map(size_t width, size_t height, size_t resolution, Logger* log)
    : map_{width * resolution, height * resolution, sizeof(float)},
      resolution_{resolution},
      log_{log} {
  map_.clear();
}

Map::~Map() {}

void Map::print_debug() {
  const size_t width = map_.width();
  const size_t height = map_.height();

  float buf[width * height * sizeof(float)];
  map_.read(0, 0, width, height, buf);

  for (size_t y = 0; y < height; ++y) {
    std::string line = "";
    for (size_t x = 0; x < width; ++x) {
      if (buf[y * width + x] >= 1.0) {
        line += '#';
      } else {
        line += ' ';
      }
    }
    LOG_DEBUG(log_) << line;
  }
}

}  // namespace gpu_planning
