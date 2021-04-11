#include "map.hpp"

namespace gpu_planning {

Map::Map() : map_{nullptr} {}

Map::Map(size_t width, size_t height) : map_{new cudaPitchedPtr()} {
  cudaExtent extent = make_cudaExtent(width, height, sizeof(float));
  if (cudaMalloc3D(map_.get(), extent) != cudaSuccess) {
    throw std::runtime_error{"Could not allocate map memory"};
  }

  if (cudaMemset3D(*map_.get(), 0, extent) != cudaSuccess) {
    throw std::runtime_error{"Could not clear map memory"};
  }
}

Map::~Map() {
  if (map_ != nullptr) {
    cudaFree(map_->ptr);
  }
}

}  // namespace gpu_planning
