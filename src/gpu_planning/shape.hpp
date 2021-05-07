#pragma once

#include "cuda_runtime_api.h"
#include "geometry.hpp"

namespace gpu_planning {

struct Circle {
  __host__ __device__ Circle();
  __host__ __device__ Circle(const Position<float>& position, float radius,
                             uint8_t id);

  Position<float> position;
  float radius;
  uint8_t id;
};

struct Rectangle {
  __host__ __device__ Rectangle();
  __host__ __device__ Rectangle(const Position<float>& position, float width,
                                float height, uint8_t id);

  Position<float> position;
  float width;
  float height;
  uint8_t id;
};

}  // namespace gpu_planning
