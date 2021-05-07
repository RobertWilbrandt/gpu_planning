#pragma once

#include "cuda_runtime_api.h"
#include "geometry.hpp"

namespace gpu_planning {

struct Circle {
  __host__ __device__ Circle();
  __host__ __device__ Circle(float radius);

  float radius;
};

struct Rectangle {
  __host__ __device__ Rectangle();
  __host__ __device__ Rectangle(float width, float height);

  float width;
  float height;
};

}  // namespace gpu_planning
