#pragma once

#include "cuda_runtime_api.h"
#include "geometry.hpp"

namespace gpu_planning {

struct Circle {
  __host__ __device__ Circle();
  __host__ __device__ Circle(float radius);

  __host__ __device__ Box<float> bounding_box(const Pose<float>& pose) const;
  __host__ __device__ bool is_inside(const Position<float>& pos) const;

  float radius;
};

struct Rectangle {
  __host__ __device__ Rectangle();
  __host__ __device__ Rectangle(float width, float height);

  __host__ __device__ Box<float> bounding_box(const Pose<float>& pose) const;
  __host__ __device__ bool is_inside(const Position<float>& pos) const;

  float width;
  float height;
};

}  // namespace gpu_planning
