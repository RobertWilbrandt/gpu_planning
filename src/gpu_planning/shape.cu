#include "shape.hpp"

namespace gpu_planning {

__host__ __device__ Circle::Circle() : radius{0.f} {}

__host__ __device__ Circle::Circle(float radius) : radius{radius} {}

__host__ __device__ Rectangle::Rectangle() : width{0.f}, height{0.f} {}

__host__ __device__ Rectangle::Rectangle(float width, float height)
    : width{width}, height{height} {}

}  // namespace gpu_planning
