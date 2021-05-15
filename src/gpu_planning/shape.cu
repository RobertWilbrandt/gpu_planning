#include "math.h"
#include "shape.hpp"

namespace gpu_planning {

__host__ __device__ Circle::Circle() : radius{0.f} {}

__host__ __device__ Circle::Circle(float radius) : radius{radius} {}

__host__ __device__ Box<float> Circle::bounding_box() const {
  return Box<float>(-radius, radius, -radius, radius);
}

__host__ __device__ Box<float> Circle::bounding_box(float orientation) const {
  return bounding_box();
}

__host__ __device__ Circle Circle::max_extent() const { return *this; }

__host__ __device__ bool Circle::is_inside(const Position<float>& pos) const {
  const Translation<float> dist = pos - Position<float>();

  return dist.x * dist.x + dist.y * dist.y <= (radius * radius);
}

__host__ __device__ Rectangle::Rectangle() : width{0.f}, height{0.f} {}

__host__ __device__ Rectangle::Rectangle(float width, float height)
    : width{width}, height{height} {}

__host__ __device__ Box<float> Rectangle::bounding_box(
    float orientation) const {
  const Translation<float> mid_to_top_right =
      Translation<float>(width / 2, height / 2).rotate(orientation);
  const Translation<float> mid_to_bot_right =
      Translation<float>(width / 2, -height / 2).rotate(orientation);

  const float max_x = max(fabs(mid_to_top_right.x), fabs(mid_to_bot_right.x));
  const float max_y = max(fabs(mid_to_top_right.y), fabs(mid_to_bot_right.y));

  return Box<float>(-max_x, max_x, -max_y, max_y);
}

__host__ __device__ Circle Rectangle::max_extent() const {
  return Circle(sqrtf(width * width / 4 + height * height / 4));
}

__host__ __device__ bool Rectangle::is_inside(
    const Position<float>& pos) const {
  return Box<float>(-width / 2, width / 2, -height / 2, height / 2)
      .is_inside(pos);
}

}  // namespace gpu_planning
