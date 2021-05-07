#include "shape.hpp"

namespace gpu_planning {

__host__ __device__ Circle::Circle() : position{}, radius{0.f}, id{0} {}

__host__ __device__ Circle::Circle(const Position<float>& position,
                                   float radius, uint8_t id)
    : position{position}, radius{radius}, id{id} {}

__host__ __device__ Rectangle::Rectangle()
    : position{}, width{0.f}, height{0.f}, id{0} {}

__host__ __device__ Rectangle::Rectangle(const Position<float>& position,
                                         float width, float height, uint8_t id)
    : position{position}, width{width}, height{height}, id{id} {}

}  // namespace gpu_planning
