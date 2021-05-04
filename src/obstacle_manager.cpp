#include "obstacle_manager.hpp"

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

ObstacleManager::ObstacleManager() : id_cnt_{0}, id_to_name_{} {}

void ObstacleManager::add_static_circle(const Position<float>& position,
                                        float radius, const std::string& name) {
  uint8_t new_id = ++id_cnt_;

  id_to_name_[new_id] = name;
  circles_to_add_.emplace_back(position, radius, new_id);
}

void ObstacleManager::add_static_rectangle(const Position<float>& position,
                                           float width, float height,
                                           const std::string& name) {
  uint8_t new_id = ++id_cnt_;

  id_to_name_[new_id] = name;
  rectangles_to_add_.emplace_back(position, width, height, new_id);
}

void ObstacleManager::insert_in_map(DeviceMap& map) {
  for (const Circle& circle : circles_to_add_) {
    map.add_obstacle_circle(circle.position.x, circle.position.y, circle.radius,
                            circle.id);
  }

  for (const Rectangle& rectangle : rectangles_to_add_) {
    map.add_obstacle_rect(rectangle.position.x, rectangle.position.y,
                          rectangle.width, rectangle.height, rectangle.id);
  }
}

const std::string& ObstacleManager::get_obstacle_name(uint8_t id) {
  return id_to_name_[id];
}

}  // namespace gpu_planning
