#include "array.hpp"
#include "obstacle_manager.hpp"

namespace gpu_planning {

ObstacleManager::ObstacleManager() : id_cnt_{0}, id_to_name_{} {}

void ObstacleManager::add_static_circle(const Position<float>& position,
                                        float radius, const std::string& name) {
  uint8_t new_id = ++id_cnt_;

  id_to_name_[new_id] = name;
  circles_to_add_.emplace_back(Circle(radius), position, new_id);
}

void ObstacleManager::add_static_rectangle(const Position<float>& position,
                                           float width, float height,
                                           const std::string& name) {
  uint8_t new_id = ++id_cnt_;

  id_to_name_[new_id] = name;
  rectangles_to_add_.emplace_back(Rectangle(width, height), position, new_id);
}

__global__ void device_map_insert_circle(
    Map* map, const Array<Obstacle<Circle>>* circle_buf) {
  Array2d<Cell>* map_data = map->data();
  Box<size_t> map_area = map_data->area();

  for (size_t i = threadIdx.z; i < circle_buf->size(); i += blockDim.z) {
    const Obstacle<Circle>& circle = (*circle_buf)[i];

    Translation<float> radius_trans(circle.shape.radius, circle.shape.radius);

    Position<size_t> left_bottom =
        map_area.clamp(map->to_index(circle.position - radius_trans));
    Position<size_t> right_top =
        map_area.clamp(map->to_index(circle.position + radius_trans));

    for (size_t y = left_bottom.y + threadIdx.y; y < right_top.y;
         y += blockDim.y) {
      for (size_t x = left_bottom.x + threadIdx.x; x < right_top.x;
           x += blockDim.x) {
        const Translation<float> delta =
            circle.position - map->from_index(Position<size_t>(x, y));

        if (delta.x * delta.x + delta.y * delta.y <
            circle.shape.radius * circle.shape.radius) {
          map_data->at(x, y) = Cell(1.0, circle.id);
        }
      }
    }
  }
}

__global__ void device_map_insert_rectangle(
    Map* map, const Array<Obstacle<Rectangle>>* rectangle_buf) {
  Array2d<Cell>* map_data = map->data();
  Box<size_t> map_area = map_data->area();

  for (size_t i = threadIdx.z; i < rectangle_buf->size(); i += blockDim.z) {
    const Obstacle<Rectangle>& rect = (*rectangle_buf)[i];

    Translation<float> corner_dist(rect.shape.width / 2, rect.shape.height / 2);

    Position<size_t> left_bottom =
        map_area.clamp(map->to_index(rect.position - corner_dist));
    Position<size_t> right_top =
        map_area.clamp(map->to_index(rect.position + corner_dist));

    for (size_t y = left_bottom.y + threadIdx.y; y < right_top.y;
         y += blockDim.y) {
      for (size_t x = left_bottom.x + threadIdx.x; x < right_top.x;
           x += blockDim.x) {
        map_data->at(x, y) = Cell(1.0, rect.id);
      }
    }
  }
}

void ObstacleManager::insert_in_map(DeviceMap& map) {
  if (circles_to_add_.size() > 0) {
    DeviceArray<Obstacle<Circle>> circle_buf =
        DeviceArray<Obstacle<Circle>>::from(circles_to_add_);

    device_map_insert_circle<<<1, dim3(32, 32)>>>(map.device_map(),
                                                  circle_buf.device_handle());
  }

  if (rectangles_to_add_.size() > 0) {
    DeviceArray<Obstacle<Rectangle>> rect_buf =
        DeviceArray<Obstacle<Rectangle>>::from(rectangles_to_add_);

    device_map_insert_rectangle<<<1, dim3(32, 32)>>>(map.device_map(),
                                                     rect_buf.device_handle());
  }
}

const std::string& ObstacleManager::get_obstacle_name(uint8_t id) {
  return id_to_name_[id];
}

}  // namespace gpu_planning
