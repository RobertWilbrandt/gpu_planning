#include "array.hpp"
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

__global__ void device_map_insert_circle(Map* map,
                                         const Array<Circle>* circle_buf) {
  Array2d<Cell>* map_data = map->data();

  for (size_t i = threadIdx.z; i < circle_buf->size(); i += blockDim.z) {
    const Circle& circle = (*circle_buf)[i];

    Translation<float> radius_trans(circle.radius, circle.radius);

    Position<size_t> left_bottom =
        map_data->clamp_index(map->to_index(circle.position - radius_trans));
    Position<size_t> right_top =
        map_data->clamp_index(map->to_index(circle.position + radius_trans));

    for (size_t y = left_bottom.y + threadIdx.y; y < right_top.y;
         y += blockDim.y) {
      for (size_t x = left_bottom.x + threadIdx.x; x < right_top.x;
           x += blockDim.x) {
        const Translation<float> delta =
            circle.position - map->from_index(Position<size_t>(x, y));

        if (delta.x * delta.x + delta.y * delta.y <
            circle.radius * circle.radius) {
          map_data->at(x, y) = Cell(1.0, circle.id);
        }
      }
    }
  }
}

__global__ void device_map_insert_rectangle(
    Map* map, const Array<Rectangle>* rectangle_buf) {
  Array2d<Cell>* map_data = map->data();

  for (size_t i = threadIdx.z; i < rectangle_buf->size(); i += blockDim.z) {
    const Rectangle& rect = (*rectangle_buf)[i];

    Translation<float> corner_dist(rect.width / 2, rect.height / 2);

    Position<size_t> left_bottom =
        map_data->clamp_index(map->to_index(rect.position - corner_dist));
    Position<size_t> right_top =
        map_data->clamp_index(map->to_index(rect.position + corner_dist));

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
    DeviceArray<Circle> circle_buf(circles_to_add_.size());
    Array<const Circle> circles_access(circles_to_add_.data(),
                                       circles_to_add_.size());
    circle_buf.memcpy_set(circles_access);

    device_map_insert_circle<<<1, dim3(32, 32)>>>(map.device_map(),
                                                  circle_buf.device_handle());
  }

  if (rectangles_to_add_.size() > 0) {
    DeviceArray<Rectangle> rect_buf(rectangles_to_add_.size());
    Array<const Rectangle> rect_access(rectangles_to_add_.data(),
                                       rectangles_to_add_.size());
    rect_buf.memcpy_set(rect_access);

    device_map_insert_rectangle<<<1, dim3(32, 32)>>>(map.device_map(),
                                                     rect_buf.device_handle());
  }
}

const std::string& ObstacleManager::get_obstacle_name(uint8_t id) {
  return id_to_name_[id];
}

}  // namespace gpu_planning
