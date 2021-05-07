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

template <typename Shape>
__global__ void device_map_insert_shape(
    Map* map, const Array<Obstacle<Shape>>* shape_buf) {
  Array2d<Cell>& map_data = *map->data();
  const Box<size_t> map_area = map_data.area();

  for (size_t i = threadIdx.z; i < shape_buf->size(); i += blockDim.z) {
    const Obstacle<Shape>& obst = (*shape_buf)[i];

    const Box<float> shape_bb =
        obst.shape.bounding_box(Pose<float>(obst.position, 0.f));
    const Box<size_t> mask(map_area.clamp(map->to_index(shape_bb.lower_left)),
                           map_area.clamp(map->to_index(shape_bb.upper_right)));

    for (size_t y = mask.lower_left.y + threadIdx.y; y < mask.upper_right.y;
         y += blockDim.y) {
      for (size_t x = mask.lower_left.x + threadIdx.x; x < mask.upper_right.x;
           x += blockDim.x) {
        const Position<size_t> pos(x, y);
        const Position<float> map_pos = map->from_index(pos);
        const Position<float> norm_pos =
            map_pos + (Position<float>() - obst.position);

        if (obst.shape.is_inside(norm_pos)) {
          map_data.at(x, y) = Cell(1.0, obst.id);
        }
      }
    }
  }
}

void ObstacleManager::insert_in_map(DeviceMap& map) {
  if (circles_to_add_.size() > 0) {
    DeviceArray<Obstacle<Circle>> circle_buf =
        DeviceArray<Obstacle<Circle>>::from(circles_to_add_);

    device_map_insert_shape<Circle>
        <<<1, dim3(32, 32)>>>(map.device_map(), circle_buf.device_handle());
  }

  if (rectangles_to_add_.size() > 0) {
    DeviceArray<Obstacle<Rectangle>> rect_buf =
        DeviceArray<Obstacle<Rectangle>>::from(rectangles_to_add_);

    device_map_insert_shape<Rectangle>
        <<<1, dim3(32, 32)>>>(map.device_map(), rect_buf.device_handle());
  }
}

const std::string& ObstacleManager::get_obstacle_name(uint8_t id) {
  return id_to_name_[id];
}

}  // namespace gpu_planning
