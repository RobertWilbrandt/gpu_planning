#include "array.hpp"
#include "obstacle_manager.hpp"

namespace gpu_planning {

ObstacleManager::ObstacleManager() : id_cnt_{0}, id_to_name_{} {}

void ObstacleManager::add_static_circle(const Pose<float>& pose, float radius,
                                        const std::string& name) {
  uint8_t new_id = ++id_cnt_;

  id_to_name_[new_id] = name;
  circles_to_add_.emplace_back(Circle(radius), pose, new_id);
}

void ObstacleManager::add_static_rectangle(const Pose<float>& pose, float width,
                                           float height,
                                           const std::string& name) {
  uint8_t new_id = ++id_cnt_;

  id_to_name_[new_id] = name;
  rectangles_to_add_.emplace_back(Rectangle(width, height), pose, new_id);
}

template <typename Shape>
__global__ void device_map_insert_shapes(
    Map* map, const Array<Obstacle<Shape>>* shape_buf) {
  for (size_t i = threadIdx.z; i < shape_buf->size(); i += blockDim.z) {
    const Obstacle<Shape>& obst = (*shape_buf)[i];

    shape_insert_into<Shape, Cell>(
        obst.shape, obst.pose, *map->data(), map->resolution(),
        Cell(1.0, obst.id),
        WorkLayout2d(threadIdx.x, blockDim.x, threadIdx.y, blockDim.y));
  }
}

void ObstacleManager::insert_in_map(DeviceMap& map) {
  if (circles_to_add_.size() > 0) {
    DeviceArray<Obstacle<Circle>> circle_buf =
        DeviceArray<Obstacle<Circle>>::from(circles_to_add_);

    device_map_insert_shapes<Circle>
        <<<1, dim3(32, 32)>>>(map.device_map(), circle_buf.device_handle());
  }

  if (rectangles_to_add_.size() > 0) {
    DeviceArray<Obstacle<Rectangle>> rect_buf =
        DeviceArray<Obstacle<Rectangle>>::from(rectangles_to_add_);

    device_map_insert_shapes<Rectangle>
        <<<1, dim3(32, 32)>>>(map.device_map(), rect_buf.device_handle());
  }
}

const std::string& ObstacleManager::get_obstacle_name(uint8_t id) {
  return id_to_name_[id];
}

}  // namespace gpu_planning
