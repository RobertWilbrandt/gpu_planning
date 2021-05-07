#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "geometry.hpp"
#include "map.hpp"
#include "shape.hpp"
#include "stdint.h"

namespace gpu_planning {

template <typename Shape>
struct Obstacle {
  __host__ __device__ Obstacle();
  __host__ __device__ Obstacle(const Shape& shape, const Pose<float>& pose,
                               uint8_t id);

  Shape shape;
  Pose<float> pose;
  uint8_t id;
};

class ObstacleManager {
 public:
  ObstacleManager();

  void add_static_circle(const Pose<float>& pose, float radius,
                         const std::string& name);
  void add_static_rectangle(const Pose<float>& pose, float width, float height,
                            const std::string& name);

  void insert_in_map(DeviceMap& map);

  const std::string& get_obstacle_name(uint8_t id);

 private:
  uint8_t id_cnt_;
  std::unordered_map<uint8_t, std::string> id_to_name_;

  std::vector<Obstacle<Circle>> circles_to_add_;
  std::vector<Obstacle<Rectangle>> rectangles_to_add_;
};

template <typename Shape>
Obstacle<Shape>::Obstacle() : shape{}, pose{}, id{0} {}

template <typename Shape>
Obstacle<Shape>::Obstacle(const Shape& shape, const Pose<float>& pose,
                          uint8_t id)
    : shape{shape}, pose{pose}, id{id} {}

}  // namespace gpu_planning
