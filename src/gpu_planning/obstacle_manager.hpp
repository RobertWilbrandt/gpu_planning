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
  __host__ __device__ Obstacle(const Shape& shape,
                               const Position<float>& postion, uint8_t id);

  Shape shape;
  Position<float> position;
  uint8_t id;
};

class ObstacleManager {
 public:
  ObstacleManager();

  void add_static_circle(const Position<float>& position, float radius,
                         const std::string& name);
  void add_static_rectangle(const Position<float>& position, float width,
                            float height, const std::string& name);

  void insert_in_map(DeviceMap& map);

  const std::string& get_obstacle_name(uint8_t id);

 private:
  uint8_t id_cnt_;
  std::unordered_map<uint8_t, std::string> id_to_name_;

  std::vector<Obstacle<Circle>> circles_to_add_;
  std::vector<Obstacle<Rectangle>> rectangles_to_add_;
};

template <typename Shape>
Obstacle<Shape>::Obstacle() : shape{}, position{}, id{0} {}

template <typename Shape>
Obstacle<Shape>::Obstacle(const Shape& shape, const Position<float>& position,
                          uint8_t id)
    : shape{shape}, position{position}, id{id} {}

}  // namespace gpu_planning
