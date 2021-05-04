#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "cuda_runtime_api.h"
#include "geometry.hpp"
#include "map.hpp"
#include "stdint.h"

namespace gpu_planning {

struct Circle {
  __host__ __device__ Circle();
  __host__ __device__ Circle(const Position<float>& position, float radius,
                             uint8_t id);

  Position<float> position;
  float radius;
  uint8_t id;
};

struct Rectangle {
  __host__ __device__ Rectangle();
  __host__ __device__ Rectangle(const Position<float>& position, float width,
                                float height, uint8_t id);

  Position<float> position;
  float width;
  float height;
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

 private:
  uint8_t id_cnt_;
  std::unordered_map<uint8_t, std::string> id_to_name_;

  std::vector<Circle> circles_to_add_;
  std::vector<Rectangle> rectangles_to_add_;
};

}  // namespace gpu_planning
