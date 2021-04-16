#pragma once

#include "logging.hpp"

struct cudaExtent;
struct cudaPitchedPtr;

namespace gpu_planning {

class Map {
 public:
  Map();
  Map(float width, float height, size_t resolution, Logger* log);

  ~Map();

  void print_debug();

  void add_obstacle_circle(float x, float y, float radius);

 private:
  cudaExtent* extent_;
  cudaPitchedPtr* pitched_ptr_;
  size_t resolution_;

  Logger* log_;
};

}  // namespace gpu_planning
