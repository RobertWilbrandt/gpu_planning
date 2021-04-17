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

  size_t width() const;
  size_t height() const;
  size_t resolution() const;

  void get_data(float* dest, size_t max_width, size_t max_height,
                size_t* result_width, size_t* result_height);

  void add_obstacle_circle(float x, float y, float radius);
  void add_obstacle_rect(float x, float y, float width, float height);

 private:
  cudaExtent* extent_;
  cudaPitchedPtr* pitched_ptr_;
  size_t resolution_;

  Logger* log_;
};

}  // namespace gpu_planning
