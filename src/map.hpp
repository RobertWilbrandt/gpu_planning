#pragma once

#include "device_2d_array_handle.hpp"
#include "logging.hpp"

namespace gpu_planning {

class DeviceMap;

class Map {
 public:
  Map();
  Map(float width, float height, size_t resolution, Logger* log);

  ~Map();

  float width() const;
  float height() const;
  size_t resolution() const;

  DeviceMap* device_map() const;

  void get_data(float* dest, size_t max_width, size_t max_height,
                size_t* result_width, size_t* result_height);

  void add_obstacle_circle(float x, float y, float radius);
  void add_obstacle_rect(float x, float y, float width, float height);

 private:
  DeviceMap* map_;
  Device2dArrayHandle data_;
  size_t resolution_;

  Logger* log_;
};

}  // namespace gpu_planning
