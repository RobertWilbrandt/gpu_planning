#pragma once

#include <memory>

namespace gpu_planning {

class DeviceArray2D {
 public:
  DeviceArray2D();
  DeviceArray2D(size_t width, size_t height, size_t mem_size);

  ~DeviceArray2D();

  size_t width() const;
  size_t height() const;

  void clear();

  void read(size_t x, size_t y, size_t w, size_t h, void* dest);
  void write(size_t x, size_t y, size_t w, size_t h, void* src);

 private:
  void* extent_;
  void* pitched_ptr_;
};

class Map {
 public:
  Map();
  Map(size_t width, size_t height);

  ~Map();

 private:
  DeviceArray2D map_;
};

}  // namespace gpu_planning
