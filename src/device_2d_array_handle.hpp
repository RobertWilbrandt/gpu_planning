#pragma once

#include "logging.hpp"

namespace gpu_planning {

class Device2dArray;

class Device2dArrayHandle {
 public:
  Device2dArrayHandle();
  Device2dArrayHandle(size_t width, size_t height, size_t depth, Logger* log);

  ~Device2dArrayHandle();

  size_t width() const;
  size_t height() const;
  size_t depth() const;

  Device2dArray* device_array() const;

  void clear();

  void get_data(void* dest);

 private:
  Device2dArray* device_array_;

  size_t width_;
  size_t height_;
  size_t depth_;
  size_t pitch_;
  void* data_;

  Logger* log_;
};

}  // namespace gpu_planning
