#pragma once

#include <vector>

#include "stddef.h"

namespace gpu_planning {

struct Color {
  Color();
  Color(char r, char g, char b);

  static const Color BLACK;
  static const Color WHITE;

  static const Color RED;
  static const Color GREEN;
  static const Color BLUE;
  static const Color YELLOW;

  char bgr[3];
};

class Image {
 public:
  Image();
  Image(size_t width, size_t height);

 private:
  size_t width_;
  size_t height_;
  std::vector<Color> data_;
};

}  // namespace gpu_planning
