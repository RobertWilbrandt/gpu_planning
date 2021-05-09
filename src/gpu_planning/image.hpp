#pragma once

#include <string>
#include <vector>

#include "geometry.hpp"
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

  size_t width() const;
  size_t height() const;

  Box<size_t> area() const;

  Color& pixel(const Position<size_t>& pos);
  const Color& pixel(const Position<size_t>& pos) const;

  void draw_marker(const Position<size_t>& pos, const Color& color);
  void draw_line(const Position<size_t>& from, const Position<size_t>& to,
                 const Color& color);

  void save_bmp(const std::string& path) const;

 private:
  void try_draw_point(const Position<size_t>& pos, const Color& color);

  size_t width_;
  size_t height_;
  std::vector<Color> data_;
};

}  // namespace gpu_planning
