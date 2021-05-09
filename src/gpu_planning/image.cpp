#include "image.hpp"

#include <cstring>
#include <fstream>

namespace gpu_planning {

Color::Color() : bgr{0, 0, 0} {}

Color::Color(char r, char g, char b) : bgr{b, g, r} {}

const Color Color::BLACK = Color(0, 0, 0);
const Color Color::WHITE = Color(255, 255, 255);
const Color Color::RED = Color(255, 0, 0);
const Color Color::GREEN = Color(0, 255, 0);
const Color Color::BLUE = Color(0, 0, 255);
const Color Color::YELLOW = Color(255, 255, 0);

Image::Image() : width_{0}, height_{0}, data_{} {}

Image::Image(size_t width, size_t height)
    : width_{width}, height_{height}, data_{width * height, Color::WHITE} {}

size_t Image::width() const { return width_; }

size_t Image::height() const { return height_; }

Box<size_t> Image::area() const {
  return Box<size_t>(0, width_ - 1, 0, height_ - 1);
}

Array2d<Color> Image::as_array() {
  return Array2d<Color>(data_.data(), width_, height_, width_ * sizeof(Color));
}

Color& Image::pixel(const Position<size_t>& pos) {
  return data_[pos.y * width_ + pos.x];
}

const Color& Image::pixel(const Position<size_t>& pos) const {
  return data_[pos.y * width_ + pos.x];
}

void Image::draw_marker(const Position<size_t>& pos, const Color& color) {
  try_draw_point(pos, color);
  try_draw_point(pos + Translation<size_t>(-1, 0), color);
  try_draw_point(pos + Translation<size_t>(1, 0), color);
  try_draw_point(pos + Translation<size_t>(0, -1), color);
  try_draw_point(pos + Translation<size_t>(0, 1), color);
}

void Image::draw_line(const Position<size_t>& from, const Position<size_t>& to,
                      const Color& color) {
  Translation<ssize_t> delta = to.cast<ssize_t>() - from.cast<ssize_t>();
  Position<size_t> start = from;
  bool transposed = false;

  if (abs(delta.x) < abs(delta.y)) {
    delta = Translation<ssize_t>(delta.y, delta.x);
    start = Position<size_t>(start.y, start.x);
    transposed = true;
  }

  const Translation<ssize_t> axis_dir = delta.signum();
  for (ssize_t ix = 0; ix != delta.x; ix += axis_dir.x) {
    const size_t x = start.x + ix;
    const size_t y = start.y + ix * delta.y / delta.x;

    if (transposed) {
      try_draw_point(Position<size_t>(y, x), color);
    } else {
      try_draw_point(Position<size_t>(x, y), color);
    }
  }
}

void Image::try_draw_point(const Position<size_t>& pos, const Color& color) {
  if (area().is_inside(pos)) {
    pixel(pos) = color;
  }
}

uint16_t little_endian_16(uint16_t val) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return val;
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return __builtin_bswap16(val);
#else
#error "Only little and big endian machines are supported"
#endif
}

uint32_t little_endian_32(uint32_t val) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return val;
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return __builtin_bswap32(val);
#else
#error "Only little and big endian machines are supported"
#endif
}

void Image::save_bmp(const std::string& path) const {
  std::ofstream out(path, std::ios::binary | std::ios::out);

  const size_t row_size = ((width_ * 3 - 1) / 4 + 1) * 4;

  struct __attribute__((__packed__)) {
    uint16_t header_field;
    uint32_t file_size;
    uint16_t reserved_1;
    uint16_t reserved_2;
    uint32_t data_offset;
  } file_header;

  struct __attribute__((__packed__)) {
    uint32_t header_size;
    uint16_t bitmap_width;
    uint16_t bitmap_height;
    uint16_t num_color_planes;
    uint16_t bits_per_pixel;
  } dib_header;

  file_header.header_field = little_endian_16(0x4D42);
  file_header.file_size = little_endian_32(
      sizeof(file_header) + sizeof(dib_header) + height_ * row_size);
  file_header.reserved_1 = 0;
  file_header.reserved_2 = 0;
  file_header.data_offset =
      little_endian_32(sizeof(file_header) + sizeof(dib_header));
  out.write(reinterpret_cast<char*>(&file_header), sizeof(file_header));

  dib_header.header_size = little_endian_32(sizeof(dib_header));
  dib_header.bitmap_width = width_;
  dib_header.bitmap_height = height_;
  dib_header.num_color_planes = little_endian_16(1);
  dib_header.bits_per_pixel = little_endian_16(24);
  out.write(reinterpret_cast<char*>(&dib_header), sizeof(dib_header));

  char buf[3];
  for (size_t y = 0; y < height_; ++y) {
    for (size_t x = 0; x < width_; ++x) {
      const Position<size_t> pos(x, y);
      std::memcpy(buf, pixel(pos).bgr, 3 * sizeof(char));

      out.write(buf, 3);
    }

    // Line padding
    std::memset(buf, 0, 3 * sizeof(char));
    out.write(buf, row_size - 3 * width_);
  }

  out.close();
  if (!out) {
    throw std::runtime_error{"Error while writing to bmp file"};
  }
}

}  // namespace gpu_planning
