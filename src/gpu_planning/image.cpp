#include "image.hpp"

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

}  // namespace gpu_planning
