#include "geometry.hpp"

namespace gpu_planning {

Point::Point() : x{0.f}, y{0.f} {}

Point::Point(float x, float y) : x{x}, y{y} {}

Point Point::operator-() const { return Point(-x, -y); }

Point Point::operator+(Point rhs) const { return Point(x + rhs.x, y + rhs.y); }

Point Point::operator-(Point rhs) const { return operator+(-rhs); }

}  // namespace gpu_planning
