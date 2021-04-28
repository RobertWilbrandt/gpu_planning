#pragma once

namespace gpu_planning {

struct Point {
  Point();
  Point(float x, float y);

  float x;
  float y;

  Point operator-() const;
  Point operator+(Point rhs) const;
  Point operator-(Point rhs) const;
};

}  // namespace gpu_planning
