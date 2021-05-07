#pragma once

#include <iostream>

#include "cuda_runtime_api.h"

#ifndef __CUDACC__
#include "math.h"
#endif  // ifndef __CUDACC__

namespace gpu_planning {

template <typename T, template <typename> typename D>
struct Vector {
  __host__ __device__ Vector();
  __host__ __device__ Vector(T x, T y);

  template <typename To>
  __host__ __device__ D<To> cast() const;

  __host__ __device__ D<T> signum() const;

  T x;
  T y;
};

template <typename T, template <typename> typename D>
__host__ __device__ bool operator==(const Vector<T, D>& v1,
                                    const Vector<T, D>& v2);
template <typename T, template <typename> typename D>
__host__ __device__ bool operator!=(const Vector<T, D>& v1,
                                    const Vector<T, D>& v2);

template <typename T, template <typename> typename D>
std::ostream& operator<<(std::ostream& os, const Vector<T, D>& v);

template <typename T>
struct Position : public Vector<T, Position> {
  __host__ __device__ Position();
  __host__ __device__ Position(T x, T y);

  __host__ __device__ Position<T> scale_up(T factor) const;
  __host__ __device__ Position<T> scale_up(T x_fact, T y_fact) const;
  __host__ __device__ Position<T> scale_down(T factor) const;
  __host__ __device__ Position<T> scale_down(T x_fact, T y_fact) const;
};

template <typename T>
struct Translation : public Vector<T, Translation> {
  __host__ __device__ Translation();
  __host__ __device__ Translation(T x, T y);

  __host__ __device__ Translation<T> rotate(float alpha) const;
};

template <typename T>
__host__ __device__ Translation<T> operator-(Translation<T> v);
template <typename T>
__host__ __device__ Translation<T> operator+(Translation<T> v1,
                                             Translation<T> v2);
template <typename T>
__host__ __device__ Translation<T> operator-(Translation<T> v1,
                                             Translation<T> v2);

template <typename T>
__host__ __device__ Position<T> operator+(Position<T> p, Translation<T> v);
template <typename T>
__host__ __device__ Position<T> operator-(Position<T> p, Translation<T> v);
template <typename T>
__host__ __device__ Translation<T> operator-(Position<T> p1, Position<T> p2);

template <typename T>
struct Pose {
  __host__ __device__ Pose();
  __host__ __device__ Pose(Position<T> p, float orientation);
  __host__ __device__ Pose(T x, T y, float orientation);

  Position<T> position;
  float orientation;
};

template <typename T>
struct Transform {
  __host__ __device__ Transform();
  __host__ __device__ Transform(Translation<T> translation, float rotation);
  __host__ __device__ Transform(T x, T y, float rotation);

  __host__ __device__ Transform<T> rotate(float alpha);

  Translation<T> translation;
  float rotation;
};

template <typename T>
__host__ __device__ Transform<T> operator*(Transform<T> t1, Transform<T> t2);

template <typename T>
__host__ __device__ Pose<T> operator*(Transform<T> t, Pose<T> p);

template <typename T>
struct Box {
  __host__ __device__ Box();
  __host__ __device__ Box(Position<T> lower_left, Position<T> upper_right);
  __host__ __device__ Box(T left, T right, T bottom, T top);

  __host__ __device__ bool is_inside(Position<T> p) const;

  __host__ __device__ Position<T> clamp(Position<T> p) const;

  Position<T> lower_left;
  Position<T> upper_right;
};

/*
 * Template implementations
 */

template <typename T, template <typename> typename D>
__host__ __device__ Vector<T, D>::Vector() : x{0}, y{0} {}

template <typename T, template <typename> typename D>
__host__ __device__ Vector<T, D>::Vector(T x, T y) : x{x}, y{y} {}

template <typename T, template <typename> typename D>
template <typename To>
__host__ __device__ D<To> Vector<T, D>::cast() const {
  return D<To>(static_cast<To>(x), static_cast<To>(y));
}

template <typename T, template <typename> typename D>
__host__ __device__ D<T> Vector<T, D>::signum() const {
  return D<T>(x < 0 ? -1 : (x > 0 ? 1 : 0), y < 0 ? -1 : (y > 0 ? 1 : 0));
}

template <typename T, template <typename> typename D>
__host__ __device__ bool operator==(const Vector<T, D>& v1,
                                    const Vector<T, D>& v2) {
  return (v1.x == v2.x) && (v1.y == v2.y);
}

template <typename T, template <typename> typename D>
__host__ __device__ bool operator!=(const Vector<T, D>& v1,
                                    const Vector<T, D>& v2) {
  return !(v1 == v2);
}

template <typename T, template <typename> typename D>
std::ostream& operator<<(std::ostream& os, const Vector<T, D>& v) {
  os << "(" << v.x << ", " << v.y << ")";
  return os;
}

template <typename T>
__host__ __device__ Position<T>::Position() : Vector<T, Position>{} {}

template <typename T>
__host__ __device__ Position<T>::Position(T x, T y)
    : Vector<T, Position>{x, y} {}

template <typename T>
__host__ __device__ Position<T> Position<T>::scale_up(T factor) const {
  return scale_up(factor, factor);
}

template <typename T>
__host__ __device__ Position<T> Position<T>::scale_up(T x_fact,
                                                      T y_fact) const {
  return Position<T>(Vector<T, Position>::x * x_fact,
                     Vector<T, Position>::y * y_fact);
}

template <typename T>
__host__ __device__ Position<T> Position<T>::scale_down(T factor) const {
  return scale_down(factor, factor);
}

template <typename T>
__host__ __device__ Position<T> Position<T>::scale_down(T x_fact,
                                                        T y_fact) const {
  return Position<T>(Vector<T, Position>::x / x_fact,
                     Vector<T, Position>::y / y_fact);
}

template <typename T>
__host__ __device__ Translation<T>::Translation() : Vector<T, Translation>{} {}

template <typename T>
__host__ __device__ Translation<T>::Translation(T x, T y)
    : Vector<T, Translation>{x, y} {}

template <typename T>
__host__ __device__ Translation<T> Translation<T>::rotate(float alpha) const {
  float s = sinf(alpha);
  float c = cosf(alpha);
  return Translation<T>(
      Vector<T, Translation>::x * c - Vector<T, Translation>::y * s,
      Vector<T, Translation>::x * s + Vector<T, Translation>::y * c);
}

template <typename T>
__host__ __device__ Translation<T> operator-(Translation<T> v) {
  return Translation<T>(-v.x, -v.y);
}

template <typename T>
__host__ __device__ Translation<T> operator+(Translation<T> v1,
                                             Translation<T> v2) {
  return Translation<T>(v1.x + v2.x, v1.y + v2.y);
}

template <typename T>
__host__ __device__ Translation<T> operator-(Translation<T> v1,
                                             Translation<T> v2) {
  return v1 + (-v2);
}

template <typename T>
__host__ __device__ Position<T> operator+(Position<T> p, Translation<T> v) {
  return Position<T>(p.x + v.x, p.y + v.y);
}

template <typename T>
__host__ __device__ Position<T> operator-(Position<T> p, Translation<T> v) {
  return p + (-v);
}

template <typename T>
__host__ __device__ Translation<T> operator-(Position<T> p1, Position<T> p2) {
  return Translation<T>(p1.x - p2.x, p1.y - p2.y);
}

template <typename T>
__host__ __device__ Pose<T>::Pose() : position{0, 0}, orientation{0.f} {}

template <typename T>
__host__ __device__ Pose<T>::Pose(Position<T> p, float orientation)
    : position{p}, orientation{orientation} {}

template <typename T>
__host__ __device__ Pose<T>::Pose(T x, T y, float orientation)
    : position{x, y}, orientation{orientation} {}

template <typename T>
__host__ __device__ Transform<T>::Transform()
    : translation{0, 0}, rotation{0} {}

template <typename T>
__host__ __device__ Transform<T>::Transform(Translation<T> translation,
                                            float rotation)
    : translation{translation}, rotation{rotation} {}

template <typename T>
__host__ __device__ Transform<T>::Transform(T x, T y, float rotation)
    : translation{x, y}, rotation{rotation} {}

template <typename T>
__host__ __device__ Transform<T> Transform<T>::rotate(float alpha) {
  return Transform<T>(translation.rotate(alpha), rotation + alpha);
}

template <typename T>
__host__ __device__ Transform<T> operator*(Transform<T> t1, Transform<T> t2) {
  return Transform<T>(t2.translation + t1.translation.rotate(t2.rotation),
                      t2.rotation + t1.rotation);
}

template <typename T>
__host__ __device__ Pose<T> operator*(Transform<T> t, Pose<T> p) {
  return Pose<T>(p.position + t.translation.rotate(p.orientation),
                 p.orientation + t.rotation);
}

template <typename T>
__host__ __device__ Box<T>::Box() : lower_left{}, upper_right{} {}

template <typename T>
__host__ __device__ Box<T>::Box(Position<T> lower_left, Position<T> upper_right)
    : lower_left{lower_left}, upper_right{upper_right} {}

template <typename T>
__host__ __device__ Box<T>::Box(T left, T right, T bottom, T top)
    : lower_left{left, bottom}, upper_right{right, top} {}

template <typename T>
__host__ __device__ bool Box<T>::is_inside(Position<T> p) const {
  return (p.x >= lower_left.x) && (p.x <= upper_right.x) &&
         (p.y >= lower_left.y) && (p.y <= upper_right.y);
}

template <typename T>
__host__ __device__ Position<T> Box<T>::clamp(Position<T> p) const {
  return Position<T>(max(min(p.x, upper_right.x), lower_left.x),
                     max(min(p.y, upper_right.y), lower_left.y));
}

}  // namespace gpu_planning
