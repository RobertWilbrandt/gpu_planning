#pragma once

#include "cuda_runtime_api.h"

namespace gpu_planning {

template <typename T>
struct Position {
  __host__ __device__ Position();
  __host__ __device__ Position(T x, T y);

  T x;
  T y;
};

template <typename T>
struct Vector {
  __host__ __device__ Vector();
  __host__ __device__ Vector(T x, T y);

  T x;
  T y;
};

template <typename T>
__host__ __device__ Vector<T> operator-(Vector<T> v);
template <typename T>
__host__ __device__ Vector<T> operator+(Vector<T> v1, Vector<T> v2);
template <typename T>
__host__ __device__ Vector<T> operator-(Vector<T> v1, Vector<T> v2);

template <typename T>
__host__ __device__ Position<T> operator+(Position<T> p, Vector<T> v);
template <typename T>
__host__ __device__ Position<T> operator-(Position<T> p, Vector<T> v);
template <typename T>
__host__ __device__ Vector<T> operator-(Position<T> p1, Position<T> p2);

/*
 * Template implementations
 */

template <typename T>
__host__ __device__ Position<T>::Position() : x{0}, y{0} {}

template <typename T>
__host__ __device__ Position<T>::Position(T x, T y) : x{x}, y{y} {}

template <typename T>
__host__ __device__ Vector<T>::Vector() : x{0}, y{0} {}

template <typename T>
__host__ __device__ Vector<T>::Vector(T x, T y) : x{x}, y{y} {}

template <typename T>
__host__ __device__ Vector<T> operator-(Vector<T> v) {
  return Vector<T>(-v.x, -v.y);
}

template <typename T>
__host__ __device__ Vector<T> operator+(Vector<T> v1, Vector<T> v2) {
  return Vector<T>(v1.x + v2.x, v1.y + v2.y);
}

template <typename T>
__host__ __device__ Vector<T> operator-(Vector<T> v1, Vector<T> v2) {
  return v1 + (-v2);
}

template <typename T>
__host__ __device__ Position<T> operator+(Position<T> p, Vector<T> v) {
  return Position<T>(p.x + v.x, p.y + v.y);
}

template <typename T>
__host__ __device__ Position<T> operator-(Position<T> p, Vector<T> v) {
  return p + (-v);
}

template <typename T>
__host__ __device__ Vector<T> operator-(Position<T> p1, Position<T> p2) {
  return Vector<T>(p1.x - p2.x, p1.y - p2.y);
}

}  // namespace gpu_planning
