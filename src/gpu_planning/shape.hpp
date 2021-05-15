#pragma once

#include "array_2d.hpp"
#include "cuda_runtime_api.h"
#include "geometry.hpp"
#include "work_layout.hpp"

namespace gpu_planning {

template <typename Shape, typename Value>
__host__ __device__ void shape_insert_into(
    const Shape& shape, const Pose<float>& pose, Array2d<Shape>& dest,
    size_t resolution, const Value& value, const WorkLayout2d& work_layout);

struct Circle {
  __host__ __device__ Circle();
  __host__ __device__ Circle(float radius);

  __host__ __device__ Box<float> bounding_box(float orientation) const;
  __host__ __device__ bool is_inside(const Position<float>& pos) const;

  float radius;
};

struct Rectangle {
  __host__ __device__ Rectangle();
  __host__ __device__ Rectangle(float width, float height);

  __host__ __device__ Box<float> bounding_box(float orientation) const;
  __host__ __device__ bool is_inside(const Position<float>& pos) const;

  float width;
  float height;
};

template <typename Shape, typename Value>
__host__ __device__ void shape_insert_into(
    const Shape& shape, const Pose<float>& pose, Array2d<Value>& dest,
    size_t resolution, const Value& value, const WorkLayout2d& work_layout) {
  const Box<size_t> dest_area = dest.area();

  const Box<float> bb = shape.bounding_box(pose.orientation)
                            .translate(pose.position.from_origin());
  const Box<size_t> bb_index(
      dest_area.clamp(bb.lower_left.scale_up(resolution).cast<size_t>()),
      dest_area.clamp(bb.upper_right.scale_up(resolution).cast<size_t>()));

  for (size_t y = bb_index.lower_left.y + work_layout.offset_y;
       y <= bb_index.upper_right.y; y += work_layout.stride_y) {
    for (size_t x = bb_index.lower_left.x + work_layout.offset_x;
         x <= bb_index.upper_right.x; x += work_layout.stride_x) {
      const Position<size_t> pos_index(x, y);
      const Position<float> pos =
          pos_index.cast<float>().scale_down(resolution);

      const Position<float> norm_pos =
          (Pose<float>(pos, 0).from_origin() * pose.from_origin().inverse() *
           Pose<float>())
              .position;

      if (shape.is_inside(norm_pos)) {
        dest.at(pos_index) = value;
      }
    }
  }
}

}  // namespace gpu_planning
