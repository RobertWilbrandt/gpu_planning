#pragma once

#include "cuda_runtime_api.h"
#include "robot.hpp"

namespace gpu_planning {

struct TrajectorySegment {
  __host__ __device__ TrajectorySegment();
  __host__ __device__ TrajectorySegment(const Configuration& start,
                                        const Configuration& end);

  __host__ __device__ Configuration interpolate(float a) const;

  Configuration start;
  Configuration end;
};

struct Trajectory {};

}  // namespace gpu_planning
