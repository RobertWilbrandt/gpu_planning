#pragma once

#include "cuda_runtime_api.h"
#include "robot.hpp"

namespace gpu_planning {

struct CTrajectorySegment {
  __host__ __device__ CTrajectorySegment();
  __host__ __device__ CTrajectorySegment(const Configuration& start,
                                         const Configuration& end);

  Configuration start;
  Configuration end;
};

struct CTrajectory {};

}  // namespace gpu_planning
