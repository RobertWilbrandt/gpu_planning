#include "trajectory.hpp"

namespace gpu_planning {

__host__ __device__ TrajectorySegment::TrajectorySegment() : start{}, end{} {}

__host__ __device__ TrajectorySegment::TrajectorySegment(
    const Configuration& start, const Configuration& end)
    : start{start}, end{end} {}

__host__ __device__ Configuration
TrajectorySegment::interpolate(float a) const {
  const float na = 1 - a;
  return Configuration(na * start.joints[0] + a * end.joints[0],
                       na * start.joints[1] + a * end.joints[1],
                       na * start.joints[2] + a * end.joints[2]);
}

}  // namespace gpu_planning
