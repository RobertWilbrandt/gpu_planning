#include "trajectory.hpp"

namespace gpu_planning {

__host__ __device__ CTrajectorySegment::CTrajectorySegment() : start{}, end{} {}

__host__ __device__ CTrajectorySegment::CTrajectorySegment(
    const Configuration& start, const Configuration& end)
    : start{start}, end{end} {}

}  // namespace gpu_planning
