#pragma once

#include "collision_checker.hpp"
#include "cuda_runtime_api.h"
#include "robot.hpp"
#include "trajectory.hpp"
#include "work_buffer.hpp"

namespace gpu_planning {

__global__ void check_seg_collisions(
    CollisionChecker* collision_checker,
    WorkBlock<TrajectorySegment, CollisionCheckResult>* segments,
    WorkBlock<Configuration, CollisionCheckResult>* conf_work);

}
