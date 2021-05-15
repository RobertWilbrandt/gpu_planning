#pragma once

#include "cuda_runtime_api.h"

namespace gpu_planning {

struct WorkLayout2d {
  __host__ __device__ WorkLayout2d();
  __host__ __device__ WorkLayout2d(int offset_x, int stride_x, int offset_y,
                                   int stride_y);

  int offset_x;
  int stride_x;
  int offset_y;
  int stride_y;
};

}  // namespace gpu_planning
