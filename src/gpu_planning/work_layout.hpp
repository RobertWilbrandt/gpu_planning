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

struct WorkLayout3d {
  __host__ __device__ WorkLayout3d();
  __host__ __device__ WorkLayout3d(int offset_x, int stride_x, int offset_y,
                                   int stride_y, int offset_z, int stride_z);

  __host__ __device__ static WorkLayout3d from(const dim3& threadIdx,
                                               const dim3& blockDim);

  int offset_x;
  int stride_x;
  int offset_y;
  int stride_y;
  int offset_z;
  int stride_z;
};

}  // namespace gpu_planning
