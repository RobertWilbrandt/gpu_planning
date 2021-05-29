#include "work_layout.hpp"

namespace gpu_planning {

__host__ __device__ WorkLayout2d::WorkLayout2d()
    : offset_x{0}, stride_x{0}, offset_y{0}, stride_y{0} {}

__host__ __device__ WorkLayout2d::WorkLayout2d(int offset_x, int stride_x,
                                               int offset_y, int stride_y)
    : offset_x{offset_x},
      stride_x{stride_x},
      offset_y{offset_y},
      stride_y{stride_y} {}

__host__ __device__ WorkLayout3d::WorkLayout3d()
    : offset_x{0},
      stride_x{0},
      offset_y{0},
      stride_y{0},
      offset_z{0},
      stride_z{0} {}

__host__ __device__ WorkLayout3d::WorkLayout3d(int offset_x, int stride_x,
                                               int offset_y, int stride_y,
                                               int offset_z, int stride_z)
    : offset_x{offset_x},
      stride_x{stride_x},
      offset_y{offset_y},
      stride_y{stride_y},
      offset_z{offset_z},
      stride_z{stride_z} {}

__host__ __device__ WorkLayout3d WorkLayout3d::from(const dim3& threadIdx,
                                                    const dim3& blockDim) {
  return WorkLayout3d(threadIdx.x, blockDim.x, threadIdx.y, blockDim.y,
                      threadIdx.z, blockDim.z);
}

}  // namespace gpu_planning
