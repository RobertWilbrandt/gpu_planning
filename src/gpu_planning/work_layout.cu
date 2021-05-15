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

}  // namespace gpu_planning
