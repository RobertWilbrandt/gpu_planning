#include "thread_block.hpp"

namespace gpu_planning {

__host__ __device__ ThreadBlock1d::ThreadBlock1d() : x_{0}, dim_x_{0} {}

__host__ __device__ ThreadBlock1d::ThreadBlock1d(int x, int dim_x)
    : x_{x}, dim_x_{dim_x} {}

__host__ __device__ int ThreadBlock1d::x() const { return x_; }

__host__ __device__ int ThreadBlock1d::dim_x() const { return dim_x_; }

__host__ __device__ void ThreadBlock1d::sync() const {
#ifdef __CUDA_ARCH__
  __syncthreads();
#endif
}

__host__ __device__ ThreadBlock2d::ThreadBlock2d()
    : x_{0}, y_{0}, dim_x_{0}, dim_y_{0} {}

__host__ __device__ ThreadBlock2d::ThreadBlock2d(int x, int y, int dim_x,
                                                 int dim_y)
    : x_{x}, y_{y}, dim_x_{dim_x}, dim_y_{dim_y} {}

__host__ __device__ int ThreadBlock2d::x() const { return x_; }

__host__ __device__ int ThreadBlock2d::y() const { return y_; }

__host__ __device__ int ThreadBlock2d::dim_x() const { return dim_x_; }

__host__ __device__ int ThreadBlock2d::dim_y() const { return dim_y_; }

__host__ __device__ void ThreadBlock2d::sync() const {
#ifdef __CUDA_ARCH__
  __syncthreads();
#endif
}

__host__ __device__ ThreadBlock3d::ThreadBlock3d()
    : x_{0}, y_{0}, z_{0}, dim_x_{0}, dim_y_{0}, dim_z_{0} {}

__host__ __device__ ThreadBlock3d::ThreadBlock3d(int x, int y, int z, int dim_x,
                                                 int dim_y, int dim_z)
    : x_{x}, y_{y}, z_{z}, dim_x_{dim_x}, dim_y_{dim_y}, dim_z_{dim_z} {}

__host__ ThreadBlock3d ThreadBlock3d::host() {
  return ThreadBlock3d(0, 0, 0, 1, 1, 1);
}

__device__ ThreadBlock3d ThreadBlock3d::device_current() {
  return ThreadBlock3d(threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x,
                       blockDim.y, blockDim.z);
}

__host__ __device__ int ThreadBlock3d::x() const { return x_; }

__host__ __device__ int ThreadBlock3d::y() const { return y_; }

__host__ __device__ int ThreadBlock3d::z() const { return z_; }

__host__ __device__ int ThreadBlock3d::dim_x() const { return dim_x_; }

__host__ __device__ int ThreadBlock3d::dim_y() const { return dim_y_; }

__host__ __device__ int ThreadBlock3d::dim_z() const { return dim_z_; }

__host__ __device__ ThreadBlock1d ThreadBlock3d::to_1d() const {
  return ThreadBlock1d((z_ * dim_y_ + y_) * dim_x_ + x_,
                       dim_x_ * dim_y_ * dim_z_);
}

__host__ __device__ void ThreadBlock3d::sync() const {
#ifdef __CUDA_ARCH__
  __syncthreads();
#endif
}

__host__ __device__ ThreadBlock2d ThreadBlock3d::slice_z() const {
  return ThreadBlock2d(x_, y_, dim_x_, dim_y_);
}

}  // namespace gpu_planning
