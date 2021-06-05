#pragma once

#include "cuda_runtime_api.h"

namespace gpu_planning {

class ThreadBlock1d {
 public:
  __host__ __device__ ThreadBlock1d();
  __host__ __device__ ThreadBlock1d(int x, int dim_x);

  __host__ __device__ int x() const;

  __host__ __device__ int dim_x() const;

  __host__ __device__ void sync() const;

 private:
  int x_;
  int dim_x_;
};

class ThreadBlock2d {
 public:
  __host__ __device__ ThreadBlock2d();
  __host__ __device__ ThreadBlock2d(int x, int y, int dim_x, int dim_y);

  __host__ __device__ int x() const;
  __host__ __device__ int y() const;

  __host__ __device__ int dim_x() const;
  __host__ __device__ int dim_y() const;

  __host__ __device__ void sync() const;

 private:
  int x_;
  int y_;
  int dim_x_;
  int dim_y_;
};

class ThreadBlock3d {
 public:
  __host__ __device__ ThreadBlock3d();
  __host__ __device__ ThreadBlock3d(int x, int y, int z, int dim_x, int dim_y,
                                    int dim_z);

  static __host__ ThreadBlock3d host();
  static __device__ ThreadBlock3d device_current();

  __host__ __device__ int x() const;
  __host__ __device__ int y() const;
  __host__ __device__ int z() const;

  __host__ __device__ int dim_x() const;
  __host__ __device__ int dim_y() const;
  __host__ __device__ int dim_z() const;

  __host__ __device__ ThreadBlock1d to_1d() const;

  __host__ __device__ void sync() const;

  __host__ __device__ ThreadBlock2d slice_z() const;

 private:
  int x_;
  int y_;
  int z_;
  int dim_x_;
  int dim_y_;
  int dim_z_;
};

}  // namespace gpu_planning
