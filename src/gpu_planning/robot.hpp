#pragma once

#include "cuda_runtime_api.h"
#include "geometry.hpp"
#include "shape.hpp"

namespace gpu_planning {

struct Configuration {
  __host__ __device__ Configuration();
  __host__ __device__ Configuration(float j1, float j2, float j3);

  float joints[3];
};

class Robot {
 public:
  __host__ __device__ Robot();
  __host__ __device__ Robot(Pose<float> base, float l1, float l2,
                            const Rectangle& ee);

  __host__ __device__ const Rectangle& ee() const;

  __host__ __device__ Pose<float> base() const;
  __host__ __device__ Pose<float> fk_elbow(const Configuration& conf) const;
  __host__ __device__ Pose<float> fk_ee(const Configuration& conf) const;

 private:
  Pose<float> base_;
  float l1_;
  float l2_;
  Rectangle ee_;
};

class DeviceRobot {
 public:
  DeviceRobot();
  DeviceRobot(Pose<float> base, float l1, float l2, const Rectangle& ee);

  ~DeviceRobot();

  Robot& robot();
  Robot* device_handle() const;

  Pose<float> base() const;
  Pose<float> fk_elbow(const Configuration& conf) const;
  Pose<float> fk_ee(const Configuration& conf) const;

 private:
  Robot robot_;
  Robot* device_handle_;
};

}  // namespace gpu_planning
