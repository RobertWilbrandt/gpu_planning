#pragma once

#include "cuda_runtime_api.h"
#include "geometry.hpp"

namespace gpu_planning {

struct Configuration {
  __host__ __device__ Configuration();
  __host__ __device__ Configuration(float j1, float j2, float j3);

  float joints[3];
};

class DeviceRobot;

class Robot {
 public:
  Robot();
  Robot(Pose<float> base, float l1, float l2, float ee_w, float ee_h);

  ~Robot();

  DeviceRobot* device_robot() const;

  Pose<float> base() const;
  Pose<float> fk_elbow(const Configuration& conf) const;
  Pose<float> fk_ee(const Configuration& conf) const;

 private:
  DeviceRobot* device_robot_;

  Pose<float> base_;
  float l1_;
  float l2_;
  float ee_w_;
  float ee_h_;
};

}  // namespace gpu_planning
