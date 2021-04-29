#pragma once

namespace gpu_planning {

struct DeviceConfiguration {
  __host__ DeviceConfiguration();
  __host__ DeviceConfiguration(float j1, float j2, float j3);

  float joints[3];
};

struct DevicePose {
  __device__ DevicePose();
  __device__ DevicePose(float x, float y, float theta);

  float x;
  float y;
  float theta;
};

class DeviceRobot {
 public:
  __host__ DeviceRobot();
  __host__ DeviceRobot(float bx, float by, float l1, float l2, float ee_w,
                       float ee_h);

  __device__ DevicePose base() const;
  __device__ DevicePose fk_elbow(DeviceConfiguration* conf) const;
  __device__ DevicePose fk_ee(DeviceConfiguration* conf) const;

 private:
  float bx_;
  float by_;
  float l1_;
  float l2_;
  float ee_w_;
  float ee_h_;
};

}  // namespace gpu_planning
