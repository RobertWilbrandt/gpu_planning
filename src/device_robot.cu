#include "device_robot.cuh"

namespace gpu_planning {

__host__ DeviceConfiguration::DeviceConfiguration() : joints{0.f, 0.f, 0.f} {}

__host__ DeviceConfiguration::DeviceConfiguration(float j1, float j2, float j3)
    : joints{j1, j2, j3} {}

__device__ DevicePose::DevicePose() : x{0.f}, y{0.f}, theta{0.f} {}

__device__ DevicePose::DevicePose(float x, float y, float theta)
    : x{x}, y{y}, theta{theta} {}

__host__ DeviceRobot::DeviceRobot()
    : bx_{0.f}, by_{0.f}, l1_{0.f}, l2_{0.f}, ee_w_{0.f}, ee_h_{0.f} {}

__host__ DeviceRobot::DeviceRobot(float bx, float by, float l1, float l2,
                                  float ee_w, float ee_h)
    : bx_{bx}, by_{by}, l1_{l1}, l2_{l2}, ee_w_{ee_w}, ee_h_{ee_h} {}

__device__ DevicePose DeviceRobot::base() const {
  return DevicePose(bx_, by_, 0.f);
}

__device__ DevicePose DeviceRobot::fk_elbow(DeviceConfiguration* conf) const {
  DevicePose b = base();
  float base_theta = b.theta + conf->joints[0];
  return DevicePose(b.x + sin(base_theta) * l1_, b.y + cos(base_theta) * l1_,
                    base_theta);
}

__device__ DevicePose DeviceRobot::fk_ee(DeviceConfiguration* conf) const {
  DevicePose b = fk_elbow(conf);
  float base_theta = b.theta + conf->joints[1];
  return DevicePose(b.x + sin(base_theta) * l2_, b.y + cos(base_theta) * l2_,
                    base_theta);
}

}  // namespace gpu_planning
