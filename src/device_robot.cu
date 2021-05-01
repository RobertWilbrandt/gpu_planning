#include "device_robot.cuh"

namespace gpu_planning {

__host__ DeviceConfiguration::DeviceConfiguration() : joints{0.f, 0.f, 0.f} {}

__host__ DeviceConfiguration::DeviceConfiguration(float j1, float j2, float j3)
    : joints{j1, j2, j3} {}

__device__ DevicePose::DevicePose() : x{0.f}, y{0.f}, theta{0.f} {}

__device__ DevicePose::DevicePose(float x, float y, float theta)
    : x{x}, y{y}, theta{theta} {}

__host__ DeviceRobot::DeviceRobot()
    : base_{}, l1_{0.f}, l2_{0.f}, ee_w_{0.f}, ee_h_{0.f} {}

__host__ DeviceRobot::DeviceRobot(float bx, float by, float l1, float l2,
                                  float ee_w, float ee_h)
    : base_{bx, by, M_PI / 2}, l1_{l1}, l2_{l2}, ee_w_{ee_w}, ee_h_{ee_h} {}

__device__ DevicePose DeviceRobot::base() const {
  return DevicePose(base_.position.x, base_.position.y, base_.orientation);
}

__device__ DevicePose DeviceRobot::fk_elbow(DeviceConfiguration* conf) const {
  DevicePose b = base();
  Transform<float> link(l1_, 0, 0);
  Pose<float> result =
      link.rotate(conf->joints[0]) * Pose<float>(b.x, b.y, b.theta);
  return DevicePose(result.position.x, result.position.y, result.orientation);
}

__device__ DevicePose DeviceRobot::fk_ee(DeviceConfiguration* conf) const {
  DevicePose b = fk_elbow(conf);
  Transform<float> link(l2_, 0, 0);
  Pose<float> result =
      link.rotate(conf->joints[1]) * Pose<float>(b.x, b.y, b.theta);
  return DevicePose(result.position.x, result.position.y, result.orientation);
}

}  // namespace gpu_planning
