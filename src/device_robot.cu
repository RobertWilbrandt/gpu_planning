#include "device_robot.cuh"

namespace gpu_planning {

__host__ DeviceRobot::DeviceRobot()
    : base_{}, l1_{0.f}, l2_{0.f}, ee_w_{0.f}, ee_h_{0.f} {}

__host__ DeviceRobot::DeviceRobot(float bx, float by, float l1, float l2,
                                  float ee_w, float ee_h)
    : base_{bx, by, M_PI / 2}, l1_{l1}, l2_{l2}, ee_w_{ee_w}, ee_h_{ee_h} {}

__device__ Pose<float> DeviceRobot::base() const { return base_; }

__device__ Pose<float> DeviceRobot::fk_elbow(const Configuration& conf) const {
  Transform<float> link(l1_, 0, 0);
  return link.rotate(conf.joints[0]) * base();
}

__device__ Pose<float> DeviceRobot::fk_ee(const Configuration& conf) const {
  Transform<float> link(l2_, 0, 0);
  return link.rotate(conf.joints[1]) * fk_elbow(conf);
}

}  // namespace gpu_planning
