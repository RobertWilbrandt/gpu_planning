#include "device_robot.cuh"

namespace gpu_planning {

__host__ DeviceConfiguration::DeviceConfiguration() : joints{0.f, 0.f, 0.f} {}

__host__ DeviceConfiguration::DeviceConfiguration(float j1, float j2, float j3)
    : joints{j1, j2, j3} {}

__host__ DeviceRobot::DeviceRobot()
    : bx_{0.f}, by_{0.f}, l1_{0.f}, l2_{0.f}, ee_w_{0.f}, ee_h_{0.f} {}

__host__ DeviceRobot::DeviceRobot(float bx, float by, float l1, float l2,
                                  float ee_w, float ee_h)
    : bx_{bx}, by_{by}, l1_{l1}, l2_{l2}, ee_w_{ee_w}, ee_h_{ee_h} {}

}  // namespace gpu_planning
