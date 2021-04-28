#include <stdexcept>

#include "cuda_util.hpp"
#include "device_robot.cuh"
#include "robot.hpp"

namespace gpu_planning {

Configuration::Configuration() : joints{0.f, 0.f, 0.f} {}

Configuration::Configuration(float j1, float j2, float j3)
    : joints{j1, j2, j3} {}

Robot::Robot()
    : device_robot_{nullptr}, l1_{0.f}, l2_{0.f}, ee_w_{0.f}, ee_h_{0.f} {}

Robot::Robot(float l1, float l2, float ee_w, float ee_h)
    : device_robot_{nullptr}, l1_{l1}, l2_{l2}, ee_w_{ee_w}, ee_h_{ee_h} {
  CHECK_CUDA(cudaMalloc(&device_robot_, sizeof(DeviceRobot)),
             "Could not allocate device storage for robot description");

  DeviceRobot device_robot(l1, l2, ee_w, ee_h);
  CHECK_CUDA(cudaMemcpy(device_robot_, &device_robot, sizeof(DeviceRobot),
                        cudaMemcpyHostToDevice),
             "Could not memcpy robot description to device");
}

Robot::~Robot() {
  if (device_robot_ != nullptr) {
    CHECK_CUDA(cudaFree(device_robot_),
               "Could not free device robot description");
  }
}

DeviceRobot* Robot::device_robot() const { return device_robot_; }

}  // namespace gpu_planning
