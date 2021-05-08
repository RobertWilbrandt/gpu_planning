#include <math.h>

#include "cuda_util.hpp"
#include "robot.hpp"

namespace gpu_planning {

__host__ __device__ Configuration::Configuration() : joints{0.f, 0.f, 0.f} {}

__host__ __device__ Configuration::Configuration(float j1, float j2, float j3)
    : joints{j1, j2, j3} {}

__host__ __device__ Robot::Robot() : base_{}, l1_{0.f}, l2_{0.f}, ee_{} {}

__host__ __device__ Robot::Robot(Pose<float> base, float l1, float l2,
                                 const Rectangle& ee)
    : base_{base}, l1_{l1}, l2_{l2}, ee_{ee} {}

__host__ __device__ const Rectangle& Robot::ee() const { return ee_; }

__host__ __device__ Pose<float> Robot::base() const { return base_; }

__host__ __device__ Pose<float> Robot::fk_elbow(
    const Configuration& conf) const {
  Transform<float> link(Translation<float>(l1_, 0), conf.joints[1]);
  return link.rotate(conf.joints[0]) * base();
}

__host__ __device__ Pose<float> Robot::fk_ee(const Configuration& conf) const {
  Transform<float> link(Translation<float>(l2_, 0), conf.joints[2]);
  return link * fk_elbow(conf);
}

DeviceRobot::DeviceRobot() : robot_{}, device_handle_{nullptr} {}

DeviceRobot::DeviceRobot(Pose<float> base, float l1, float l2,
                         const Rectangle& ee)
    : robot_{base, l1, l2, ee}, device_handle_{nullptr} {
  CHECK_CUDA(cudaMalloc(&device_handle_, sizeof(Robot)),
             "Could not allocate device memory for robot");
  CHECK_CUDA(cudaMemcpy(device_handle_, &robot_, sizeof(Robot),
                        cudaMemcpyHostToDevice),
             "Could not memcpy robot to device");
}

DeviceRobot::~DeviceRobot() {
  SAFE_CUDA_FREE(device_handle_, "Could not free device robot");
}

Robot& DeviceRobot::robot() { return robot_; }

Robot* DeviceRobot::device_handle() const { return device_handle_; }

Pose<float> DeviceRobot::base() const { return robot_.base(); }

Pose<float> DeviceRobot::fk_elbow(const Configuration& conf) const {
  return robot_.fk_elbow(conf);
}

Pose<float> DeviceRobot::fk_ee(const Configuration& conf) const {
  return robot_.fk_ee(conf);
}

}  // namespace gpu_planning
