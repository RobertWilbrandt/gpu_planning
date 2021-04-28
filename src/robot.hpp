#pragma once

namespace gpu_planning {

struct Configuration {
  Configuration();
  Configuration(float j1, float j2, float j3);

  float joints[3];
};

class DeviceRobot;

class Robot {
 public:
  Robot();
  Robot(float bx, float by, float l1, float l2, float ee_w, float ee_h);

  ~Robot();

  DeviceRobot* device_robot() const;

 private:
  DeviceRobot* device_robot_;

  float bx_;
  float by_;
  float l1_;
  float l2_;
  float ee_w_;
  float ee_h_;
};

}  // namespace gpu_planning
