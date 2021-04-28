#pragma once

#include "geometry.hpp"

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
  Robot(Point base, float l1, float l2, float ee_w, float ee_h);

  ~Robot();

  DeviceRobot* device_robot() const;

  Point base() const;
  Point fk_elbow(const Configuration& conf) const;
  Point fk_ee(const Configuration& conf) const;

 private:
  DeviceRobot* device_robot_;

  Point base_;
  float l1_;
  float l2_;
  float ee_w_;
  float ee_h_;
};

}  // namespace gpu_planning
