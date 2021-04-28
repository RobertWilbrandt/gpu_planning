#pragma once

namespace gpu_planning {

class DeviceRobot {
 public:
  __host__ DeviceRobot();
  __host__ DeviceRobot(float l1, float l2, float ee_w, float ee_h);

 private:
  float l1_;
  float l2_;
  float ee_w_;
  float ee_h_;
};

}  // namespace gpu_planning
