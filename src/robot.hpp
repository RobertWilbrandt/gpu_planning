#pragma once

namespace gpu_planning {

struct Configuration {
  Configuration();
  Configuration(float j1, float j2, float j3);

  float joints[3];
};

class Robot {
 public:
  Robot();

 private:
};

}  // namespace gpu_planning
