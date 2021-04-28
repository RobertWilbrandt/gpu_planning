#include "robot.hpp"

namespace gpu_planning {

Configuration::Configuration() : joints{0.f, 0.f, 0.f} {}

Configuration::Configuration(float j1, float j2, float j3)
    : joints{j1, j2, j3} {}

Robot::Robot() {}

}  // namespace gpu_planning
