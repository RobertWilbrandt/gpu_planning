#pragma once

#include <array>
#include <ostream>

namespace gpu_planning {

struct Configuration {
  Configuration();
  Configuration(double j1, double j2, double j3);

  std::array<float, 3> joints;
};

std::ostream& operator<<(std::ostream& os, Configuration& conf);

}  // namespace gpu_planning
