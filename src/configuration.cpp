#include "configuration.hpp"

#include <boost/format.hpp>
#include <iomanip>

namespace gpu_planning {

Configuration::Configuration() {}

Configuration::Configuration(double j1, double j2, double j3) {
  joints[0] = j1;
  joints[1] = j2;
  joints[2] = j3;
}

std::ostream& operator<<(std::ostream& os, Configuration& conf) {
  return os << boost::format("(%1$5.2f, %2$5.2f, %3$5.2f)") % conf.joints[0] %
                   conf.joints[1] % conf.joints[2];
}

}  // namespace gpu_planning
