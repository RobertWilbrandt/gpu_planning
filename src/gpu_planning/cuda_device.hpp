#pragma once

#include "logging.hpp"

namespace gpu_planning {

void cuda_set_device(int dev, Logger* log);
void cuda_list_devices(Logger* log);

}  // namespace gpu_planning
