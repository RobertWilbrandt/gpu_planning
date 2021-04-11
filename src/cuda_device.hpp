#pragma once

#include "logging.hpp"

namespace gpu_planning {

void cuda_set_device(int dev, logger& log);
void cuda_list_devices(logger& log);

}  // namespace gpu_planning
