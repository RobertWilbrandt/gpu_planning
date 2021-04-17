#pragma once

#include <string>

#include "logging.hpp"

namespace gpu_planning {

void save_bmp(float* data, size_t width, size_t height, const std::string& path,
              Logger* log);

}
