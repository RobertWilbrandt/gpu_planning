#pragma once

#include <memory>

namespace gpu_planning {

#if !defined(__CUDACC__)
struct cudaPitchedPtr;
#endif

struct Cell {
  float occupancy;
};

class Map {
 public:
  Map();
  Map(size_t width, size_t height);

  ~Map();

 private:
  std::unique_ptr<cudaPitchedPtr> map_;
};

}  // namespace gpu_planning
