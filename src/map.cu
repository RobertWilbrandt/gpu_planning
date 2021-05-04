#include <algorithm>
#include <string>

#include "cuda_util.hpp"
#include "map.hpp"

namespace gpu_planning {

Map::Map() : data_{nullptr}, resolution_{0} {}

Map::Map(Array2d<float>* data, size_t resolution)
    : data_{data}, resolution_{resolution} {}

__device__ float Map::width() const {
  return (float)data_->width() / resolution_;
}

__device__ float Map::height() const {
  return (float)data_->height() / resolution_;
}

__device__ size_t Map::resolution() const { return resolution_; }

__device__ Array2d<float>* Map::data() const { return data_; }

__device__ Position<size_t> Map::to_index(
    const Position<float>& position) const {
  return Position<size_t>(position.x * resolution_, position.y * resolution_);
}

__device__ Pose<size_t> Map::to_index(const Pose<float>& pose) const {
  return Pose<size_t>(to_index(pose.position), pose.orientation);
}

__device__ Position<float> Map::from_index(
    const Position<size_t>& index) const {
  return Position<float>(static_cast<float>(index.x) / resolution_,
                         static_cast<float>(index.y) / resolution_);
}

__device__ Pose<float> Map::from_index(const Pose<size_t>& index) const {
  return Pose<float>(from_index(index.position), index.orientation);
}

__device__ float Map::get(const Position<float>& position) {
  return data_->at(to_index(position));
}

DeviceMap::DeviceMap() : map_{nullptr}, data_{}, resolution_{}, log_{nullptr} {}

DeviceMap::DeviceMap(float width, float height, size_t resolution, Logger* log)
    : map_{nullptr},
      data_{static_cast<size_t>(width * resolution),
            static_cast<size_t>(height * resolution)},
      resolution_{resolution},
      log_{log} {
  CHECK_CUDA(cudaMalloc(&map_, sizeof(Map)), "Could not allocate device map");
  Map map(data_.device_handle(), resolution_);
  CHECK_CUDA(cudaMemcpy(map_, &map, sizeof(Map), cudaMemcpyHostToDevice),
             "Could not memcpy device map to device");

  data_.memset(0);

  LOG_DEBUG(log_) << "Created map of size " << width << "x" << height
                  << " and resolution " << resolution_;
}

DeviceMap::~DeviceMap() { SAFE_CUDA_FREE(map_, "Could not free device map"); }

float DeviceMap::width() const { return (float)data_.width() / resolution_; }

float DeviceMap::height() const { return (float)data_.height() / resolution_; }

size_t DeviceMap::resolution() const { return resolution_; }

Map* DeviceMap::device_map() const { return map_; }

__global__ void device_consolidate_data(Array2d<float>* map,
                                        Array2d<float>* dest) {
  const size_t x_fact = map->width() / dest->width();
  const size_t y_fact = map->height() / dest->height();

  for (size_t j = threadIdx.y; j < dest->height(); j += blockDim.y) {
    for (size_t i = threadIdx.x; i < dest->width(); i += blockDim.x) {
      float sum = 0.f;

      for (size_t cy = 0; cy < y_fact; ++cy) {
        for (size_t cx = 0; cx < x_fact; ++cx) {
          const size_t x = i * x_fact + cx;
          const size_t y = j * y_fact + cy;

          sum += map->at(x, y);
        }
      }

      dest->at(i, j) = sum / (x_fact * y_fact);
    }
  }
}

void DeviceMap::get_data(float* dest, size_t max_width, size_t max_height,
                         size_t* result_width, size_t* result_height) {
  const size_t map_width = data_.width();
  const size_t map_height = data_.height();

  const size_t x_fact = map_width / (max_width + 1) + 1;
  const size_t y_fact = map_height / (max_height + 1) + 1;

  const size_t sub_width = map_width / x_fact;
  const size_t sub_height = map_height / y_fact;

  DeviceArray2d<float> sub(sub_width, sub_height);
  device_consolidate_data<<<1, dim3(32, 32)>>>(data_.device_handle(),
                                               sub.device_handle());
  Array2d<float> dest_array(dest, sub_width, sub_height,
                            sub_width * sizeof(float));
  sub.memcpy_get(dest_array);

  *result_width = sub_width;
  *result_height = sub_height;
}

__global__ void device_add_obstacle_circle(Map* map, float cx, float cy,
                                           float crad) {
  const size_t resolution = map->resolution();
  Array2d<float>* map_data = map->data();

  size_t first_row = max((int)((cy - crad) * resolution), 0);
  size_t last_row =
      min((int)((cy + crad) * resolution), (int)map_data->width());
  size_t first_col = max((int)((cx - crad) * resolution), 0);
  size_t last_col =
      min((int)((cx + crad) * resolution), (int)map_data->height());

  for (size_t j = threadIdx.x; j < (last_row - first_row); j += blockDim.x) {
    for (size_t i = threadIdx.y; i < (last_col - first_col); i += blockDim.y) {
      size_t x = first_col + i;
      size_t y = first_row + j;

      double dx = (float)x / resolution - cx;
      double dy = (float)y / resolution - cy;

      if (dx * dx + dy * dy < crad * crad) {
        map_data->at(x, y) = 1.0;
      }
    }
  }
}

void DeviceMap::add_obstacle_circle(float x, float y, float radius) {
  device_add_obstacle_circle<<<1, dim3(32, 32)>>>(map_, x, y, radius);
}

__global__ void device_add_obstacle_rect(Map* map, float cx, float cy,
                                         float width, float height) {
  const size_t resolution = map->resolution();
  Array2d<float>* map_data = map->data();

  const size_t first_col = max((int)((cx - 0.5 * width) * resolution), 0);
  const size_t last_col =
      min((int)((cx + 0.5 * width) * resolution), (int)map_data->width());
  const size_t first_row = max((int)((cy - 0.5 * height) * resolution), 0);
  const size_t last_row =
      min((int)((cy + 0.5 * height) * resolution), (int)map_data->height());

  for (size_t y = first_row + threadIdx.y; y < last_row; y += blockDim.y) {
    for (size_t x = first_col + threadIdx.x; x < last_col; x += blockDim.x) {
      map_data->at(x, y) = 1.0;
    }
  }
}

void DeviceMap::add_obstacle_rect(float x, float y, float width, float height) {
  device_add_obstacle_rect<<<1, dim3(32, 32)>>>(map_, x, y, width, height);
}

}  // namespace gpu_planning
