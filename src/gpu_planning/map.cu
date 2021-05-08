#include <algorithm>
#include <string>

#include "cuda_util.hpp"
#include "map.hpp"

namespace gpu_planning {

__host__ __device__ Cell::Cell() : value{0.f}, id{0} {}

__host__ __device__ Cell::Cell(float value, uint8_t id)
    : value{value}, id{id} {}

__host__ __device__ Map::Map() : data_{nullptr}, resolution_{0} {}

__host__ __device__ Map::Map(Array2d<Cell>* data, size_t resolution)
    : data_{data}, resolution_{resolution} {}

__host__ __device__ float Map::width() const {
  return (float)data_->width() / resolution_;
}

__host__ __device__ float Map::height() const {
  return (float)data_->height() / resolution_;
}

__host__ __device__ size_t Map::resolution() const { return resolution_; }

__host__ __device__ Array2d<Cell>* Map::data() const { return data_; }

__host__ __device__ Position<size_t> Map::to_index(
    const Position<float>& position) const {
  return Position<size_t>(position.x * resolution_, position.y * resolution_);
}

__host__ __device__ Pose<size_t> Map::to_index(const Pose<float>& pose) const {
  return Pose<size_t>(to_index(pose.position), pose.orientation);
}

__host__ __device__ Position<float> Map::from_index(
    const Position<size_t>& index) const {
  return Position<float>(static_cast<float>(index.x) / resolution_,
                         static_cast<float>(index.y) / resolution_);
}

__host__ __device__ Pose<float> Map::from_index(
    const Pose<size_t>& index) const {
  return Pose<float>(from_index(index.position), index.orientation);
}

__host__ __device__ const Cell& Map::get(const Position<float>& position) {
  return data_->at(to_index(position));
}

HostMap::HostMap() : map_storage_{}, map_array_{}, map_{}, log_{nullptr} {}

HostMap::HostMap(float width, float height, size_t resolution, Logger* log)
    : map_storage_{static_cast<size_t>(width * resolution) *
                   static_cast<size_t>(height * resolution)},
      map_array_{map_storage_.data(), static_cast<size_t>(width * resolution),
                 static_cast<size_t>(height * resolution),
                 static_cast<size_t>(width * resolution) * sizeof(Cell)},
      map_{&map_array_, resolution},
      log_{log} {}

Map& HostMap::map() { return map_; }

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

size_t DeviceMap::index_width() const { return data_.width(); }

size_t DeviceMap::index_height() const { return data_.height(); }

size_t DeviceMap::resolution() const { return resolution_; }

Map* DeviceMap::device_map() const { return map_; }

Position<size_t> DeviceMap::to_index(const Position<float>& position) const {
  return Position<size_t>(position.x * resolution_, position.y * resolution_);
}

Pose<size_t> DeviceMap::to_index(const Pose<float>& pose) const {
  return Pose<size_t>(to_index(pose.position), pose.orientation);
}

Position<float> DeviceMap::from_index(const Position<size_t>& index) const {
  return Position<float>(static_cast<float>(index.x) / resolution_,
                         static_cast<float>(index.y) / resolution_);
}

Pose<float> DeviceMap::from_index(const Pose<size_t>& index) const {
  return Pose<float>(from_index(index.position), index.orientation);
}

HostMap DeviceMap::load_to_host() {
  HostMap result(width(), height(), resolution_, log_);
  data_.memcpy_get(*result.map().data());

  return result;
}

__global__ void device_consolidate_data(Array2d<Cell>* map,
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

          sum += map->at(x, y).value;
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
  const size_t fact = max(x_fact, y_fact);

  const size_t sub_width = map_width / fact;
  const size_t sub_height = map_height / fact;

  DeviceArray2d<float> sub(sub_width, sub_height);
  device_consolidate_data<<<1, dim3(32, 32)>>>(data_.device_handle(),
                                               sub.device_handle());
  Array2d<float> dest_array(dest, sub_width, sub_height,
                            sub_width * sizeof(float));
  sub.memcpy_get(dest_array);

  *result_width = sub_width;
  *result_height = sub_height;
}

}  // namespace gpu_planning
