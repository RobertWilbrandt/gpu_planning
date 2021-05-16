#include <cstring>

#include "array_2d.hpp"
#include "collision_checker.hpp"
#include "cuda_util.hpp"
#include "map.hpp"
#include "obstacle_manager.hpp"

namespace gpu_planning {

CollisionCheckResult::CollisionCheckResult() : result{false} {}

CollisionCheckResult::CollisionCheckResult(bool result, uint8_t obstacle_id)
    : result{result}, obstacle_id{obstacle_id} {}

CollisionChecker::CollisionChecker()
    : device_work_buf_{},
      mask_bufs_{},
      mask_buf_handles_{},
      map_{nullptr},
      robot_{nullptr},
      obstacle_manager_{nullptr},
      log_{nullptr} {}

CollisionChecker::CollisionChecker(DeviceMap* map, DeviceRobot* robot,
                                   ObstacleManager* obstacle_manager,
                                   Logger* log)
    : device_work_buf_{25},
      mask_bufs_{},
      mask_buf_handles_{device_work_buf_.block_size()},
      map_{map},
      robot_{robot},
      obstacle_manager_{obstacle_manager},
      log_{log} {
  const Box<float> ee_bb = robot->robot().ee().max_extent().bounding_box();
  const Translation<float> ee_diag(ee_bb.upper_right - ee_bb.lower_left);
  const Position<size_t> ee_bb_size =
      map->to_index(Position<float>() + ee_diag);

  std::vector<Map*> handles;
  for (int i = 0; i < device_work_buf_.block_size(); ++i) {
    mask_bufs_.emplace_back(ee_bb_size.x, ee_bb_size.y, map->resolution(), log);
    handles.push_back(mask_bufs_[i].device_map());
  }

  mask_buf_handles_.memcpy_set(handles);
}

__global__ void check_collisions(
    Map* map, Array<Map*>* mask_bufs, Robot* robot,
    WorkBlock<Configuration, CollisionCheckResult>* work) {
  extern __shared__ CollisionCheckResult thread_results[];

  for (int i = threadIdx.z; i < work->size(); i += blockDim.z) {
    const int thread_result_base = i * blockDim.x * blockDim.y;
    const int thread_result_idx =
        thread_result_base + threadIdx.y * blockDim.x + threadIdx.x;

    const Pose<float> ee = robot->fk_ee(work->data(i));
    const Box<size_t> map_index_area = map->data()->area();

    const Map& mask = *(*mask_bufs)[i];
    const Rectangle ee_shape = robot->ee();

    // Clear mask (can be used multiple times)
    for (int y = threadIdx.y; y < mask.data()->height(); y += blockDim.y) {
      for (int x = threadIdx.x; x < mask.data()->width(); x += blockDim.x) {
        mask.data()->at(x, y) = Cell(0.f, 0);
      }
    }

    // Insert shape into mask
    const Position<float> shape_offset(mask.width() / 2, mask.height() / 2);
    shape_insert_into(
        ee_shape, Pose<float>(shape_offset, ee.orientation), *mask.data(),
        mask.resolution(), Cell(1.f, 0),
        WorkLayout2d(threadIdx.x, blockDim.x, threadIdx.y, blockDim.y));

    // Check for collisions
    Cell result(0.f, 0);
    for (int y = threadIdx.y; y < mask.data()->height(); y += blockDim.y) {
      for (int x = threadIdx.x; x < mask.data()->width(); x += blockDim.x) {
        if (mask.data()->at(x, y).value >= 1.f) {
          const Position<float> global_pos =
              mask.from_index(Position<size_t>(x, y)) -
              shape_offset.from_origin() + ee.position.from_origin();

          const Position<size_t> map_pos = map->to_index(global_pos);
          if (map_index_area.is_inside(map_pos)) {
            const Cell& map_cell = map->data()->at(map_pos.x, map_pos.y);
            if (map_cell.value >= 1.f) {
              result = map_cell;
            }
          }
        }
      }
    }

    // Write thread result to shared buffer
    thread_results[thread_result_idx] =
        CollisionCheckResult(result.value >= 1.f, result.id);
  }

  // After all configuration results are known we need to reduce them
  __syncthreads();

  for (int i = threadIdx.z; i < work->size(); i += blockDim.z) {
    // TODO this could be a parallel reduction
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
      const int thread_result_base = i * blockDim.x * blockDim.y;

      CollisionCheckResult reduced_result;
      for (int y = 0; y < blockDim.y; ++y) {
        for (int x = 0; x < blockDim.x; ++x) {
          const CollisionCheckResult& result =
              thread_results[thread_result_base + y * blockDim.x + x];
          if (result.result) {
            reduced_result = result;
          }
        }
      }

      work->result(i) = reduced_result;
    }
  }
}

void CollisionChecker::check(const std::vector<Configuration>& configurations) {
  LOG_DEBUG(log_) << "Checking " << configurations.size()
                  << " configurations for collisions in blocks of "
                  << device_work_buf_.block_size();
  std::vector<CollisionCheckResult> result;
  result.resize(configurations.size());

  device_work_buf_.set_work(configurations.size(), configurations.data(),
                            result.data());

  while (!device_work_buf_.done()) {
    DeviceWorkHandle<Configuration, CollisionCheckResult> work =
        device_work_buf_.next_work_block();
    check_collisions<<<1, dim3(4, 16, 16),
                       device_work_buf_.block_size() * 4 * 16 *
                           sizeof(CollisionCheckResult)>>>(
        map_->device_map(), mask_buf_handles_.device_handle(),
        robot_->device_handle(), work.device_handle());
  }

  for (size_t i = 0; i < result.size(); ++i) {
    if (result[i].result) {
      const std::string obst_name =
          obstacle_manager_->get_obstacle_name(result[i].obstacle_id);
      LOG_DEBUG(log_) << "Configuration " << i << ": X   (" << obst_name << ")";
    } else {
      LOG_DEBUG(log_) << "Configuration " << i << ":   X";
    }
  }
}

}  // namespace gpu_planning
