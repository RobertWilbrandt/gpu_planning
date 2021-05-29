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

DeviceCollisionChecker::DeviceCollisionChecker()
    : device_work_buf_{},
      mask_bufs_{},
      mask_buf_handles_{},
      map_{nullptr},
      robot_{nullptr},
      obstacle_manager_{nullptr},
      log_{nullptr} {}

DeviceCollisionChecker::DeviceCollisionChecker(
    DeviceMap* map, DeviceRobot* robot, ObstacleManager* obstacle_manager,
    Logger* log)
    : device_work_buf_{32},
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
  /*
   * Basic Idea:
   *   1. Go over configurations in blockDim.z blocks
   *   2. Get robot ee shapes for each and insert it into corresponding mask_buf
   *   3. Go over masks and check for collisions with map in
   *      blockDim.x*blockDim.y blocks, store each thread result in
   *      thread_results
   *   4. Reduce thread results for each configuration
   *
   * Implicit assumption:
   *   - blockDim.x and blockDim.y are powers of two (for efficient reduction)
   *   - work->size() is a multiple of blockDim.z (no special __syncthreads()
   *     considerations needed)
   *
   * thread_results stores one set of blockDim.x*blockDim.y results for each of
   * the blockDim.z configurations in one work block
   *   => size blockDim.z*blockDim.x*blockDim.y
   */
  extern __shared__ CollisionCheckResult thread_results[];

  int confs_done = 0;
  for (int i = threadIdx.z; i < work->size(); i += blockDim.z) {
    Array2d<CollisionCheckResult> thread_result(
        &thread_results[threadIdx.z * blockDim.x * blockDim.y], blockDim.x,
        blockDim.y);

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
    thread_result.at(threadIdx.x, threadIdx.y) =
        CollisionCheckResult(result.value >= 1.f, result.id);
    __syncthreads();

    // After all configuration results are known we need to reduce them
    int cur_width = blockDim.x;
    int cur_height = blockDim.y;

    while ((cur_width > 1) && (cur_height > 1)) {
      const int x_fact = cur_width > 1 ? 2 : 1;
      const int y_fact = cur_height > 1 ? 2 : 1;

      const int next_width = cur_width / x_fact;
      const int next_height = cur_height / y_fact;

      if ((threadIdx.x < next_width) && (threadIdx.y < next_height)) {
        for (int iy = 0; iy < y_fact; ++iy) {
          for (int ix = 0; ix < x_fact; ++ix) {
            const CollisionCheckResult& cur_result = thread_result.at(
                threadIdx.x + ix * next_width, threadIdx.y + iy * next_height);

            if (cur_result.result) {
              thread_result.at(threadIdx.x, threadIdx.y) = cur_result;
            }
          }
        }
      }

      __syncthreads();
      cur_width = next_width;
      cur_height = next_height;
    }

    confs_done += blockDim.z;
    work->result(i) = thread_result.at(0, 0);
  }

  // Sync threads that had no work in last configuration block
  if ((threadIdx.z >= (work->size() - confs_done)) &&
      (work->size() != confs_done)) {
    __syncthreads();
    for (int sync_count = max(blockDim.x, blockDim.y); sync_count > 1;
         sync_count /= 2) {
      __syncthreads();
    }
  }
}

void DeviceCollisionChecker::check(
    const std::vector<Configuration>& configurations) {
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
                       4 * 16 * 16 * sizeof(CollisionCheckResult)>>>(
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
