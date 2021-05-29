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

__host__ __device__ CollisionChecker::CollisionChecker()
    : map_{nullptr}, robot_{nullptr}, mask_bufs_{nullptr} {}

__host__ __device__ CollisionChecker::CollisionChecker(Map* map, Robot* robot,
                                                       Array<Map*>* mask_bufs)
    : map_{map}, robot_{robot}, mask_bufs_{mask_bufs} {}

__device__ void CollisionChecker::check_configurations(
    WorkBlock<Configuration, CollisionCheckResult>& work, void* shared_buf,
    const WorkLayout3d& work_layout) {
  /*
   * Basic Idea:
   *   1. Go over configurations in work_layout.stride_z blocks
   *   2. Get robot ee shapes for each and insert it into corresponding
   *      mask_buf
   *   3. Go over masks and check for collisions with map in
   *      work_layout.stride_y*work_layout.stride_x blocks, store each thread
   *      result in shared_buf
   *   4. Reduce thread results for each configuration
   */
  CollisionCheckResult* result_buf = (CollisionCheckResult*)shared_buf;

  for (int i = work_layout.offset_z; i < work.size();
       i += work_layout.stride_z) {
    Array2d<CollisionCheckResult> thread_result(
        &result_buf[work_layout.offset_z * work_layout.stride_x *
                    work_layout.stride_y],
        work_layout.stride_x, work_layout.stride_y);

    const Pose<float> ee = robot_->fk_ee(work.data(i));
    const Box<size_t> map_index_area = map_->data()->area();

    const Map& mask = *(*mask_bufs_)[i];
    const Rectangle ee_shape = robot_->ee();

    // Clear mask (can be used multiple times)
    for (int y = work_layout.offset_y; y < mask.data()->height();
         y += work_layout.stride_y) {
      for (int x = work_layout.offset_x; x < mask.data()->width();
           x += work_layout.stride_x) {
        mask.data()->at(x, y) = Cell(0.f, 0);
      }
    }

    // Insert shape into mask
    const Position<float> shape_offset(mask.width() / 2, mask.height() / 2);
    shape_insert_into(ee_shape, Pose<float>(shape_offset, ee.orientation),
                      *mask.data(), mask.resolution(), Cell(1.f, 0),
                      WorkLayout2d(work_layout.offset_x, work_layout.stride_x,
                                   work_layout.offset_y, work_layout.stride_y));

    // Check for collisions
    Cell result(0.f, 0);
    for (int y = work_layout.offset_y; y < mask.data()->height();
         y += work_layout.stride_y) {
      for (int x = work_layout.offset_x; x < mask.data()->width();
           x += work_layout.stride_x) {
        if (mask.data()->at(x, y).value >= 1.f) {
          const Position<float> global_pos =
              mask.from_index(Position<size_t>(x, y)) -
              shape_offset.from_origin() + ee.position.from_origin();

          const Position<size_t> map_pos = map_->to_index(global_pos);
          if (map_index_area.is_inside(map_pos)) {
            const Cell& map_cell = map_->data()->at(map_pos.x, map_pos.y);
            if (map_cell.value >= 1.f) {
              result = map_cell;
            }
          }
        }
      }
    }

    // Write thread result to shared buffer
    thread_result.at(work_layout.offset_x, work_layout.offset_y) =
        CollisionCheckResult(result.value >= 1.f, result.id);

    // These syncs are fine as work.size() is required to be a multiple of
    // work_layout.stride_z
    __syncthreads();

    // After all configuration results are known we need to reduce them
    int cur_width = work_layout.stride_x;
    int cur_height = work_layout.stride_y;

    while ((cur_width > 1) && (cur_height > 1)) {
      const int x_fact = cur_width > 1 ? 2 : 1;
      const int y_fact = cur_height > 1 ? 2 : 1;

      const int next_width = cur_width / x_fact;
      const int next_height = cur_height / y_fact;

      if ((work_layout.offset_x < next_width) &&
          (work_layout.offset_y < next_height)) {
        for (int iy = 0; iy < y_fact; ++iy) {
          for (int ix = 0; ix < x_fact; ++ix) {
            const CollisionCheckResult& cur_result =
                thread_result.at(work_layout.offset_x + ix * next_width,
                                 work_layout.offset_y + iy * next_height);

            if (cur_result.result) {
              thread_result.at(work_layout.offset_x, work_layout.offset_y) =
                  cur_result;
            }
          }
        }
      }

      __syncthreads();
      cur_width = next_width;
      cur_height = next_height;
    }

    work.result(i) = thread_result.at(0, 0);
  }
}

DeviceCollisionChecker::DeviceCollisionChecker()
    : collision_checker_{nullptr},
      device_work_buf_{},
      mask_bufs_{},
      mask_buf_handles_{},
      map_{nullptr},
      robot_{nullptr},
      log_{nullptr} {}

DeviceCollisionChecker::DeviceCollisionChecker(DeviceMap* map,
                                               DeviceRobot* robot, Logger* log)
    : collision_checker_{},
      device_work_buf_{32},
      mask_bufs_{},
      mask_buf_handles_{device_work_buf_.block_size()},
      map_{map},
      robot_{robot},
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

  CollisionChecker collision_checker_host(map_->device_map(),
                                          robot_->device_handle(),
                                          mask_buf_handles_.device_handle());
  collision_checker_.memcpy_set(&collision_checker_host);
}

__global__ void check_collisions(
    CollisionChecker* collision_checker,
    WorkBlock<Configuration, CollisionCheckResult>* work) {
  extern __shared__ CollisionCheckResult thread_results[];

  // This enforces work.size() % work_layout.stride_z == 0 and is safe because
  // we use a multiple of blockDim.z as WorkBuffer.block_size()
  size_t aligned_size = (1 + (work->size() - 1) / blockDim.z) * blockDim.z;
  WorkBlock<Configuration, CollisionCheckResult> aligned_work(
      aligned_size, &work->data(0), &work->result(0), work->offset());

  collision_checker->check_configurations(
      aligned_work, thread_results, WorkLayout3d::from(threadIdx, blockDim));
}

std::vector<CollisionCheckResult> DeviceCollisionChecker::check(
    const std::vector<Configuration>& configurations, cudaStream_t stream,
    bool async) {
  LOG_DEBUG(log_) << "Checking " << configurations.size()
                  << " configurations for collisions in blocks of "
                  << device_work_buf_.block_size();
  std::vector<CollisionCheckResult> result;
  result.resize(configurations.size());

  device_work_buf_.set_work(configurations.size(), configurations.data(),
                            result.data(), stream);

  while (!device_work_buf_.done()) {
    DeviceWorkHandle<Configuration, CollisionCheckResult> work =
        device_work_buf_.next_work_block();

    // Be sure that device_work_buf_.block_size() is a multiple of blockDim.z
    check_collisions<<<1, dim3(4, 16, 16),
                       4 * 16 * 16 * sizeof(CollisionCheckResult), stream>>>(
        collision_checker_.device_handle(), work.device_handle());
  }

  if (!async) {
    device_work_buf_.sync_result();
  }

  return result;
}

std::vector<CollisionCheckResult> DeviceCollisionChecker::check(
    const std::vector<TrajectorySegment>& segments, cudaStream_t stream,
    bool async) {
  std::vector<CollisionCheckResult> result;

  std::vector<Configuration> configurations;
  for (size_t i = 0; i < segments.size(); ++i) {
    configurations.push_back(segments[i].start);
    configurations.push_back(segments[i].interpolate(0.5f));
    configurations.push_back(segments[i].end);
  }

  std::vector<CollisionCheckResult> conf_result =
      check(configurations, stream, async);

  for (size_t i = 0; i < segments.size(); ++i) {
    CollisionCheckResult seg_result;
    for (size_t j = 0; j < 3; ++j) {
      if (conf_result[i * 3 + j].result) {
        seg_result = conf_result[i * 3 + j];
      }
    }

    result.push_back(seg_result);
  }

  return result;
}

}  // namespace gpu_planning
