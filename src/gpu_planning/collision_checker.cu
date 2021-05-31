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
    const ThreadBlock3d& thread_block) {
  /*
   * Basic Idea:
   *   1. Go over configurations in thread_block.dim_z() blocks
   *   2. Get robot ee shapes for each and insert it into corresponding
   *      mask_buf
   *   3. Go over masks and check for collisions with map in
   *      thread_block.dim_y()*thread_block.dim_x() blocks, store each thread
   *      result in shared_buf
   *   4. Reduce thread results for each configuration
   */
  CollisionCheckResult* result_buf = (CollisionCheckResult*)shared_buf;

  for (int i = thread_block.z(); i < work.size(); i += thread_block.dim_z()) {
    Array2d<CollisionCheckResult> thread_result(
        &result_buf[thread_block.z() * thread_block.dim_x() *
                    thread_block.dim_y()],
        thread_block.dim_x(), thread_block.dim_y());

    const Pose<float> ee = robot_->fk_ee(work.data(i));
    const Box<size_t> map_index_area = map_->data()->area();

    const Map& mask = *(*mask_bufs_)[i];
    const Rectangle ee_shape = robot_->ee();

    // Clear mask (can be used multiple times)
    for (int y = thread_block.y(); y < mask.data()->height();
         y += thread_block.dim_y()) {
      for (int x = thread_block.x(); x < mask.data()->width();
           x += thread_block.dim_x()) {
        mask.data()->at(x, y) = Cell(0.f, 0);
      }
    }

    // Insert shape into mask
    const Position<float> shape_offset(mask.width() / 2, mask.height() / 2);
    shape_insert_into(ee_shape, Pose<float>(shape_offset, ee.orientation),
                      *mask.data(), mask.resolution(), Cell(1.f, 0),
                      WorkLayout2d(thread_block.x(), thread_block.dim_x(),
                                   thread_block.y(), thread_block.dim_y()));

    // Check for collisions
    Cell result(0.f, 0);
    for (int y = thread_block.y(); y < mask.data()->height();
         y += thread_block.dim_y()) {
      for (int x = thread_block.x(); x < mask.data()->width();
           x += thread_block.dim_x()) {
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
    thread_result.at(thread_block.x(), thread_block.y()) =
        CollisionCheckResult(result.value >= 1.f, result.id);

    // These syncs are fine as work.size() is required to be a multiple of
    // thread_block.dim_z()
    thread_block.sync();

    // After all configuration results are known we need to reduce them
    int cur_width = thread_block.dim_x();
    int cur_height = thread_block.dim_y();

    while ((cur_width > 1) && (cur_height > 1)) {
      const int x_fact = cur_width > 1 ? 2 : 1;
      const int y_fact = cur_height > 1 ? 2 : 1;

      const int next_width = cur_width / x_fact;
      const int next_height = cur_height / y_fact;

      if ((thread_block.x() < next_width) && (thread_block.y() < next_height)) {
        for (int iy = 0; iy < y_fact; ++iy) {
          for (int ix = 0; ix < x_fact; ++ix) {
            const CollisionCheckResult& cur_result =
                thread_result.at(thread_block.x() + ix * next_width,
                                 thread_block.y() + iy * next_height);

            if (cur_result.result) {
              thread_result.at(thread_block.x(), thread_block.y()) = cur_result;
            }
          }
        }
      }

      thread_block.sync();
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

  // This enforces work.size() % thread_block.dim_z() == 0 and is safe because
  // we use a multiple of blockDim.z as WorkBuffer.block_size()
  size_t aligned_size = (1 + (work->size() - 1) / blockDim.z) * blockDim.z;
  WorkBlock<Configuration, CollisionCheckResult> aligned_work(
      aligned_size, &work->data(0), &work->result(0), work->offset());

  collision_checker->check_configurations(aligned_work, thread_results,
                                          ThreadBlock3d::device_current());
}

std::vector<CollisionCheckResult> DeviceCollisionChecker::check(
    const std::vector<Configuration>& configurations, const Stream& stream,
    bool async) {
  LOG_DEBUG(log_) << "Checking " << configurations.size()
                  << " configurations for collisions in blocks of "
                  << device_work_buf_.block_size();
  std::vector<CollisionCheckResult> result;
  result.resize(configurations.size());

  device_work_buf_.set_work(configurations.size(), configurations.data(),
                            result.data(), &stream);

  while (!device_work_buf_.done()) {
    DeviceWorkHandle<Configuration, CollisionCheckResult> work =
        device_work_buf_.next_work_block();

    // Be sure that device_work_buf_.block_size() is a multiple of blockDim.z
    check_collisions<<<1, dim3(4, 16, 16),
                       4 * 16 * 16 * sizeof(CollisionCheckResult),
                       stream.stream>>>(collision_checker_.device_handle(),
                                        work.device_handle());
  }

  if (!async) {
    device_work_buf_.sync_result();
  }

  return result;
}

std::vector<CollisionCheckResult> DeviceCollisionChecker::check(
    const std::vector<TrajectorySegment>& segments, const Stream& stream,
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
