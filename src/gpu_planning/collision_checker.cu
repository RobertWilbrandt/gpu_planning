#include <cstring>
#include <gpu_planning_tracepoints/tracepoints.hpp>

#include "array_2d.hpp"
#include "collision_checker.hpp"
#include "cuda_util.hpp"
#include "map.hpp"
#include "obstacle_manager.hpp"

namespace gpu_planning {

__host__ __device__ CollisionCheckResult<Configuration>::CollisionCheckResult()
    : result{false}, obstacle_id{0} {}

__host__ __device__ CollisionCheckResult<Configuration>::CollisionCheckResult(
    bool result, uint8_t obstacle_id)
    : result{result}, obstacle_id{obstacle_id} {}

__host__ __device__
CollisionCheckResult<TrajectorySegment>::CollisionCheckResult()
    : result{false}, obstacle_id{0} {}

__host__ __device__
CollisionCheckResult<TrajectorySegment>::CollisionCheckResult(
    bool result, uint8_t obstacle_id)
    : result{result}, obstacle_id{obstacle_id} {}

__host__ __device__ CollisionChecker::CollisionChecker()
    : map_{nullptr}, robot_{nullptr}, mask_bufs_{nullptr} {}

__host__ __device__ CollisionChecker::CollisionChecker(Map* map, Robot* robot,
                                                       Array<Map*>* mask_bufs)
    : map_{map}, robot_{robot}, mask_bufs_{mask_bufs} {}

__device__ void CollisionChecker::check_configurations(
    WorkBlock<Configuration, Result<Configuration>>& work, void* shared_buf,
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
  Result<Configuration>* result_buf = (Result<Configuration>*)shared_buf;

  for (int i = thread_block.z(); i < work.size(); i += thread_block.dim_z()) {
    Array2d<Result<Configuration>> thread_result(
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
                      thread_block.slice_z());

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
        Result<Configuration>(result.value >= 1.f, result.id);

    // These syncs are fine as work.size() is required to be a multiple of
    // thread_block.dim_z()
    thread_block.sync();

    struct ResultReducer {
      static __host__ __device__ void reduce(Result<Configuration>& r1,
                                             const Result<Configuration>& r2) {
        if (r2.result) {
          r1 = r2;
        }
      }
    };

    work.result(i) =
        thread_result.reduce<ResultReducer>(thread_block.slice_z());
  }
}

DeviceCollisionChecker::DeviceCollisionChecker()
    : collision_checker_{nullptr},
      device_conf_work_buf_{},
      device_seg_work_buf_{},
      mask_bufs_{},
      mask_buf_handles_{},
      map_{nullptr},
      robot_{nullptr},
      log_{nullptr} {}

DeviceCollisionChecker::DeviceCollisionChecker(DeviceMap* map,
                                               DeviceRobot* robot, Logger* log)
    : collision_checker_{},
      device_conf_work_buf_{32},
      device_seg_work_buf_{16},
      mask_bufs_{},
      mask_buf_handles_{device_conf_work_buf_.block_size()},
      map_{map},
      robot_{robot},
      log_{log} {
  const Box<float> ee_bb = robot->robot().ee().max_extent().bounding_box();
  const Translation<float> ee_diag(ee_bb.upper_right - ee_bb.lower_left);
  const Position<size_t> ee_bb_size =
      map->to_index(Position<float>() + ee_diag);

  std::vector<Map*> handles;
  for (int i = 0; i < device_conf_work_buf_.block_size(); ++i) {
    mask_bufs_.emplace_back(ee_bb_size.x, ee_bb_size.y, map->resolution(), log);
    handles.push_back(mask_bufs_[i].device_map());
  }

  mask_buf_handles_.memcpy_set(handles);

  CollisionChecker collision_checker_host(map_->device_map(),
                                          robot_->device_handle(),
                                          mask_buf_handles_.device_handle());
  collision_checker_.memcpy_set(&collision_checker_host);
}

__global__ void collision_checker_check_conf_collision(
    CollisionChecker* collision_checker,
    WorkBlock<Configuration, CollisionChecker::Result<Configuration>>* work) {
  extern __shared__ CollisionChecker::Result<Configuration> thread_results[];

  // This enforces work.size() % thread_block.dim_z() == 0 and is safe because
  // we use a multiple of blockDim.z as WorkBuffer.block_size()
  size_t aligned_size = (1 + (work->size() - 1) / blockDim.z) * blockDim.z;
  WorkBlock<Configuration, CollisionChecker::Result<Configuration>>
      aligned_work(aligned_size, &work->data(0), &work->result(0),
                   work->offset());

  collision_checker->check_configurations(aligned_work, thread_results,
                                          ThreadBlock3d::device_current());
}

std::vector<CollisionChecker::Result<Configuration>>
DeviceCollisionChecker::check_async(
    const std::vector<Configuration>& configurations, const Stream& stream) {
  tracepoint(gpu_planning, collision_check_configuration,
             configurations.size());

  std::vector<CollisionChecker::Result<Configuration>> result;
  result.resize(configurations.size());

  device_conf_work_buf_.set_work(configurations.size(), configurations.data(),
                                 result.data(), &stream);

  while (!device_conf_work_buf_.done()) {
    DeviceWorkHandle<Configuration, CollisionChecker::Result<Configuration>>
        work = device_conf_work_buf_.next_work_block();

    // Be sure that device_conf_work_buf_.block_size() is a multiple of
    // blockDim.z
    collision_checker_check_conf_collision<<<
        1, dim3(8, 8, 16),
        8 * 8 * 16 * sizeof(CollisionChecker::Result<Configuration>),
        stream.stream>>>(collision_checker_.device_handle(),
                         work.device_handle());
  }

  return result;
}

__global__ void collision_checker_check_seg_collision(
    CollisionChecker* collision_checker,
    WorkBlock<TrajectorySegment, CollisionChecker::Result<TrajectorySegment>>*
        segments,
    WorkBlock<Configuration, CollisionChecker::Result<Configuration>>*
        conf_work) {
  extern __shared__ CollisionChecker::Result<Configuration> thread_results[];

  // Create configurations from segments
  // TODO make sure to not overflow conf_work
  for (int i = threadIdx.z; i < segments->size(); i += blockDim.z) {
    const TrajectorySegment& segment = segments->data(i);

    conf_work->data(3 * i) = segment.start;
    conf_work->data(3 * i + 1) = segment.interpolate(0.5f);
    conf_work->data(3 * i + 2) = segment.end;
  }

  // Test configurations
  const size_t conf_size = segments->size() * 3;
  const size_t aligned_size = (1 + (conf_size - 1) / blockDim.z) * blockDim.z;
  WorkBlock<Configuration, CollisionChecker::Result<Configuration>>
      aligned_conf_work(aligned_size, &conf_work->data(0),
                        &conf_work->result(0), 0);

  collision_checker->check_configurations(aligned_conf_work, thread_results,
                                          ThreadBlock3d::device_current());

  // Read results
  for (int i = threadIdx.z; i < segments->size(); i += blockDim.z) {
    CollisionChecker::Result<TrajectorySegment> seg_result(false, 0);
    for (int j = 0; j < 3; ++j) {
      const CollisionChecker::Result<Configuration>& conf_result =
          conf_work->result(3 * i + j);

      if (conf_result.result) {
        seg_result = CollisionChecker::Result<TrajectorySegment>(
            conf_result.result, conf_result.obstacle_id);
      }
    }

    segments->result(i) = seg_result;
  }
}

std::vector<CollisionChecker::Result<TrajectorySegment>>
DeviceCollisionChecker::check_async(
    const std::vector<TrajectorySegment>& segments, const Stream& stream) {
  tracepoint(gpu_planning, collision_check_segment, segments.size());

  std::vector<CollisionChecker::Result<TrajectorySegment>> result;
  result.resize(segments.size());

  device_seg_work_buf_.set_work(segments.size(), segments.data(), result.data(),
                                &stream);

  while (!device_seg_work_buf_.done()) {
    DeviceWorkHandle<TrajectorySegment,
                     CollisionChecker::Result<TrajectorySegment>>
        work = device_seg_work_buf_.next_work_block();

    // Be sure that device_seg_work_buf_.block_size() is a multiple of
    // blockDim.z
    collision_checker_check_seg_collision<<<
        1, dim3(8, 8, 16),
        8 * 8 * 16 * sizeof(CollisionChecker::Result<Configuration>),
        stream.stream>>>(collision_checker_.device_handle(),
                         work.device_handle(),
                         device_conf_work_buf_.device_full_block());
  }

  return result;
}

}  // namespace gpu_planning
