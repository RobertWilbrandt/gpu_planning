#include <gtest/gtest.h>

#include <gpu_planning/work_buffer.hpp>

__global__ void kernel_increment(gpu_planning::WorkBlock<int, int>* work) {
  for (size_t i = threadIdx.x; i < work->size(); i += blockDim.x) {
    work->result(i) = work->data(i) + 1;
  }
}

TEST(WorkBuffer, LessThanBlocksize) {
  const size_t block_size = 10;
  gpu_planning::WorkBuffer<int, int> work_buffer(block_size);

  // Set up data and result buffer
  std::vector<int> data;
  for (size_t i = 0; i < 8; ++i) {
    data.push_back(i);
  }

  std::vector<int> result;
  result.resize(data.size());

  work_buffer.set_work(data.size(), data.data(), result.data());

  // Count number of transfers
  size_t num_iterations;
  for (num_iterations = 0; !work_buffer.done(); ++num_iterations) {
    gpu_planning::DeviceWorkHandle<int, int> work =
        work_buffer.next_work_block();
    kernel_increment<<<1, 1>>>(work.device_handle());
  }
  work_buffer.sync_result();

  EXPECT_EQ(1, num_iterations)
      << "Took too many iterations for case work < block_size";
  for (size_t i = 0; i < data.size(); ++i) {
    EXPECT_EQ(i + 1, result[i]) << "Wrong result at index " << i;
  }
}

__global__ void kernel_add_index(gpu_planning::WorkBlock<int, int>* work) {
  for (size_t i = threadIdx.x; i < work->size(); i += blockDim.x) {
    work->result(i) = work->data(i) + work->offset() + i;
  }
}

TEST(WorkBuffer, MoreThanOneBlock) {
  const size_t block_size = 5;

  gpu_planning::WorkBuffer<int, int> work_buffer(block_size);

  // Setup data and result
  std::vector<int> data;
  for (size_t i = 0; i < 8; ++i) {
    data.push_back(i);
  }

  std::vector<int> result;
  result.resize(data.size());

  work_buffer.set_work(data.size(), data.data(), result.data());

  // Count number of transfers
  size_t num_iterations;
  for (num_iterations = 0; !work_buffer.done(); ++num_iterations) {
    gpu_planning::DeviceWorkHandle<int, int> work =
        work_buffer.next_work_block();
    kernel_add_index<<<1, 1>>>(work.device_handle());
  }
  work_buffer.sync_result();

  EXPECT_EQ(2, num_iterations) << "Data should only need 2 transfers";
  for (size_t i = 0; i < result.size(); ++i) {
    size_t block = i / block_size;
    EXPECT_EQ(2 * i, result[i]) << "Wrong value in block " << block
                                << ", index " << (i - block_size * block);
  }
}

TEST(WorkBuffer, MultipleFullBlocks) {
  const size_t block_size = 4;
  gpu_planning::WorkBuffer<int, int> work_buffer(block_size);

  // Set up data and result buffer
  std::vector<int> data;
  for (size_t i = 0; i < 12; ++i) {
    data.push_back(i);
  }

  std::vector<int> result;
  result.resize(data.size());

  work_buffer.set_work(data.size(), data.data(), result.data());

  // Count number of transfers
  size_t num_iterations;
  for (num_iterations = 0; !work_buffer.done(); ++num_iterations) {
    gpu_planning::DeviceWorkHandle<int, int> work =
        work_buffer.next_work_block();
    kernel_increment<<<1, 16>>>(work.device_handle());
  }
  work_buffer.sync_result();

  EXPECT_EQ(3, num_iterations)
      << "Wrong number of transfers for 3 full data blocks";
  for (size_t i = 0; i < result.size(); ++i) {
    size_t block = i / block_size;
    EXPECT_EQ(i + 1, result[i]) << "Wrong value in block " << block
                                << ", index " << (i - block_size * block);
  }
}
