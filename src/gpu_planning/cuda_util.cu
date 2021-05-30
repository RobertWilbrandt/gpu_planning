#include "cuda_util.hpp"

namespace gpu_planning {

Stream::Stream() : stream{0} {}

Stream::Stream(cudaStream_t stream) : stream{stream} {}

Stream Stream::default_stream() { return Stream(0); }

Stream Stream::create() {
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream), "Could not create CUDA stream");
  return Stream(stream);
}

Stream::Stream(Stream&& other) noexcept : stream{other.stream} {
  other.stream = 0;
}

Stream& Stream::operator=(Stream&& other) noexcept {
  if (this != &other) {
    stream = other.stream;
    other.stream = 0;
  }

  return *this;
}

Stream::~Stream() {
  if (stream != 0) {
    CHECK_CUDA(cudaStreamDestroy(stream), "Could not destroy CUDA stream");
  }
}

void Stream::sync() const {
  CHECK_CUDA(cudaStreamSynchronize(stream), "Could not synchronize to stream");
}

}  // namespace gpu_planning
