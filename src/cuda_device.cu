#include "cuda_device.hpp"

void check_cuda(cudaError_t err) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") +
                             cudaGetErrorString(err));
  }
}

namespace gpu_planning {

void cuda_list_devices(logger& log) {
  int device_count;
  check_cuda(cudaGetDeviceCount(&device_count));

  LOG_INFO(log) << "Found " << device_count << " CUDA devices:";
  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp prop;
    check_cuda(cudaGetDeviceProperties(&prop, i));

    LOG_INFO(log) << "- " << i << ": " << prop.name << ", PCI "
                  << prop.pciDomainID << ':' << prop.pciBusID << ':'
                  << prop.pciDeviceID << ", SM " << prop.major << '.'
                  << prop.minor;
  }
}

}  // namespace gpu_planning
