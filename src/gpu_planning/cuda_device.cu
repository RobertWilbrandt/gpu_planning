#include "cuda_device.hpp"
#include "cuda_util.hpp"

namespace gpu_planning {

void cuda_set_device(int dev, Logger* log) {
  CHECK_CUDA(cudaSetDevice(dev), "Could not set CUDA device");
  LOG_INFO(log) << "Using CUDA device " << dev;
}

void cuda_list_devices(Logger* log) {
  int device_count;
  CHECK_CUDA(cudaGetDeviceCount(&device_count),
             "Could not get CUDA device count");

  int used_device;
  CHECK_CUDA(cudaGetDevice(&used_device), "Could not get used CUDA device");

  LOG_INFO(log) << "Found " << device_count << " CUDA devices:";
  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, i),
               "Could not get CUDA device properties");

    std::string selected_string;
    if (i == used_device) {
      selected_string = "*";
    } else {
      selected_string = " ";
    }

    LOG_INFO(log) << "- " << selected_string << i << ": " << prop.name
                  << ", PCI " << prop.pciDomainID << ':' << prop.pciBusID << ':'
                  << prop.pciDeviceID << ", SM " << prop.major << '.'
                  << prop.minor;
  }
}

}  // namespace gpu_planning
