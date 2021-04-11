#include <iostream>

#include "cli.hpp"
#include "cuda_device.hpp"
#include "logging.hpp"
#include "map.hpp"

#define TEST_WIDTH 20
#define TEST_HEIGHT 20

using namespace gpu_planning;

void print_test(float* arr, size_t w, size_t h) {
  for (int i = 0; i < w; ++i) {
    std::cout << '-';
  }
  std::cout << '\n';

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (arr[y * w + x] < 1.0) {
        std::cout << ' ';
      } else {
        std::cout << 'X';
      }
    }
    std::cout << '\n';
  }

  for (int i = 0; i < w; ++i) {
    std::cout << '-';
  }
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
  CliArgs args = parse_cli_args(argc, argv);

  init_logging(args.verbose);

  logger log = create_logger();

  try {
    if (args.device >= 0) {
      cuda_set_device(args.device, log);
    }

    if (args.list_devices) {
      cuda_list_devices(log);
    } else {
      LOG_INFO(log) << "Done";
    }
  } catch (std::runtime_error& ex) {
    std::cerr << ex.what() << std::endl;
  }

  DeviceArray2D arr(TEST_WIDTH, TEST_HEIGHT, sizeof(float));
  arr.clear();

  float test_clear[TEST_WIDTH * TEST_HEIGHT] = {0};
  arr.read(0, 0, TEST_WIDTH, TEST_HEIGHT, test_clear);
  print_test(test_clear, TEST_WIDTH, TEST_HEIGHT);

  float test_write[TEST_WIDTH * TEST_HEIGHT] = {0};
  test_write[0] = 1.0;
  test_write[TEST_WIDTH] = 2.0;
  test_write[9 * TEST_WIDTH + 9] = 5.0;
  test_write[TEST_WIDTH * TEST_HEIGHT - 1] = 5.0;
  arr.write(0, 0, TEST_WIDTH, TEST_HEIGHT, test_write);

  float test_written[TEST_WIDTH * TEST_HEIGHT] = {0};
  arr.read(0, 0, TEST_WIDTH, TEST_HEIGHT, test_written);
  print_test(test_written, TEST_WIDTH, TEST_HEIGHT);

  float test_small[5 * 5] = {0};
  test_small[0] = 4.0;
  test_small[4 * 5] = 5.0;
  test_small[5 * 5 - 1] = 6.0;
  arr.write(10, 10, 5, 5, test_small);

  float test_small_written[TEST_WIDTH * TEST_HEIGHT] = {0};
  arr.read(0, 0, TEST_WIDTH, TEST_HEIGHT, test_small_written);
  print_test(test_small_written, TEST_WIDTH, TEST_HEIGHT);

  float test_small_read[10 * 10] = {0};
  arr.read(10, 10, 10, 10, test_small_read);
  print_test(test_small_read, 10, 10);

  return 0;
}
