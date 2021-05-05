#include "debug.hpp"

#include <stdint.h>
#include <string.h>

#include <fstream>
#include <memory>

#include "geometry.hpp"

namespace gpu_planning {

Overlay::Overlay() : width_{0}, height_{0}, data_{} {}

Overlay::Overlay(size_t width, size_t height)
    : width_{width},
      height_{height},
      data_{new OverlayClass[width_ * height_]} {
  memset(data_.get(), 0, width_ * height_ * sizeof(OverlayClass));
}

void Overlay::draw_point(const Position<size_t>& pos, OverlayClass cls) {
  data_[pos.y * width_ + pos.x] = cls;
  try_draw_point(pos + Translation<size_t>(1, 0), cls);
  try_draw_point(pos + Translation<size_t>(-1, 0), cls);
  try_draw_point(pos + Translation<size_t>(0, 1), cls);
  try_draw_point(pos + Translation<size_t>(0, -1), cls);
}

Overlay::OverlayClass Overlay::get(const Position<size_t>& pos) const {
  return data_[pos.y * width_ + pos.x];
}

void Overlay::try_draw_point(const Position<size_t>& pos, OverlayClass cls) {
  if (pos.x >= 0 && pos.x < width_ && pos.y >= 0 && pos.y < height_) {
    data_[pos.y * width_ + pos.x] = cls;
  }
}

void debug_print_map(DeviceMap& map, size_t max_width, size_t max_height,
                     Logger* log) {
  std::unique_ptr<float[]> buf(new float[max_width * max_height]);
  size_t width;
  size_t height;
  map.get_data(buf.get(), max_width, max_height, &width, &height);

  LOG_DEBUG(log) << "--- " << map.width() << "x" << map.height()
                 << " with resolution " << map.resolution() << " (shown as "
                 << width << "x" << height << ") ---";
  for (size_t y = 0; y < height; ++y) {
    std::string line = "|";
    for (size_t x = 0; x < width; ++x) {
      float val = buf[(height - y - 1) * width + x];

      if (val < 0.5) {
        line += ' ';
      } else if (val < 1.0) {
        line += 'X';
      } else {
        line += '#';
      }
    }
    LOG_DEBUG(log) << line << "|";
  }
  LOG_DEBUG(log) << "---";
}

struct __attribute__((__packed__)) bmp_file_header {
  uint16_t header_field;
  uint32_t file_size;
  uint16_t reserved_1;
  uint16_t reserved_2;
  uint32_t data_offset;
};

struct __attribute__((__packed__)) bmp_dib_header_core {
  uint32_t header_size;
  uint16_t bitmap_width;
  uint16_t bitmap_height;
  uint16_t num_color_planes;
  uint16_t bits_per_pixel;
};

uint16_t little_endian_16(uint16_t val) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return val;
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return __builtin_bswap16(val);
#else
#error "Only little and big endian machines are supported"
#endif
}

uint32_t little_endian_32(uint32_t val) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  return val;
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  return __builtin_bswap32(val);
#else
#error "Only little and big endian machines are supported"
#endif
}

void write_bmp(std::ostream& os, float* map, const Overlay& overlay,
               size_t width, size_t height) {
  const size_t row_size = ((width * 3 - 1) / 4 + 1) * 4;

  bmp_file_header file_header;
  file_header.header_field = little_endian_16(0x4D42);
  file_header.file_size =
      little_endian_32(sizeof(bmp_file_header) + sizeof(bmp_dib_header_core) +
                       height * row_size);
  file_header.reserved_1 = 0;
  file_header.reserved_2 = 0;
  file_header.data_offset =
      little_endian_32(sizeof(bmp_file_header) + sizeof(bmp_dib_header_core));
  os.write(reinterpret_cast<char*>(&file_header), sizeof(file_header));

  bmp_dib_header_core dib_header;
  dib_header.header_size = little_endian_32(sizeof(bmp_dib_header_core));
  dib_header.bitmap_width = width;
  dib_header.bitmap_height = height;
  dib_header.num_color_planes = little_endian_16(1);
  dib_header.bits_per_pixel = little_endian_16(24);
  os.write(reinterpret_cast<char*>(&dib_header), sizeof(dib_header));

  char buf[3];  // BGR
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      switch (overlay.get(Position<size_t>(x, y))) {
        case Overlay::OverlayClass::NONE: {
          char scaled = (char)(map[y * width + x] * 255);
          buf[0] = 255 - scaled;
          buf[1] = 255;
          buf[2] = 255 - scaled;
          break;
        }

        case Overlay::OverlayClass::BASE: {
          buf[0] = 255;
          buf[1] = 0;
          buf[2] = 0;
          break;
        }

        case Overlay::OverlayClass::ELBOW: {
          buf[0] = 0;
          buf[1] = 255;
          buf[2] = 255;
          break;
        }

        case Overlay::OverlayClass::EE: {
          buf[0] = 0;
          buf[1] = 0;
          buf[2] = 255;
          break;
        }

        default:
          buf[0] = 0;
          buf[1] = 0;
          buf[2] = 0;
      }

      os.write(buf, 3);
    }

    // Line padding
    buf[0] = 0;
    buf[1] = 0;
    buf[2] = 0;
    os.write(buf, row_size - 3 * width);
  }
}

void debug_save_state(DeviceMap& map, DeviceRobot& robot,
                      const std::vector<Configuration>& configurations,
                      size_t max_width, size_t max_height,
                      const std::string& path, Logger* log) {
  std::unique_ptr<float[]> data(new float[max_width * max_height]);
  size_t img_width;
  size_t img_height;
  map.get_data(data.get(), max_width, max_height, &img_width, &img_height);

  Overlay overlay(img_width, img_height);

  const size_t fact_x = map.index_width() / img_width;
  const size_t fact_y = map.index_height() / img_height;
  overlay.draw_point(
      map.to_index(robot.base().position).scale_down(fact_x, fact_y),
      Overlay::OverlayClass::BASE);

  for (const Configuration& conf : configurations) {
    overlay.draw_point(
        map.to_index(robot.fk_elbow(conf).position).scale_down(fact_x, fact_y),
        Overlay::OverlayClass::ELBOW);
    overlay.draw_point(
        map.to_index(robot.fk_ee(conf).position).scale_down(fact_x, fact_y),
        Overlay::OverlayClass::EE);
  }

  std::ofstream out(path, std::ios::binary | std::ios::out);
  write_bmp(out, data.get(), overlay, img_width, img_height);

  out.close();
  if (!out) {
    throw std::runtime_error{"Error while writing to bmp file"};
  }
}

}  // namespace gpu_planning
