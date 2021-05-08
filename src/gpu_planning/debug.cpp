#include "debug.hpp"

#include <stdint.h>
#include <string.h>

#include <fstream>
#include <memory>
#include <unordered_map>

#include "geometry.hpp"

namespace gpu_planning {

Color::Color() : bgr{0, 0, 0} {}

Color::Color(char r, char g, char b) : bgr{b, g, r} {}

const Color Color::BLACK = Color(0, 0, 0);
const Color Color::WHITE = Color(255, 255, 255);
const Color Color::RED = Color(255, 0, 0);
const Color Color::GREEN = Color(0, 255, 0);
const Color Color::BLUE = Color(0, 0, 255);
const Color Color::YELLOW = Color(255, 255, 0);

Overlay::Overlay() : width_{0}, height_{0}, data_{} {}

Overlay::Overlay(size_t width, size_t height)
    : width_{width},
      height_{height},
      data_{new OverlayClass[width_ * height_]} {
  memset(data_.get(), 0, width_ * height_ * sizeof(OverlayClass));
}

void Overlay::draw_point(const Position<size_t>& pos, OverlayClass cls) {
  data_[pos.y * width_ + pos.x] = cls;
}

void Overlay::draw_marker(const Position<size_t>& pos, OverlayClass cls) {
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

void Overlay::draw_line(const Position<size_t>& from,
                        const Position<size_t>& to, OverlayClass cls) {
  Translation<ssize_t> delta = to.cast<ssize_t>() - from.cast<ssize_t>();
  Position<size_t> start = from;
  bool transposed = false;

  if (abs(delta.x) < abs(delta.y)) {
    delta = Translation<ssize_t>(delta.y, delta.x);
    start = Position<size_t>(start.y, start.x);
    transposed = true;
  }

  const Translation<ssize_t> axis_dir = delta.signum();
  for (ssize_t ix = 0; ix != delta.x; ix += axis_dir.x) {
    const size_t x = start.x + ix;
    const size_t y = start.y + ix * delta.y / delta.x;

    if (transposed) {
      data_[x * width_ + y] = cls;
    } else {
      data_[y * width_ + x] = cls;
    }
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

struct OverlayClassHash {
  template <typename T>
  size_t operator()(T t) const {
    return static_cast<size_t>(t);
  }
};

void write_bmp(std::ostream& os, const Map& map, const Overlay& overlay) {
  const size_t width = map.data()->width();
  const size_t height = map.data()->height();

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

  std::unordered_map<Overlay::OverlayClass, Color, OverlayClassHash>
      cls_to_color;
  cls_to_color[Overlay::OverlayClass::NONE] = Color::BLACK;
  cls_to_color[Overlay::OverlayClass::BASE] = Color::BLUE;
  cls_to_color[Overlay::OverlayClass::ELBOW] = Color::YELLOW;
  cls_to_color[Overlay::OverlayClass::EE] = Color::RED;
  cls_to_color[Overlay::OverlayClass::S1] = Color(100, 100, 100);
  cls_to_color[Overlay::OverlayClass::S2] = Color(150, 150, 150);
  cls_to_color[Overlay::OverlayClass::EE_RECT] = Color(180, 180, 180);

  char buf[3];
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      Overlay::OverlayClass cls = overlay.get(Position<size_t>(x, y));
      if (cls != Overlay::OverlayClass::NONE) {
        Color color = cls_to_color[overlay.get(Position<size_t>(x, y))];
        memcpy(buf, color.bgr, 3 * sizeof(char));
      } else {
        char scaled = static_cast<char>(map.data()->at(x, y).value * 255);
        buf[0] = 255 - scaled;
        buf[1] = 255;
        buf[2] = 255 - scaled;
      }
      os.write(buf, 3);
    }

    // Line padding
    buf[0] = 0;
    buf[1] = 0;
    buf[2] = 0;
    os.write(buf, row_size - 3 * width);
  }
}  // namespace gpu_planning

void debug_save_state(DeviceMap& map, DeviceRobot& robot,
                      const std::vector<Configuration>& configurations,
                      const std::string& path, Logger* log) {
  HostMap host_map = map.load_to_host();

  Overlay overlay(host_map.map().data()->width(),
                  host_map.map().data()->height());

  const Position<size_t> base = host_map.map().to_index(robot.base().position);
  overlay.draw_marker(base, Overlay::OverlayClass::BASE);

  const Box<size_t> map_area = host_map.map().data()->area();
  for (const Configuration& conf : configurations) {
    const Pose<float> ee_pose = robot.fk_ee(conf);
    const Position<size_t> elbow =
        host_map.map().to_index(robot.fk_elbow(conf).position);
    const Position<size_t> ee_index = host_map.map().to_index(ee_pose.position);

    const Rectangle ee = robot.robot().ee();
    const Box<float> ee_bb =
        ee.bounding_box(ee_pose.orientation)
            .translate(ee_pose.position - Position<float>());
    const Box<size_t> ee_mask(
        map_area.clamp(host_map.map().to_index(ee_bb.lower_left)),
        map_area.clamp(host_map.map().to_index(ee_bb.upper_right)));

    for (size_t y = ee_mask.lower_left.y; y <= ee_mask.upper_right.y; ++y) {
      for (size_t x = ee_mask.lower_left.x; x <= ee_mask.upper_right.x; ++x) {
        const Position<size_t> pos(x, y);
        const Position<float> map_pos = host_map.map().from_index(pos);
        const Position<float> norm_pos =
            Position<float>() +
            (map_pos - ee_pose.position).rotate(-ee_pose.orientation);

        if (ee.is_inside(norm_pos)) {
          overlay.draw_point(pos, Overlay::OverlayClass::EE_RECT);
        }
      }
    }

    overlay.draw_line(base, elbow, Overlay::OverlayClass::S1);
    overlay.draw_line(elbow, ee_index, Overlay::OverlayClass::S2);

    overlay.draw_marker(elbow, Overlay::OverlayClass::ELBOW);
    overlay.draw_marker(ee_index, Overlay::OverlayClass::EE);
  }

  std::ofstream out(path, std::ios::binary | std::ios::out);
  write_bmp(out, host_map.map(), overlay);

  out.close();
  if (!out) {
    throw std::runtime_error{"Error while writing to bmp file"};
  }
}

}  // namespace gpu_planning
