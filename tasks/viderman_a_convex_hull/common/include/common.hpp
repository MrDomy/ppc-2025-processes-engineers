#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace viderman_a_convex_hull {

using Point = std::pair<int, int>;

struct Component {
  std::vector<Point> pixels;
  std::vector<Point> hull;
};

struct ImageData {
  int width{0};
  int height{0};
  std::vector<uint8_t> pixels;
};

using Components = std::vector<Component>;

using InType = ImageData;
using OutType = Components;
using TestType = std::tuple<std::string, std::string>;

using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace viderman_a_convex_hull
