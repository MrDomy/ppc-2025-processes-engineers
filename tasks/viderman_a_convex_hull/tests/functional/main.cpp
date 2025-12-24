#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "viderman_a_convex_hull/common/include/common.hpp"
#include "viderman_a_convex_hull/mpi/include/ops_mpi.hpp"
#include "viderman_a_convex_hull/seq/include/ops_seq.hpp"

namespace viderman_a_convex_hull {

class VidermanARunFuncConvexHull : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 protected:
  void SetUp() override {
    const int size = 1024;
    input_data_ = CreateComplexImage(size, size);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.empty() || std::ranges::all_of(output_data.begin(), output_data.end(),
                                                      [](const auto &component) { return !component.pixels.empty(); });
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;

  static ImageData CreateComplexImage(int width, int height) {
    ImageData image;
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height), 0);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist_x(0, width - 1);
    std::uniform_int_distribution<> dist_y(0, height - 1);

    const size_t num_random_points = (static_cast<size_t>(width) * static_cast<size_t>(height)) / 200;
    for (size_t idx = 0; idx < num_random_points; ++idx) {
      const int x = dist_x(gen);
      const int y = dist_y(gen);
      image.pixels[(static_cast<size_t>(y) * static_cast<size_t>(width)) + static_cast<size_t>(x)] = 255;
    }

    for (int square_idx = 0; square_idx < 8; ++square_idx) {
      const int square_size = 40 + (square_idx * 20);
      const int start_x = (square_idx * 150) % (width - square_size);
      const int start_y = (square_idx * 120) % (height - square_size);

      for (int dy = 0; dy < square_size; ++dy) {
        for (int dx = 0; dx < square_size; ++dx) {
          const int x = start_x + dx;
          const int y = start_y + dy;
          if (x < width && y < height) {
            image.pixels[(static_cast<size_t>(y) * static_cast<size_t>(width)) + static_cast<size_t>(x)] = 255;
          }
        }
      }
    }

    for (int line_idx = 0; line_idx < 10; ++line_idx) {
      const int length = 300;
      for (int j = 0; j < length; ++j) {
        const int x = (line_idx * 80 + j * 2) % width;
        const int y = (line_idx * 60 + j) % height;
        if (x < width && y < height) {
          image.pixels[(static_cast<size_t>(y) * static_cast<size_t>(width)) + static_cast<size_t>(x)] = 255;
        }
      }
    }

    return image;
  }

 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }
};

TEST_P(VidermanARunFuncConvexHull, FullCycle) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 2> kTestParam = {std::make_tuple("ImagePath", "tasks/viderman_a_convex_hull/data/pic.jpg"),
                                            std::make_tuple("ImageGenerated", "")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<VidermanAConvexHullSEQ, InType>(kTestParam, PPC_SETTINGS_viderman_a_convex_hull),
    ppc::util::AddFuncTask<VidermanAConvexHullMPI, InType>(kTestParam, PPC_SETTINGS_viderman_a_convex_hull));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kTestName = VidermanARunFuncConvexHull::PrintFuncTestName<VidermanARunFuncConvexHull>;

INSTANTIATE_TEST_SUITE_P(ConvexHullFuncTests, VidermanARunFuncConvexHull, kGtestValues, kTestName);

}  // namespace viderman_a_convex_hull
