#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "viderman_a_convex_hull/common/include/common.hpp"
#include "viderman_a_convex_hull/seq/include/ops_seq.hpp"

namespace viderman_a_convex_hull {

namespace {

static ImageData CreateTestImage(int width, int height, const std::vector<std::vector<Point>> &components);

static bool IsConvex(const std::vector<Point> &hull);

static ImageData CreateTestImage(int width, int height, const std::vector<std::vector<Point>> &components) {
  ImageData image;
  image.width = width;
  image.height = height;
  image.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height), 0);

  for (const auto &component : components) {
    for (const auto &point : component) {
      if (point.first >= 0 && point.first < width && point.second >= 0 && point.second < height) {
        const size_t index =
            (static_cast<size_t>(point.second) * static_cast<size_t>(width)) + static_cast<size_t>(point.first);
        image.pixels[index] = 255;
      }
    }
  }

  return image;
}

static bool IsConvex(const std::vector<Point> &hull) {
  if (hull.size() < 3) {
    return true;
  }

  bool has_positive = false;
  bool has_negative = false;

  for (size_t idx = 0; idx < hull.size(); ++idx) {
    const size_t j = (idx + 1) % hull.size();
    const size_t k = (idx + 2) % hull.size();

    const int64_t cross = ((static_cast<int64_t>(hull[j].first - hull[idx].first) *
                            static_cast<int64_t>(hull[k].second - hull[j].second)) -
                           (static_cast<int64_t>(hull[j].second - hull[idx].second) *
                            static_cast<int64_t>(hull[k].first - hull[j].first)));

    if (cross > 0) {
      has_positive = true;
    }
    if (cross < 0) {
      has_negative = true;
    }

    if (has_positive && has_negative) {
      return false;
    }
  }

  return true;
}

}  // namespace

class VidermanARunFuncConvexHull : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return "case_" + std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    test_case_ = std::get<0>(params);
    test_name_ = std::get<1>(params);

    switch (test_case_) {
      case 0: {
        input_data_ = {.width = 10, .height = 10, .pixels = std::vector<uint8_t>(100, 0)};
        expected_count_ = 0;
        break;
      }
      case 1: {
        std::vector<std::vector<Point>> components = {{{5, 5}}};
        input_data_ = CreateTestImage(10, 10, components);
        expected_count_ = 1;
        break;
      }
      case 2: {
        std::vector<std::vector<Point>> components = {{{5, 3}, {5, 4}, {5, 5}, {5, 6}}};
        input_data_ = CreateTestImage(10, 10, components);
        expected_count_ = 1;
        break;
      }
      case 3: {
        std::vector<std::vector<Point>> components = {{{3, 5}, {4, 5}, {5, 5}, {6, 5}}};
        input_data_ = CreateTestImage(10, 10, components);
        expected_count_ = 1;
        break;
      }
      case 4: {
        std::vector<std::vector<Point>> components = {
            {{2, 2}, {3, 2}, {4, 2}, {2, 3}, {3, 3}, {4, 3}, {2, 4}, {3, 4}, {4, 4}}};
        input_data_ = CreateTestImage(10, 10, components);
        expected_count_ = 1;
        break;
      }
      case 5: {
        std::vector<std::vector<Point>> components = {{{1, 1}, {2, 1}, {1, 2}, {2, 2}},
                                                      {{6, 6}, {7, 6}, {6, 7}, {7, 7}}};
        input_data_ = CreateTestImage(10, 10, components);
        expected_count_ = 2;
        break;
      }
      default: {
        throw std::runtime_error("Unknown test case: " + std::to_string(test_case_));
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_count_) {
      return false;
    }

    if (test_case_ == 0) {
      return output_data.empty();
    }

    return std::ranges::all_of(output_data, [this](const auto &component) {
      if (component.pixels.empty()) {
        return false;
      }

      if (component.hull.empty()) {
        return false;
      }

      if (!IsConvex(component.hull)) {
        return false;
      }

      if (test_case_ == 1 && component.hull.size() != 1) {
        return false;
      }

      if ((test_case_ == 2 || test_case_ == 3) && component.hull.size() != 2) {
        return false;
      }

      return true;
    });
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  int test_case_ = 0;
  std::string test_name_;
  size_t expected_count_ = 0;
};

namespace {

TEST_P(VidermanARunFuncConvexHull, ConvexHullTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParams = {std::make_tuple(0, "empty_image"),   std::make_tuple(1, "single_pixel"),
                                             std::make_tuple(2, "vertical_line"), std::make_tuple(3, "horizontal_line"),
                                             std::make_tuple(4, "square_3x3"),    std::make_tuple(5, "two_squares")};

const auto kTestTasksList =
    ppc::util::AddFuncTask<VidermanAConvexHullSEQ, InType>(kTestParams, PPC_SETTINGS_viderman_a_convex_hull);

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = VidermanARunFuncConvexHull::PrintFuncTestName<VidermanARunFuncConvexHull>;

INSTANTIATE_TEST_SUITE_P(ConvexHullTests, VidermanARunFuncConvexHull, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace viderman_a_convex_hull
