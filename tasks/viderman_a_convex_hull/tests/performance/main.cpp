#include <gtest/gtest.h>

#include "util/include/perf_test_util.hpp"
#include "viderman_a_convex_hull/common/include/common.hpp"
#include "viderman_a_convex_hull/mpi/include/ops_mpi.hpp"
#include "viderman_a_convex_hull/seq/include/ops_seq.hpp"

namespace viderman_a_convex_hull {

class VidermanARunPerfConvexHull : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    int size = 1024;
    input_data_ = CreateSimpleImage(size, size);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (!output_data.empty()) {
      for (const auto &component : output_data) {
        if (component.pixels.empty()) {
          return false;
        }
      }
      return true;
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;

  static ImageData CreateSimpleImage(int width, int height) {
    ImageData image;
    image.width = width;
    image.height = height;
    image.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height), 0);

    if (width > 0 && height > 0) {
      image.pixels[0] = 255;
      image.pixels[width - 1] = 255;
      image.pixels[(height - 1) * width] = 255;
      image.pixels[(height - 1) * width + (width - 1)] = 255;

      if (width > 1 && height > 1) {
        image.pixels[(height / 2) * width + (width / 2)] = 255;
      }
    }

    return image;
  }
};

TEST_P(VidermanARunPerfConvexHull, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, VidermanAConvexHullMPI, VidermanAConvexHullSEQ>(
    PPC_SETTINGS_viderman_a_convex_hull);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = VidermanARunPerfConvexHull::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(ConvexHullPerfTests, VidermanARunPerfConvexHull, kGtestValues, kPerfTestName);

}  // namespace viderman_a_convex_hull
