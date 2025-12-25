#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "viderman_a_convex_hull/common/include/common.hpp"
#include "viderman_a_convex_hull/mpi/include/ops_mpi.hpp"
#include "viderman_a_convex_hull/seq/include/ops_seq.hpp"

namespace viderman_a_convex_hull {

class VidermanARunPerfConvexHull : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const int size = 1024;
    input_data = CreateComplexImage(size, size);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.empty() || std::ranges::all_of(output_data.begin(), output_data.end(),
                                                      [](const auto &component) { return !component.pixels.empty(); });
  }

  InType GetTestInputData() final {
    return input_data;
  }

 private:
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
  InType input_data;
};

TEST_F(VidermanARunPerfConvexHull, ComplexImageSEQ) {
  VidermanAConvexHullSEQ task(input_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

TEST_F(VidermanARunPerfConvexHull, ComplexImageMPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  VidermanAConvexHullMPI task(input_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  SUCCEED();
}

TEST(VidermanARunPerfConvexHullAdditional, EmptyImageSEQ) {
  ImageData input;
  input.width = 1024;
  input.height = 1024;
  input.pixels.resize(static_cast<size_t>(input.width) * static_cast<size_t>(input.height), 0);

  VidermanAConvexHullSEQ task_run(input);
  ASSERT_TRUE(task_run.Validation());
  ASSERT_TRUE(task_run.PreProcessing());
  ASSERT_TRUE(task_run.Run());
  ASSERT_TRUE(task_run.PostProcessing());
}

TEST(VidermanARunPerfConvexHullAdditional, EmptyImageMPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ImageData input;
  input.width = 1024;
  input.height = 1024;
  input.pixels.resize(static_cast<size_t>(input.width) * static_cast<size_t>(input.height), 0);

  VidermanAConvexHullMPI task_run(input);
  ASSERT_TRUE(task_run.Validation());
  ASSERT_TRUE(task_run.PreProcessing());
  ASSERT_TRUE(task_run.Run());
  ASSERT_TRUE(task_run.PostProcessing());

  SUCCEED();
}

TEST(VidermanARunPerfConvexHullAdditional, SinglePixelSEQ) {
  ImageData input;
  input.width = 1024;
  input.height = 1024;
  input.pixels.resize(static_cast<size_t>(input.width) * static_cast<size_t>(input.height), 0);
  input.pixels[(5 * input.width) + 5] = 255;

  VidermanAConvexHullSEQ task_run(input);
  ASSERT_TRUE(task_run.Validation());
  ASSERT_TRUE(task_run.PreProcessing());
  ASSERT_TRUE(task_run.Run());
  ASSERT_TRUE(task_run.PostProcessing());
}

TEST(VidermanARunPerfConvexHullAdditional, SinglePixelMPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ImageData input;
  input.width = 1024;
  input.height = 1024;
  input.pixels.resize(static_cast<size_t>(input.width) * static_cast<size_t>(input.height), 0);
  input.pixels[(5 * input.width) + 5] = 255;

  VidermanAConvexHullMPI task_run(input);
  ASSERT_TRUE(task_run.Validation());
  ASSERT_TRUE(task_run.PreProcessing());
  ASSERT_TRUE(task_run.Run());
  ASSERT_TRUE(task_run.PostProcessing());

  SUCCEED();
}

TEST(VidermanARunPerfConvexHullAdditional, VerticalLineSEQ) {
  ImageData input;
  input.width = 1024;
  input.height = 1024;
  input.pixels.resize(static_cast<size_t>(input.width) * static_cast<size_t>(input.height), 0);

  for (int width_idx = 0; width_idx < input.height; ++width_idx) {
    input.pixels[(width_idx * input.width) + 4] = 255;
  }

  VidermanAConvexHullSEQ task_run(input);
  ASSERT_TRUE(task_run.Validation());
  ASSERT_TRUE(task_run.PreProcessing());
  ASSERT_TRUE(task_run.Run());
  ASSERT_TRUE(task_run.PostProcessing());
}

TEST(VidermanARunPerfConvexHullAdditional, VerticalLineMPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ImageData input;
  input.width = 1024;
  input.height = 1024;
  input.pixels.resize(static_cast<size_t>(input.width) * static_cast<size_t>(input.height), 0);

  for (int width_idx = 0; width_idx < input.height; ++width_idx) {
    input.pixels[(width_idx * input.width) + 4] = 255;
  }

  VidermanAConvexHullMPI task_run(input);
  ASSERT_TRUE(task_run.Validation());
  ASSERT_TRUE(task_run.PreProcessing());
  ASSERT_TRUE(task_run.Run());
  ASSERT_TRUE(task_run.PostProcessing());

  SUCCEED();
}

TEST(VidermanARunPerfConvexHullAdditional, HorizontalLineSEQ) {
  ImageData input;
  input.width = 1024;
  input.height = 1024;
  input.pixels.resize(static_cast<size_t>(input.width) * static_cast<size_t>(input.height), 0);

  for (int height_idx = 0; height_idx < input.width; ++height_idx) {
    input.pixels[(6 * input.width) + height_idx] = 255;
  }

  VidermanAConvexHullSEQ task_run(input);
  ASSERT_TRUE(task_run.Validation());
  ASSERT_TRUE(task_run.PreProcessing());
  ASSERT_TRUE(task_run.Run());
  ASSERT_TRUE(task_run.PostProcessing());
}

TEST(VidermanARunPerfConvexHullAdditional, HorizontalLineMPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ImageData input;
  input.width = 1024;
  input.height = 1024;
  input.pixels.resize(static_cast<size_t>(input.width) * static_cast<size_t>(input.height), 0);

  for (int height_idx = 0; height_idx < input.width; ++height_idx) {
    input.pixels[(6 * input.width) + height_idx] = 255;
  }

  VidermanAConvexHullMPI task_run(input);
  ASSERT_TRUE(task_run.Validation());
  ASSERT_TRUE(task_run.PreProcessing());
  ASSERT_TRUE(task_run.Run());
  ASSERT_TRUE(task_run.PostProcessing());

  SUCCEED();
}

TEST(VidermanARunPerfConvexHullAdditional, FullSquare3x3SEQ) {
  ImageData input;
  input.width = 512;
  input.height = 512;
  input.pixels.resize(static_cast<size_t>(input.width) * static_cast<size_t>(input.height), 0);

  for (int width_idx = 2; width_idx < input.height / 2; ++width_idx) {
    for (int height_idx = 2; height_idx < input.width / 2; ++height_idx) {
      input.pixels[(width_idx * input.width) + height_idx] = 255;
    }
  }

  VidermanAConvexHullSEQ task_run(input);
  ASSERT_TRUE(task_run.Validation());
  ASSERT_TRUE(task_run.PreProcessing());
  ASSERT_TRUE(task_run.Run());
  ASSERT_TRUE(task_run.PostProcessing());
}

TEST(VidermanARunPerfConvexHullAdditional, FullSquare3x3MPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ImageData input;
  input.width = 512;
  input.height = 512;
  input.pixels.resize(static_cast<size_t>(input.width) * static_cast<size_t>(input.height), 0);

  for (int width_idx = 2; width_idx < input.height / 2; ++width_idx) {
    for (int height_idx = 2; height_idx < input.width / 2; ++height_idx) {
      input.pixels[(width_idx * input.width) + height_idx] = 255;
    }
  }

  VidermanAConvexHullMPI task_run(input);
  ASSERT_TRUE(task_run.Validation());
  ASSERT_TRUE(task_run.PreProcessing());
  ASSERT_TRUE(task_run.Run());
  ASSERT_TRUE(task_run.PostProcessing());

  SUCCEED();
}

TEST(VidermanARunPerfConvexHullAdditional, TwoSeparateSquaresSEQ) {
  ImageData input;
  input.width = 512;
  input.height = 512;
  input.pixels.resize(static_cast<size_t>(input.width) * static_cast<size_t>(input.height), 0);

  for (int width_idx = 1; width_idx < (input.height / 2) - 1; ++width_idx) {
    for (int height_idx = 1; height_idx < input.width / 4; ++height_idx) {
      input.pixels[(width_idx * input.width) + height_idx] = 255;
    }
  }

  for (int width_idx = (input.height / 2) + 1; width_idx < input.height - 1; ++width_idx) {
    for (int height_idx = (input.width * 3 / 4); height_idx < input.width - 1; ++height_idx) {
      input.pixels[(width_idx * input.width) + height_idx] = 255;
    }
  }

  VidermanAConvexHullSEQ task_run(input);
  ASSERT_TRUE(task_run.Validation());
  ASSERT_TRUE(task_run.PreProcessing());
  ASSERT_TRUE(task_run.Run());
  ASSERT_TRUE(task_run.PostProcessing());
}

TEST(VidermanARunPerfConvexHullAdditional, TwoSeparateSquaresMPI) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ImageData input;
  input.width = 512;
  input.height = 512;
  input.pixels.resize(static_cast<size_t>(input.width) * static_cast<size_t>(input.height), 0);

  for (int width_idx = 1; width_idx < (input.height / 2) - 1; ++width_idx) {
    for (int height_idx = 1; height_idx < input.width / 4; ++height_idx) {
      input.pixels[(width_idx * input.width) + height_idx] = 255;
    }
  }

  for (int width_idx = (input.height / 2) + 1; width_idx < input.height - 1; ++width_idx) {
    for (int height_idx = (input.width * 3 / 4); height_idx < input.width - 1; ++height_idx) {
      input.pixels[(width_idx * input.width) + height_idx] = 255;
    }
  }

  VidermanAConvexHullMPI task_run(input);
  ASSERT_TRUE(task_run.Validation());
  ASSERT_TRUE(task_run.PreProcessing());
  ASSERT_TRUE(task_run.Run());
  ASSERT_TRUE(task_run.PostProcessing());

  SUCCEED();
}

TEST_P(VidermanARunPerfConvexHull, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, VidermanAConvexHullSEQ, VidermanAConvexHullMPI>(
    PPC_SETTINGS_viderman_a_convex_hull);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = VidermanARunPerfConvexHull::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(ConvexHullPerfTests, VidermanARunPerfConvexHull, kGtestValues, kPerfTestName);

}  // namespace viderman_a_convex_hull
