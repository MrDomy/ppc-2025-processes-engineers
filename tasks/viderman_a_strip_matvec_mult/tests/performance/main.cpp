#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "viderman_a_strip_matvec_mult/common/include/common.hpp"
#include "viderman_a_strip_matvec_mult/mpi/include/ops_mpi.hpp"
#include "viderman_a_strip_matvec_mult/seq/include/ops_seq.hpp"

namespace viderman_a_strip_matvec_mult {

class VidermanAStripMatvecMultPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_result_;

  void SetUp() override {
    const int base_rows = 500;
    const int base_cols = 500;

    std::vector<std::vector<double>> matrix_base(static_cast<size_t>(base_rows),
                                                 std::vector<double>(static_cast<size_t>(base_cols)));
    for (int i = 0; i < base_rows; i++) {
      for (int j = 0; j < base_cols; j++) {
        matrix_base[static_cast<size_t>(i)][static_cast<size_t>(j)] = static_cast<double>(((i + j) % 100) + 1) + 0.5;
      }
    }

    std::vector<double> vector_base(static_cast<size_t>(base_cols));
    for (int j = 0; j < base_cols; j++) {
      vector_base[static_cast<size_t>(j)] = static_cast<double>((j % 50) + 1) + 0.25;
    }

    expected_result_.resize(static_cast<size_t>(base_rows));
    for (int i = 0; i < base_rows; i++) {
      double sum = 0.0;
      for (int j = 0; j < base_cols; j++) {
        sum += matrix_base[static_cast<size_t>(i)][static_cast<size_t>(j)] * vector_base[static_cast<size_t>(j)];
      }
      expected_result_[static_cast<size_t>(i)] = sum;
    }
    const int multiplier = 4;

    const int big_rows = base_rows * multiplier;
    const int big_cols = base_cols * multiplier;

    std::vector<std::vector<double>> big_matrix(static_cast<size_t>(big_rows),
                                                std::vector<double>(static_cast<size_t>(big_cols)));
    for (int i = 0; i < big_rows; i++) {
      const int block_row = i / base_rows;
      const int base_i = i % base_rows;
      for (int j = 0; j < big_cols; j++) {
        const int block_col = j / base_cols;
        const int base_j = j % base_cols;
        big_matrix[static_cast<size_t>(i)][static_cast<size_t>(j)] =
            matrix_base[static_cast<size_t>(base_i)][static_cast<size_t>(base_j)] *
            (1.0 + 0.1 * static_cast<double>(block_row) + 0.01 * static_cast<double>(block_col));
      }
    }

    std::vector<double> big_vector(static_cast<size_t>(big_cols));
    for (int j = 0; j < big_cols; j++) {
      const int block_col = j / base_cols;
      const int base_j = j % base_cols;
      big_vector[static_cast<size_t>(j)] =
          vector_base[static_cast<size_t>(base_j)] * (1.0 + 0.05 * static_cast<double>(block_col));
    }

    OutType big_expected;
    for (int block = 0; block < multiplier; block++) {
      for (int i = 0; i < base_rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < big_cols; j++) {
          const int base_i = i % base_rows;
          const int base_j = j % base_cols;
          const int block_col = j / base_cols;
          const double matrix_val = matrix_base[static_cast<size_t>(base_i)][static_cast<size_t>(base_j)] *
                                    (1.0 + 0.1 * static_cast<double>(block) + 0.01 * static_cast<double>(block_col));
          const double vector_val =
              vector_base[static_cast<size_t>(base_j)] * (1.0 + 0.05 * static_cast<double>(block_col));
          sum += matrix_val * vector_val;
        }
        big_expected.push_back(sum);
      }
    }

    input_data_ = {big_matrix, big_vector};
    expected_result_ = big_expected;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_result_.size()) {
      return false;
    }

    double max_relative_error = 0.0;
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::fabs(expected_result_[i]) > 1e-12) {
        const double relative_error = std::fabs(output_data[i] - expected_result_[i]) / std::fabs(expected_result_[i]);
        max_relative_error = std::max(max_relative_error, relative_error);
        if (relative_error > std::numeric_limits<double>::epsilon() * 1000) {
          return false;
        }
      } else {
        if (std::fabs(output_data[i] - expected_result_[i]) > 1e-12) {
          return false;
        }
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(VidermanAStripMatvecMultPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, VidermanAStripMatvecMultMPI, VidermanAStripMatvecMultSEQ>(
        PPC_SETTINGS_viderman_a_strip_matvec_mult);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = VidermanAStripMatvecMultPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(PerfTests, VidermanAStripMatvecMultPerfTests, kGtestValues, kPerfTestName);

}  // namespace viderman_a_strip_matvec_mult
