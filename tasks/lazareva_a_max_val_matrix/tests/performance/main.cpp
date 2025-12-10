#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <random>

#include "lazareva_a_max_val_matrix/common/include/common.hpp"
#include "lazareva_a_max_val_matrix/mpi/include/ops_mpi.hpp"
#include "lazareva_a_max_val_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace lazareva_a_max_val_matrix {

class LazarevaARunPerfTestMaxValMatrix : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  int rows{};
  int cols{};
  InType input_data;
  OutType expected_output;

  void GenerateMatrix(int rows_param, int cols_param) {
    rows = rows_param;
    cols = cols_param;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(-10000, 10000);

    input_data.clear();
    input_data.reserve(2 + (rows * cols));

    input_data.push_back(rows);
    input_data.push_back(cols);

    expected_output.clear();
    expected_output.reserve(rows);

    for (int i = 0; i < rows; i++) {
      int max_val = std::numeric_limits<int>::min();
      for (int j = 0; j < cols; j++) {
        int val = dis(gen);
        input_data.push_back(val);
        max_val = std::max(max_val, val);
      }
      expected_output.push_back(max_val);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return true;
    }

    if (output_data.size() != expected_output.size()) {
      return false;
    }
    return std::equal(output_data.begin(), output_data.end(), expected_output.begin());
  }

  InType GetTestInputData() final {
    return input_data;
  }
};

class LazarevaARunPerfTestSmallMatrix : public LazarevaARunPerfTestMaxValMatrix {
 protected:
  void SetUp() override {
    GenerateMatrix(2000, 2000);
  }
};

class LazarevaARunPerfTestMediumMatrix : public LazarevaARunPerfTestMaxValMatrix {
 protected:
  void SetUp() override {
    GenerateMatrix(3500, 3500);
  }
};

class LazarevaARunPerfTestLargeMatrix : public LazarevaARunPerfTestMaxValMatrix {
 protected:
  void SetUp() override {
    GenerateMatrix(5000, 5000);
  }
};

class LazarevaARunPerfTestTallMatrix : public LazarevaARunPerfTestMaxValMatrix {
 protected:
  void SetUp() override {
    GenerateMatrix(5000, 2000);
  }
};

class LazarevaARunPerfTestWideMatrix : public LazarevaARunPerfTestMaxValMatrix {
 protected:
  void SetUp() override {
    GenerateMatrix(2000, 5000);
  }
};

TEST_P(LazarevaARunPerfTestSmallMatrix, RunPerfModes) {
  ExecuteTest(GetParam());
}

TEST_P(LazarevaARunPerfTestMediumMatrix, RunPerfModes) {
  ExecuteTest(GetParam());
}

TEST_P(LazarevaARunPerfTestLargeMatrix, RunPerfModes) {
  ExecuteTest(GetParam());
}

TEST_P(LazarevaARunPerfTestTallMatrix, RunPerfModes) {
  ExecuteTest(GetParam());
}

TEST_P(LazarevaARunPerfTestWideMatrix, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kSmallMatrixTasks = ppc::util::MakeAllPerfTasks<InType, LazarevaAMaxValMatrixMPI, LazarevaAMaxValMatrixSEQ>(
    PPC_SETTINGS_lazareva_a_max_val_matrix);

const auto kMediumMatrixTasks = ppc::util::MakeAllPerfTasks<InType, LazarevaAMaxValMatrixMPI, LazarevaAMaxValMatrixSEQ>(
    PPC_SETTINGS_lazareva_a_max_val_matrix);

const auto kLargeMatrixTasks = ppc::util::MakeAllPerfTasks<InType, LazarevaAMaxValMatrixMPI, LazarevaAMaxValMatrixSEQ>(
    PPC_SETTINGS_lazareva_a_max_val_matrix);

const auto kTallMatrixTasks = ppc::util::MakeAllPerfTasks<InType, LazarevaAMaxValMatrixMPI, LazarevaAMaxValMatrixSEQ>(
    PPC_SETTINGS_lazareva_a_max_val_matrix);

const auto kWideMatrixTasks = ppc::util::MakeAllPerfTasks<InType, LazarevaAMaxValMatrixMPI, LazarevaAMaxValMatrixSEQ>(
    PPC_SETTINGS_lazareva_a_max_val_matrix);

INSTANTIATE_TEST_SUITE_P(SmallMatrixTests, LazarevaARunPerfTestSmallMatrix,
                         ppc::util::TupleToGTestValues(kSmallMatrixTasks),
                         LazarevaARunPerfTestSmallMatrix::CustomPerfTestName);

INSTANTIATE_TEST_SUITE_P(MediumMatrixTests, LazarevaARunPerfTestMediumMatrix,
                         ppc::util::TupleToGTestValues(kMediumMatrixTasks),
                         LazarevaARunPerfTestMediumMatrix::CustomPerfTestName);

INSTANTIATE_TEST_SUITE_P(LargeMatrixTests, LazarevaARunPerfTestLargeMatrix,
                         ppc::util::TupleToGTestValues(kLargeMatrixTasks),
                         LazarevaARunPerfTestLargeMatrix::CustomPerfTestName);

INSTANTIATE_TEST_SUITE_P(TallMatrixTests, LazarevaARunPerfTestTallMatrix,
                         ppc::util::TupleToGTestValues(kTallMatrixTasks),
                         LazarevaARunPerfTestTallMatrix::CustomPerfTestName);

INSTANTIATE_TEST_SUITE_P(WideMatrixTests, LazarevaARunPerfTestWideMatrix,
                         ppc::util::TupleToGTestValues(kWideMatrixTasks),
                         LazarevaARunPerfTestWideMatrix::CustomPerfTestName);

}  // namespace lazareva_a_max_val_matrix
