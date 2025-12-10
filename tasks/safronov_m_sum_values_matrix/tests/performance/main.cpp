#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "safronov_m_sum_values_matrix/common/include/common.hpp"
#include "safronov_m_sum_values_matrix/mpi/include/ops_mpi.hpp"
#include "safronov_m_sum_values_matrix/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace safronov_m_sum_values_matrix {

class SafronovMSumValuesMatrixPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 4000000;
  InType input_data_;
  OutType res_;

  void SetUp() override {
    int n = static_cast<int>(std::sqrt(kCount_));
    input_data_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 2));
    res_ = std::vector<double>(2000, 4000.0);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (res_.size() != output_data.size()) {
      return false;
    }
    for (size_t i = 0; i < res_.size(); i++) {
      if (std::abs(res_[i] - output_data[i]) > 1e-10) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(SafronovMSumValuesMatrixPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, SafronovMSumValuesMatrixMPI, SafronovMSumValuesMatrixSEQ>(
        PPC_SETTINGS_safronov_m_sum_values_matrix);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = SafronovMSumValuesMatrixPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, SafronovMSumValuesMatrixPerfTest, kGtestValues, kPerfTestName);

}  // namespace safronov_m_sum_values_matrix
