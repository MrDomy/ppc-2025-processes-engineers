#include <gtest/gtest.h>

#include <cstddef>
#include <random>

#include "redkina_a_min_elem_vec/common/include/common.hpp"
#include "redkina_a_min_elem_vec/mpi/include/ops_mpi.hpp"
#include "redkina_a_min_elem_vec/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace redkina_a_min_elem_vec {

class RedkinaAMinElemVecRunPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 public:
  static constexpr size_t kSize = 100000000;

 protected:
  void SetUp() override {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> rand(-1000, 1000);

    input_vec_.resize(kSize);
    for (size_t i = 0; i < kSize; i++) {
      input_vec_[i] = rand(gen);
    }

    expected_res_ = -2000;
    input_vec_[kSize / 2] = expected_res_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_res_ == output_data;
  }

  InType GetTestInputData() final {
    return input_vec_;
  }

 private:
  InType input_vec_;
  OutType expected_res_{};
};

TEST_P(RedkinaAMinElemVecRunPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, RedkinaAMinElemVecMPI, RedkinaAMinElemVecSEQ>(
    PPC_SETTINGS_redkina_a_min_elem_vec);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = RedkinaAMinElemVecRunPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, RedkinaAMinElemVecRunPerfTests, kGtestValues, kPerfTestName);

}  // namespace redkina_a_min_elem_vec
