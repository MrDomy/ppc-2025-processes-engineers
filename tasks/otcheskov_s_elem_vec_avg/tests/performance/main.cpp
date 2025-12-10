#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"
#include "otcheskov_s_elem_vec_avg/mpi/include/ops_mpi.hpp"
#include "otcheskov_s_elem_vec_avg/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"

namespace otcheskov_s_elem_vec_avg {

class OtcheskovSElemVecAvgPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_avg_ = NAN;

  void SetUp() override {
    std::string filename = "test_vec_one_million_elems.txt";
    expected_avg_ = -2.60988;

    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_otcheskov_s_elem_vec_avg, filename);
    std::ifstream file(abs_path);

    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + abs_path);
    }

    int num{};
    while (file >> num) {
      input_data_.push_back(num);
    }
    file.close();

    // 2 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 4 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 8 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 16 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 32 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 64 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 128 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 256 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
    // 512 000 000 elements
    input_data_.insert(input_data_.end(), input_data_.begin(), input_data_.end());
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return std::fabs(expected_avg_ - output_data) < std::numeric_limits<double>::epsilon();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(OtcheskovSElemVecAvgPerfTests, VectorAveragePerfTests) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, OtcheskovSElemVecAvgMPI, OtcheskovSElemVecAvgSEQ>(
    PPC_SETTINGS_otcheskov_s_elem_vec_avg);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = OtcheskovSElemVecAvgPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(VectorAveragePerfTests, OtcheskovSElemVecAvgPerfTests, kGtestValues, kPerfTestName);

}  // namespace otcheskov_s_elem_vec_avg
