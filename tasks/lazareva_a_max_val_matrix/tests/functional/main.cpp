#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>

#include "lazareva_a_max_val_matrix/common/include/common.hpp"
#include "lazareva_a_max_val_matrix/mpi/include/ops_mpi.hpp"
#include "lazareva_a_max_val_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace lazareva_a_max_val_matrix {

class LazarevaAMaxValMatrixFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    std::string filename = "matrix_" + std::get<1>(params) + ".txt";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_lazareva_a_max_val_matrix, filename);

    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open matrix file: " + abs_path);
    }

    int n = 0;
    int m = 0;
    file >> n >> m;

    if (n <= 0 || m <= 0) {
      file.close();
      throw std::runtime_error("Invalid matrix dimensions: n=" + std::to_string(n) + ", m=" + std::to_string(m));
    }

    input_data_.clear();
    input_data_.push_back(n);
    input_data_.push_back(m);
    input_data_.reserve(2 + (n * m));

    expected_output_.clear();
    expected_output_.reserve(n);

    for (int i = 0; i < n; i++) {
      int max_val = std::numeric_limits<int>::min();
      for (int j = 0; j < m; j++) {
        int val = 0;
        if (!(file >> val)) {
          file.close();
          throw std::runtime_error("Failed to read matrix element at position [" + std::to_string(i) + "][" +
                                   std::to_string(j) + "]");
        }
        input_data_.push_back(val);
        max_val = std::max(max_val, val);
      }
      expected_output_.push_back(max_val);
    }

    file.close();
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return true;
    }

    if (output_data.size() != expected_output_.size()) {
      return false;
    }
    return std::equal(output_data.begin(), output_data.end(), expected_output_.begin());
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(LazarevaAMaxValMatrixFuncTest, FindMaxInRows) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 5> kTestParam = {std::make_tuple(1, "small"), std::make_tuple(2, "medium"),
                                            std::make_tuple(3, "large"), std::make_tuple(4, "negative"),
                                            std::make_tuple(5, "single")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<LazarevaAMaxValMatrixMPI, InType>(kTestParam, PPC_SETTINGS_lazareva_a_max_val_matrix),
    ppc::util::AddFuncTask<LazarevaAMaxValMatrixSEQ, InType>(kTestParam, PPC_SETTINGS_lazareva_a_max_val_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = LazarevaAMaxValMatrixFuncTest::PrintFuncTestName<LazarevaAMaxValMatrixFuncTest>;

INSTANTIATE_TEST_SUITE_P(MaxValMatrixTests, LazarevaAMaxValMatrixFuncTest, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace lazareva_a_max_val_matrix
