#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "safronov_m_sum_values_matrix/common/include/common.hpp"
#include "safronov_m_sum_values_matrix/mpi/include/ops_mpi.hpp"
#include "safronov_m_sum_values_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace safronov_m_sum_values_matrix {

class SafronovMSumValuesMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<1>(params);
    res_ = std::get<2>(params);
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

 private:
  InType input_data_;
  OutType res_;
};

namespace {

TEST_P(SafronovMSumValuesMatrixFuncTests, SumColumnsMatrix) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 10> kTestParam = {
    std::make_tuple("a", std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
                    std::vector<double>({12.0, 15.0, 18.0})),
    std::make_tuple("b", std::vector<std::vector<double>>{{1, 2, 3}}, std::vector<double>({1.0, 2.0, 3.0})),
    std::make_tuple("v", std::vector<std::vector<double>>{{1, 2, 3, 4}, {4, 5, 6, 7}, {7, 8, 9, 10}},
                    std::vector<double>({12.0, 15.0, 18.0, 21.0})),
    std::make_tuple("d", std::vector<std::vector<double>>{{-1.1, -2.2, -3.3}, {-4.4, 5, 6}, {2.4, 6, 4.5}},
                    std::vector<double>({-3.1, 8.8, 7.2})),
    std::make_tuple("e", std::vector<std::vector<double>>(100, std::vector<double>(100, 1)),
                    std::vector<double>(100, 100.0)),
    std::make_tuple("i", std::vector<std::vector<double>>(0), std::vector<double>(0)),
    std::make_tuple("g",
                    std::vector<std::vector<double>>{
                        {1.5, -2.0, 3.3, 4.1}, {0.5, 2.2, -1.3, 0.9}, {-3.0, 1.1, 4.0, -2.0}, {2.0, 0.0, -2.0, 1.0}},
                    std::vector<double>({1.0, 1.3, 4.0, 4.0})),
    std::make_tuple("c",
                    std::vector<std::vector<double>>{
                        {-10.0, 2.5, 3.0, -4.0, 100.0}, {5.0, -1.5, 20.0, 3.0, -50.0}, {2.0, 0.0, -5.0, 1.0, 25.0}},
                    std::vector<double>({-3.0, 1.0, 18.0, 0.0, 75.0})),
    std::make_tuple("j",
                    std::vector<std::vector<double>>{
                        {1.1, 2.2, 3.3}, {1.1, 2.2, 3.3}, {1.1, 2.2, 3.3}, {1.1, 2.2, 3.3}, {1.1, 2.2, 3.3}},
                    std::vector<double>({5.5, 11.0, 16.5})),
    std::make_tuple("m",
                    std::vector<std::vector<double>>{{3.0, 1.0, -2.0, 4.0, 0.5, -1.5},
                                                     {-1.0, 2.5, 3.5, -2.0, 4.0, 0.0},
                                                     {5.0, -3.0, 1.0, 0.0, -2.5, 10.0},
                                                     {0.0, 4.0, -1.0, 3.0, 1.0, -4.0}},
                    std::vector<double>({7.0, 4.5, 1.5, 5.0, 3.0, 4.5}))};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<SafronovMSumValuesMatrixMPI, InType>(kTestParam, PPC_SETTINGS_safronov_m_sum_values_matrix),
    ppc::util::AddFuncTask<SafronovMSumValuesMatrixSEQ, InType>(kTestParam, PPC_SETTINGS_safronov_m_sum_values_matrix));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = SafronovMSumValuesMatrixFuncTests::PrintFuncTestName<SafronovMSumValuesMatrixFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, SafronovMSumValuesMatrixFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace safronov_m_sum_values_matrix
