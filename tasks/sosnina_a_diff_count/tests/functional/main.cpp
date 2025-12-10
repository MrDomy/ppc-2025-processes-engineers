#include <array>
#include <cstddef>
#include <map>
#include <string>
#include <tuple>
#include <utility>

// Добавляем необходимые заголовки для Google Test
#include <gtest/gtest.h>

#include "sosnina_a_diff_count/common/include/common.hpp"
#include "sosnina_a_diff_count/mpi/include/ops_mpi.hpp"
#include "sosnina_a_diff_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sosnina_a_diff_count {

class SosninaADiffCountFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    test_id_ = std::get<0>(params);
    test_name_ = std::get<1>(params);

    if (test_name_ == "__") {
      str1_ = "";
      str2_ = "";
    } else {
      auto pos = test_name_.find('_');
      if (pos != std::string::npos) {
        str1_ = test_name_.substr(0, pos);
        str2_ = test_name_.substr(pos + 1);
      } else {
        str1_ = test_name_;
        str2_ = "";
      }
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    static const std::map<int, int> kExpectedResults = {// FunctionalTests
                                                        {1, 1},
                                                        {2, 2},
                                                        {3, 0},
                                                        {4, 5},
                                                        {5, 3},
                                                        {6, 0},
                                                        {7, 4},
                                                        {8, 0},
                                                        {9, 1},
                                                        {10, 1},
                                                        {11, 1},
                                                        {12, 0},
                                                        {13, 2},
                                                        {14, 6},
                                                        {15, 3},
                                                        {16, 5},
                                                        {17, 0},
                                                        {18, 0},
                                                        {19, 3},
                                                        {20, 1},
                                                        {21, 20},
                                                        {22, 27},

                                                        // CoverageTests
                                                        {23, 0},
                                                        {24, 16},
                                                        {25, 3},
                                                        {26, 1},
                                                        {27, 1},
                                                        {28, 0},
                                                        {29, 0},
                                                        {30, 5},
                                                        {31, 5},
                                                        {32, 1}};

    auto it = kExpectedResults.find(test_id_);
    if (it != kExpectedResults.end()) {
      return output_data == it->second;
    }

    ADD_FAILURE() << "No expected result defined for test ID: " << test_id_;
    return false;
  }

  InType GetTestInputData() final {
    return std::make_pair(str1_, str2_);
  }

 private:
  int test_id_{};
  // Убираем избыточные инициализаторы для строк
  std::string test_name_;
  std::string str1_;
  std::string str2_;
};

namespace {

// Functional Tests
TEST_P(SosninaADiffCountFuncTests, FunctionalTests) {
  ExecuteTest(GetParam());
}

// Coverage Tests
TEST_P(SosninaADiffCountFuncTests, CoverageTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 22> kFunctionalTests = {std::make_tuple(1, "happy_heppy"),
                                                   std::make_tuple(2, "abcdef_abzzef"),
                                                   std::make_tuple(3, "baby_baby"),
                                                   std::make_tuple(4, "abc_defgh"),
                                                   std::make_tuple(5, "_mpi"),
                                                   std::make_tuple(6, "longstring_longstring"),
                                                   std::make_tuple(7, "abcd_efgh"),
                                                   std::make_tuple(8, "__"),
                                                   std::make_tuple(9, "z_"),
                                                   std::make_tuple(10, "_v"),
                                                   std::make_tuple(11, "z_v"),
                                                   std::make_tuple(12, "z_z"),
                                                   std::make_tuple(13, "zv_vz"),
                                                   std::make_tuple(14, "prizet_PRIZET"),
                                                   std::make_tuple(15, "zzz_vvv"),
                                                   std::make_tuple(16, "54321_09876"),
                                                   std::make_tuple(17, "TEST_TEST"),
                                                   std::make_tuple(18, "zv_zv"),
                                                   std::make_tuple(19, "veryverylongstringone_veryverylongstringtwo"),
                                                   std::make_tuple(20, "abcdefghij_abcdefghix"),
                                                   std::make_tuple(21, "short_very_long_string_mpi"),
                                                   std::make_tuple(22, "abc_hhfjjsalznzbzfgzzmookzmafgx")};

const std::array<TestType, 10> kCoverageTests = {
    std::make_tuple(23, "__"),        std::make_tuple(24, "short_very_long_string"),
    std::make_tuple(25, "a_bbb"),     std::make_tuple(26, "hello_hxllo"),
    std::make_tuple(27, "test_text"), std::make_tuple(28, ""),
    std::make_tuple(29, "_"),         std::make_tuple(30, "empty_"),
    std::make_tuple(31, "_empty"),    std::make_tuple(32, "a_")};

const auto kFunctionalTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<sosnina_a_diff_count::SosninaADiffCountMPI, InType>(
                       kFunctionalTests, PPC_SETTINGS_sosnina_a_diff_count),
                   ppc::util::AddFuncTask<sosnina_a_diff_count::SosninaADiffCountSEQ, InType>(
                       kFunctionalTests, PPC_SETTINGS_sosnina_a_diff_count));

const auto kCoverageTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<sosnina_a_diff_count::SosninaADiffCountMPI, InType>(
                       kCoverageTests, PPC_SETTINGS_sosnina_a_diff_count),
                   ppc::util::AddFuncTask<sosnina_a_diff_count::SosninaADiffCountSEQ, InType>(
                       kCoverageTests, PPC_SETTINGS_sosnina_a_diff_count));

inline const auto kFunctionalGtestValues = ppc::util::ExpandToValues(kFunctionalTasksList);
inline const auto kCoverageGtestValues = ppc::util::ExpandToValues(kCoverageTasksList);

inline const auto kPerfTestName = SosninaADiffCountFuncTests::PrintFuncTestName<SosninaADiffCountFuncTests>;

INSTANTIATE_TEST_SUITE_P(Functional, SosninaADiffCountFuncTests, kFunctionalGtestValues, kPerfTestName);
INSTANTIATE_TEST_SUITE_P(Coverage, SosninaADiffCountFuncTests, kCoverageGtestValues, kPerfTestName);

}  // namespace
}  // namespace sosnina_a_diff_count
