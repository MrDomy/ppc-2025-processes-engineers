#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "redkina_a_min_elem_vec/common/include/common.hpp"
#include "redkina_a_min_elem_vec/mpi/include/ops_mpi.hpp"
#include "redkina_a_min_elem_vec/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace redkina_a_min_elem_vec {

using TestType = std::tuple<int, std::vector<int>>;

class RedkinaAMinElemVecFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param));
  }

 protected:
  void SetUp() override {
    const TestType &params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    test_vector_ = std::get<1>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == -7;
  }

  InType GetTestInputData() final {
    return test_vector_;
  }

 private:
  std::vector<int> test_vector_;
};

namespace {

TEST_P(RedkinaAMinElemVecFuncTests, FunctionalTests) {
  ExecuteTest(GetParam());
}

TEST_P(RedkinaAMinElemVecFuncTests, CoverageTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 13> kFunctionalTests = {
    std::make_tuple(1, std::vector<int>{7, 2, 5, 3, 4, -7}),
    std::make_tuple(2, std::vector<int>{-7, -2, -5, -3, -4, -1}),
    std::make_tuple(3, std::vector<int>{4, -1, 5, -7, 3, -3}),
    std::make_tuple(4, std::vector<int>{-7}),
    std::make_tuple(5, std::vector<int>{3, 3, 3, 3, -7}),
    std::make_tuple(6, std::vector<int>{6, 1, 3, 9, 2, -7}),
    std::make_tuple(7, std::vector<int>{7000, 1500, 4000, -7, 6000}),
    std::make_tuple(8, std::vector<int>{10, -5, 20, 0, -7, 15}),
    std::make_tuple(9, std::vector<int>{-7, 1, 9, 14, 20}),
    std::make_tuple(10, std::vector<int>{88, 8, 18, 4, -7}),
    std::make_tuple(11, std::vector<int>{18, 8, -7, 17, 19}),
    std::make_tuple(12,
                    []() {
  std::vector<int> vec(1000);
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = static_cast<int>(vec.size() - i);
  }
  vec[500] = -7;
  return vec;
}()),
    std::make_tuple(13, std::vector<int>{std::numeric_limits<int>::max(), -7, 0, 100})};

const std::array<TestType, 20> kCoverageTests = {
    std::make_tuple(14, std::vector<int>{1, 2, -7}),
    std::make_tuple(16, std::vector<int>{1, 5, 8, 2, -7}),
    std::make_tuple(18, std::vector<int>{5, 8, 10, 4, -7}),
    std::make_tuple(19, std::vector<int>{-3, -1, -6, -7}),
    std::make_tuple(22, std::vector<int>{2, -7}),
    std::make_tuple(23, std::vector<int>{3, -7, 2}),
    std::make_tuple(25,
                    []() {
  std::vector<int> vec(100);
  for (size_t i = 0; i < vec.size(); ++i) {
    vec[i] = static_cast<int>(vec.size() - i);
  }
  vec[50] = -7;
  return vec;
}()),
    std::make_tuple(28, std::vector<int>{7, -6, 0, -7, 9, -1}),
    std::make_tuple(29, std::vector<int>{std::numeric_limits<int>::max(), -7, 0}),
    std::make_tuple(36, std::vector<int>{-7}),
    std::make_tuple(39, std::vector<int>{-7, 1, 2, 3}),
    std::make_tuple(40, std::vector<int>{-7, 1, 2, 3, 4}),
    std::make_tuple(41, std::vector<int>{-7, 1, 2, 3, 4, 5}),
    std::make_tuple(42, std::vector<int>{-7, 1, 2, 3, 4, 5, 6}),
    std::make_tuple(43, std::vector<int>{-7, 1, 2, 3, 4}),
    std::make_tuple(44, std::vector<int>{2, 3, 4, -7, 5}),
    std::make_tuple(45, std::vector<int>{5, 4, 3, 2, -7}),
    std::make_tuple(46, std::vector<int>{11, 6, 7, 2, 6, -7, 10, 4}),
    std::make_tuple(47, std::vector<int>{-1, -2, -3, -4, -7}),
    std::make_tuple(48, std::vector<int>{0, 0, 0, 0, -7})};

const auto kFunctionalTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<redkina_a_min_elem_vec::RedkinaAMinElemVecMPI, InType>(
                       kFunctionalTests, PPC_SETTINGS_redkina_a_min_elem_vec),
                   ppc::util::AddFuncTask<redkina_a_min_elem_vec::RedkinaAMinElemVecSEQ, InType>(
                       kFunctionalTests, PPC_SETTINGS_redkina_a_min_elem_vec));

const auto kCoverageTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<redkina_a_min_elem_vec::RedkinaAMinElemVecMPI, InType>(
                       kCoverageTests, PPC_SETTINGS_redkina_a_min_elem_vec),
                   ppc::util::AddFuncTask<redkina_a_min_elem_vec::RedkinaAMinElemVecSEQ, InType>(
                       kCoverageTests, PPC_SETTINGS_redkina_a_min_elem_vec));

inline const auto kFunctionalGtestValues = ppc::util::ExpandToValues(kFunctionalTasksList);
inline const auto kCoverageGtestValues = ppc::util::ExpandToValues(kCoverageTasksList);
inline const auto kPerfTestName = RedkinaAMinElemVecFuncTests::PrintFuncTestName<RedkinaAMinElemVecFuncTests>;

INSTANTIATE_TEST_SUITE_P(Functional, RedkinaAMinElemVecFuncTests, kFunctionalGtestValues, kPerfTestName);
INSTANTIATE_TEST_SUITE_P(Coverage, RedkinaAMinElemVecFuncTests, kCoverageGtestValues, kPerfTestName);

TEST(RedkinaAMinElemVecValidation, MpiEmptyVectorValidationFails) {
  InType vec = {};
  RedkinaAMinElemVecMPI task(vec);
  EXPECT_FALSE(task.Validation());
}

TEST(RedkinaAMinElemVecValidation, SeqEmptyVectorValidationFails) {
  InType vec = {};
  RedkinaAMinElemVecSEQ task(vec);
  EXPECT_FALSE(task.Validation());
}

}  // namespace
}  // namespace redkina_a_min_elem_vec
