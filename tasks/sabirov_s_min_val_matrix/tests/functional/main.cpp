#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>

#include "sabirov_s_min_val_matrix/common/include/common.hpp"
#include "sabirov_s_min_val_matrix/mpi/include/ops_mpi.hpp"
#include "sabirov_s_min_val_matrix/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace sabirov_s_min_val_matrix {

class SabirovSMinValMatrixFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != static_cast<size_t>(input_data_)) {
      return false;
    }

    auto generate_value = [](int64_t i, int64_t j) -> InType {
      constexpr int64_t kA = 1103515245LL;
      constexpr int64_t kC = 12345LL;
      constexpr int64_t kM = 2147483648LL;
      int64_t seed = ((i % kM) * (100000007LL % kM) + (j % kM) * (1000000009LL % kM)) % kM;
      seed = (seed ^ 42LL) % kM;
      int64_t val = ((kA % kM) * (seed % kM) + kC) % kM;
      return static_cast<InType>((val % 2000001LL) - 1000000LL);
    };

    for (size_t i = 0; i < output_data.size(); i++) {
      InType expected_min = generate_value(static_cast<int64_t>(i), 0);
      for (InType j = 1; j < input_data_; j++) {
        InType val = generate_value(static_cast<int64_t>(i), static_cast<int64_t>(j));
        expected_min = std::min(expected_min, val);
      }

      if (output_data[i] != expected_min) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(SabirovSMinValMatrixFuncTests, ComputesRowMinimumsForDiverseSizes) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 12> kFunctionalParams = {
    std::make_tuple(1, "size_1_unit"),       std::make_tuple(2, "size_2_pair"),
    std::make_tuple(3, "size_3_small"),      std::make_tuple(5, "size_5_fibonacci"),
    std::make_tuple(7, "size_7_prime"),      std::make_tuple(17, "size_17_prime"),
    std::make_tuple(31, "size_31_prime"),    std::make_tuple(64, "size_64_power2"),
    std::make_tuple(99, "size_99_odd"),      std::make_tuple(128, "size_128_even"),
    std::make_tuple(256, "size_256_power2"), std::make_tuple(512, "size_512_stress")};

const auto kTaskMatrix = std::tuple_cat(
    ppc::util::AddFuncTask<SabirovSMinValMatrixMPI, InType>(kFunctionalParams, PPC_SETTINGS_sabirov_s_min_val_matrix),
    ppc::util::AddFuncTask<SabirovSMinValMatrixSEQ, InType>(kFunctionalParams, PPC_SETTINGS_sabirov_s_min_val_matrix));

const auto kParameterizedValues = ppc::util::ExpandToValues(kTaskMatrix);

const auto kFunctionalTestName = SabirovSMinValMatrixFuncTests::PrintFuncTestName<SabirovSMinValMatrixFuncTests>;

INSTANTIATE_TEST_SUITE_P(RowMinimumSearchSuite, SabirovSMinValMatrixFuncTests, kParameterizedValues,
                         kFunctionalTestName);

template <typename TaskType>
void ExpectFullPipelineSuccess(InType n) {
  auto task = std::make_shared<TaskType>(n);
  ASSERT_TRUE(task->Validation());
  ASSERT_TRUE(task->PreProcessing());
  ASSERT_TRUE(task->Run());
  ASSERT_TRUE(task->PostProcessing());
  ASSERT_EQ(task->GetOutput().size(), static_cast<std::size_t>(n));

  auto generate_value = [](int64_t i, int64_t j) -> InType {
    constexpr int64_t kA = 1103515245;
    constexpr int64_t kC = 12345;
    constexpr int64_t kM = 2147483648;
    int64_t seed = (i * 100000007LL + j * 1000000009LL) ^ 42;
    int64_t val = (kA * seed + kC) % kM;
    return static_cast<InType>((val % 2000001) - 1000000);
  };

  for (InType i = 0; i < n; i++) {
    InType expected_min = generate_value(static_cast<int64_t>(i), 0);
    for (InType j = 1; j < n; j++) {
      InType val = generate_value(static_cast<int64_t>(i), static_cast<int64_t>(j));
      expected_min = std::min(expected_min, val);
    }
    ASSERT_EQ(task->GetOutput()[i], expected_min);
  }
}

TEST(SabirovSMinValMatrixStandalone, SeqPipelineHandlesEdgeSizes) {
  const std::array<InType, 6> k_sizes = {1, 4, 15, 33, 127, 255};
  for (InType size : k_sizes) {
    ExpectFullPipelineSuccess<SabirovSMinValMatrixSEQ>(size);
  }
}

TEST(SabirovSMinValMatrixStandalone, MpiPipelineHandlesEdgeSizes) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  const std::array<InType, 6> k_sizes = {1, 5, 18, 37, 130, 257};
  for (InType size : k_sizes) {
    ExpectFullPipelineSuccess<SabirovSMinValMatrixMPI>(size);
  }
}

TEST(SabirovSMinValMatrixValidation, RejectsZeroInputSeq) {
  SabirovSMinValMatrixSEQ task(0);
  EXPECT_FALSE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

TEST(SabirovSMinValMatrixValidation, RejectsZeroInputMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  SabirovSMinValMatrixMPI task(0);
  EXPECT_FALSE(task.Validation());
  task.PreProcessing();
  task.Run();
  task.PostProcessing();
}

TEST(SabirovSMinValMatrixValidation, AcceptsPositiveInputSeq) {
  SabirovSMinValMatrixSEQ task(10);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
}

TEST(SabirovSMinValMatrixValidation, AcceptsPositiveInputMpi) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  SabirovSMinValMatrixMPI task(10);
  EXPECT_TRUE(task.Validation());
  EXPECT_TRUE(task.PreProcessing());
  EXPECT_TRUE(task.Run());
  EXPECT_TRUE(task.PostProcessing());
}

template <typename TaskType>
void RunTaskTwice(TaskType &task, InType n) {
  task.GetInput() = n;
  task.GetOutput().clear();
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  ASSERT_EQ(task.GetOutput().size(), static_cast<std::size_t>(n));

  auto generate_value = [](int64_t i, int64_t j) -> InType {
    constexpr int64_t kA = 1103515245;
    constexpr int64_t kC = 12345;
    constexpr int64_t kM = 2147483648;
    int64_t seed = (i * 100000007LL + j * 1000000009LL) ^ 42;
    int64_t val = (kA * seed + kC) % kM;
    return static_cast<InType>((val % 2000001) - 1000000);
  };

  for (InType i = 0; i < n; i++) {
    InType expected_min = generate_value(static_cast<int64_t>(i), 0);
    for (InType j = 1; j < n; j++) {
      InType val = generate_value(static_cast<int64_t>(i), static_cast<int64_t>(j));
      expected_min = std::min(expected_min, val);
    }
    ASSERT_EQ(task.GetOutput()[i], expected_min);
  }
}

TEST(SabirovSMinValMatrixPipeline, SeqTaskCanBeReusedAcrossRuns) {
  SabirovSMinValMatrixSEQ task(4);
  RunTaskTwice(task, 4);
  RunTaskTwice(task, 9);
}

TEST(SabirovSMinValMatrixPipeline, MpiTaskCanBeReusedAcrossRuns) {
  if (!ppc::util::IsUnderMpirun()) {
    GTEST_SKIP();
  }
  SabirovSMinValMatrixMPI task(6);
  RunTaskTwice(task, 6);
  RunTaskTwice(task, 14);
}

}  // namespace

}  // namespace sabirov_s_min_val_matrix
