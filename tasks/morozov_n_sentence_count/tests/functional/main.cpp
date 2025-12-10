#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>

#include "morozov_n_sentence_count/common/include/common.hpp"
#include "morozov_n_sentence_count/mpi/include/ops_mpi.hpp"
#include "morozov_n_sentence_count/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace morozov_n_sentence_count {

class MorozovNRunSentenceCountTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    if (std::get<1>(test_param).empty()) {
      return std::to_string(std::get<0>(test_param)) + "_" + "gen";
    }
    return std::to_string(std::get<0>(test_param)) + "_" + "file";
  }

 protected:
  void SetUp() override {
    std::string text;
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string test_file_path = std::get<1>(params);
    // Read text from params
    if (test_file_path.empty()) {
      task_answer_ = std::get<2>(params);
      input_data_ = GenerateTestData(task_answer_, 0);
    } else {
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_morozov_n_sentence_count, test_file_path);
      std::ifstream file(abs_path);
      if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + abs_path);
      }
      std::stringstream ss;
      ss << file.rdbuf();
      text = ss.str();

      task_answer_ = std::get<2>(params);
      input_data_ = text;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == task_answer_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  std::size_t task_answer_ = 0;

  static std::string GenerateTestData(const std::size_t s_count, const int seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dist('A', 'z');
    std::string res;
    char *sentence = new char['z' + 2];
    for (std::size_t i = 0; i < s_count; i++) {
      int sentence_size = dist(gen);
      for (int j = 0; j < sentence_size; j++) {
        sentence[j] = static_cast<char>(dist(gen));
      }
      sentence[sentence_size] = '.';
      sentence[sentence_size + 1] = '\0';
      res += sentence;
    }
    delete[] sentence;
    return res;
  }
};

namespace {

TEST(MorozovNSentenceCountTests, EmptyStringInputMPI) {
  MorozovNSentenceCountMPI task("");
  EXPECT_FALSE(task.Validation());
  EXPECT_FALSE(task.PreProcessing());
  EXPECT_FALSE(task.Run());
  EXPECT_FALSE(task.PostProcessing());
}

TEST(MorozovNSentenceCountTests, EmpyStringInputSEQ) {
  MorozovNSentenceCountSEQ task("");
  EXPECT_FALSE(task.Validation());
  EXPECT_FALSE(task.PreProcessing());
  EXPECT_FALSE(task.Run());
  EXPECT_FALSE(task.PostProcessing());
}

TEST_P(MorozovNRunSentenceCountTests, SentenceCountFromText) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 6> kTestParam = {std::make_tuple(0, "test_0.txt", 0), std::make_tuple(1, "test_1.txt", 1),
                                            std::make_tuple(2, "test_2.txt", 4), std::make_tuple(3, "test_3.txt", 100),
                                            std::make_tuple(4, "test_4.txt", 1), std::make_tuple(5, "", 10)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<MorozovNSentenceCountMPI, InType>(kTestParam, PPC_SETTINGS_morozov_n_sentence_count),
    ppc::util::AddFuncTask<MorozovNSentenceCountSEQ, InType>(kTestParam, PPC_SETTINGS_morozov_n_sentence_count));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = MorozovNRunSentenceCountTests::PrintFuncTestName<MorozovNRunSentenceCountTests>;

INSTANTIATE_TEST_SUITE_P(SentenceCountTest, MorozovNRunSentenceCountTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace morozov_n_sentence_count
