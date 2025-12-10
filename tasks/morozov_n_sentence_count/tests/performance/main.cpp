#include <gtest/gtest.h>

#include <cstddef>
#include <random>
#include <string>

#include "morozov_n_sentence_count/common/include/common.hpp"
#include "morozov_n_sentence_count/mpi/include/ops_mpi.hpp"
#include "morozov_n_sentence_count/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace morozov_n_sentence_count {

class MorozovNRunSentenceCountPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  // const std::string test_file_path_ = "test_4.txt";
  const std::size_t task_answer_ = 6000000;
  InType input_data_;

  void SetUp() override {
    input_data_ = GenerateTestData(task_answer_, 0);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == task_answer_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

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

TEST_P(MorozovNRunSentenceCountPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, MorozovNSentenceCountMPI, MorozovNSentenceCountSEQ>(
    PPC_SETTINGS_morozov_n_sentence_count);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = MorozovNRunSentenceCountPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MorozovNRunSentenceCountPerfTests, kGtestValues, kPerfTestName);

}  // namespace morozov_n_sentence_count
