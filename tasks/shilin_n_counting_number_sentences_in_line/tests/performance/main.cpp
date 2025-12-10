#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <random>

#include "shilin_n_counting_number_sentences_in_line/common/include/common.hpp"
#include "shilin_n_counting_number_sentences_in_line/mpi/include/ops_mpi.hpp"
#include "shilin_n_counting_number_sentences_in_line/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace shilin_n_counting_number_sentences_in_line {

class ShilinNCountingNumberSentencesInLinePerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  InType input_data_;
  OutType expected_output_ = 0;

  void SetUp() override {
    const int num_sentences = 1000000;
    input_data_.clear();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> word_count_dist(3, 10);
    std::uniform_int_distribution<> word_len_dist(3, 8);
    std::uniform_int_distribution<> punct_dist(0, 2);

    expected_output_ = 0;

    for (int i = 0; i < num_sentences; ++i) {
      int word_count = word_count_dist(gen);
      for (int j = 0; j < word_count; ++j) {
        int word_len = word_len_dist(gen);
        for (int k = 0; k < word_len; ++k) {
          input_data_ += static_cast<char>('a' + (gen() % 26));
        }
        if (j < word_count - 1) {
          input_data_ += ' ';
        }
      }

      constexpr std::array<char, 3> kPunctuation = {'.', '!', '?'};
      input_data_ += kPunctuation.at(static_cast<size_t>(punct_dist(gen)));
      input_data_ += ' ';
      expected_output_++;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ShilinNCountingNumberSentencesInLinePerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, ShilinNCountingNumberSentencesInLineMPI,
                                                       ShilinNCountingNumberSentencesInLineSEQ>(
    PPC_SETTINGS_shilin_n_counting_number_sentences_in_line);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ShilinNCountingNumberSentencesInLinePerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ShilinNCountingNumberSentencesInLinePerfTests, kGtestValues, kPerfTestName);

}  // namespace shilin_n_counting_number_sentences_in_line
