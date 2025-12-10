#include <gtest/gtest.h>

#include <array>
#include <cctype>
#include <cstddef>
#include <functional>
#include <string>
#include <tuple>

#include "shilin_n_counting_number_sentences_in_line/common/include/common.hpp"
#include "shilin_n_counting_number_sentences_in_line/mpi/include/ops_mpi.hpp"
#include "shilin_n_counting_number_sentences_in_line/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"

namespace shilin_n_counting_number_sentences_in_line {

class ShilinNCountingNumberSentencesInLineRunFuncTestsProcesses
    : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string input = std::get<0>(test_param);
    std::string expected = std::get<1>(test_param);

    std::string sanitized;
    sanitized.reserve(input.length() + expected.length() + 20);

    for (char ch : input) {
      if (std::isalnum(static_cast<unsigned char>(ch)) != 0) {
        sanitized += ch;
      } else {
        sanitized += '_';
      }
    }

    std::size_t input_hash = std::hash<std::string>{}(input);
    sanitized += "_count_";
    sanitized += expected;
    sanitized += "_";
    sanitized += std::to_string(input_hash % 10000);

    return sanitized;
  }

 protected:
  void SetUp() override {
    auto params = std::get<2>(GetParam());
    input_data_ = std::get<0>(params);
    expected_output_ = std::stoi(std::get<1>(params));
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_ = 0;
};

namespace {

TEST_P(ShilinNCountingNumberSentencesInLineRunFuncTestsProcesses, CountSentences) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 24> kTestParam = {
    std::make_tuple("Hello world.", "1"),
    std::make_tuple("Hello! How are you?", "2"),
    std::make_tuple("This is a test. Another sentence! And one more?", "3"),
    std::make_tuple("", "0"),
    std::make_tuple("No sentences here", "0"),
    std::make_tuple("One. Two. Three.", "3"),
    std::make_tuple("Multiple punctuation...!!!", "1"),
    std::make_tuple("Mix. Of! Different? Endings.", "4"),
    std::make_tuple("Only dots...", "1"),
    std::make_tuple("Single! Exclamation!", "2"),
    std::make_tuple("Question? Answer! Statement.", "3"),
    std::make_tuple("Wow!!! Amazing!!! Great!!!", "3"),
    std::make_tuple("A.B.C.D.E.F.G.H.I.J.", "10"),
    std::make_tuple("Just one very long sentence without any ending", "0"),
    std::make_tuple("Start. Middle! End?", "3"),
    std::make_tuple("A.", "1"),               // один символ с точкой
    std::make_tuple(".!?", "1"),              // смешанная пунктуация в одном месте
    std::make_tuple("!?.", "1"),              // смешанная пунктуация в обратном порядке
    std::make_tuple("Hello . World !", "2"),  // пробелы вокруг знаков препинания
    std::make_tuple("A.B.C", "2"),            // минимальные предложения (2 точки, последнее без точки)
    std::make_tuple("abc..def.", "2"),        // последовательные точки на границе чанка (2 отдельные точки)
    std::make_tuple("abc...def.", "2"),       // многоточие на границе чанка (первая и последняя точки)
    std::make_tuple("abc!!!def.", "2"),       // многоточие и точка на границе
    std::make_tuple("abc.!?def.", "2")        // смешанная пунктуация на границе
};

const auto kTestTasksList = std::tuple_cat(ppc::util::AddFuncTask<ShilinNCountingNumberSentencesInLineMPI, InType>(
                                               kTestParam, PPC_SETTINGS_shilin_n_counting_number_sentences_in_line),
                                           ppc::util::AddFuncTask<ShilinNCountingNumberSentencesInLineSEQ, InType>(
                                               kTestParam, PPC_SETTINGS_shilin_n_counting_number_sentences_in_line));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = ShilinNCountingNumberSentencesInLineRunFuncTestsProcesses::PrintFuncTestName<
    ShilinNCountingNumberSentencesInLineRunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(SentenceCountingTests, ShilinNCountingNumberSentencesInLineRunFuncTestsProcesses, kGtestValues,
                         kPerfTestName);

}  // namespace

}  // namespace shilin_n_counting_number_sentences_in_line
