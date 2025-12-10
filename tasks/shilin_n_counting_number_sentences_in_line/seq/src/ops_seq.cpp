#include "shilin_n_counting_number_sentences_in_line/seq/include/ops_seq.hpp"

#include <cstddef>
#include <string>

#include "shilin_n_counting_number_sentences_in_line/common/include/common.hpp"

namespace shilin_n_counting_number_sentences_in_line {

ShilinNCountingNumberSentencesInLineSEQ::ShilinNCountingNumberSentencesInLineSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool ShilinNCountingNumberSentencesInLineSEQ::ValidationImpl() {
  return GetOutput() == 0;
}

bool ShilinNCountingNumberSentencesInLineSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool ShilinNCountingNumberSentencesInLineSEQ::RunImpl() {
  const std::string &input = GetInput();
  int count = 0;

  for (size_t i = 0; i < input.length(); ++i) {
    char ch = input[i];
    if (ch == '.' || ch == '!' || ch == '?') {
      count++;
      while (i + 1 < input.length() && (input[i + 1] == '.' || input[i + 1] == '!' || input[i + 1] == '?')) {
        ++i;
      }
    }
  }

  GetOutput() = count;
  return true;
}

bool ShilinNCountingNumberSentencesInLineSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace shilin_n_counting_number_sentences_in_line
