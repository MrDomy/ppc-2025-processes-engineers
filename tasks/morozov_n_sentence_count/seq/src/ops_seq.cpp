#include "morozov_n_sentence_count/seq/include/ops_seq.hpp"

#include <cstddef>
#include <string>

#include "morozov_n_sentence_count/common/include/common.hpp"

namespace morozov_n_sentence_count {

MorozovNSentenceCountSEQ::MorozovNSentenceCountSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MorozovNSentenceCountSEQ::ValidationImpl() {
  validated_ = (!GetInput().empty()) && (GetOutput() == 0);
  return validated_;
}

bool MorozovNSentenceCountSEQ::PreProcessingImpl() {
  if (!validated_) {
    return false;
  }
  if (GetInput()[0] == '.' || GetInput()[0] == '!' || GetInput()[0] == '?') {
    GetInput()[0] = ' ';
  }
  return true;
}

bool MorozovNSentenceCountSEQ::RunImpl() {
  if (!validated_) {
    return false;
  }

  std::string &input = GetInput();
  std::size_t counter = 0;
  for (size_t i = 0; i < input.length(); i++) {
    if ((input[i] == '.' || input[i] == '!' || input[i] == '?') && (input[i - 1] != '.') && (input[i - 1] != '?') &&
        (input[i - 1] != '!')) {
      counter++;
    }
  }

  if (counter != 0) {
    GetOutput() = counter;
  }
  return true;
}

bool MorozovNSentenceCountSEQ::PostProcessingImpl() {
  return validated_;
}

}  // namespace morozov_n_sentence_count
