#pragma once

#include <string>

#include "shilin_n_counting_number_sentences_in_line/common/include/common.hpp"
#include "task/include/task.hpp"

namespace shilin_n_counting_number_sentences_in_line {

class ShilinNCountingNumberSentencesInLineMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit ShilinNCountingNumberSentencesInLineMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static int CountSentencesInChunk(const std::string &input_str, int start_pos, int end_pos, char left_boundary_char);
  static int SkipPunctuationSequence(const std::string &input_str, int start_pos, int end_pos);
  static int CountSentencesFromPosition(const std::string &input_str, int start_pos, int end_pos);
};

}  // namespace shilin_n_counting_number_sentences_in_line
