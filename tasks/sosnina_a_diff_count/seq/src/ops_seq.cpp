#include "sosnina_a_diff_count/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>

#include "sosnina_a_diff_count/common/include/common.hpp"

namespace sosnina_a_diff_count {

SosninaADiffCountSEQ::SosninaADiffCountSEQ(InType in) : input_(std::move(in)) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetOutput() = 0;
}

bool SosninaADiffCountSEQ::ValidationImpl() {
  return true;
}

bool SosninaADiffCountSEQ::PreProcessingImpl() {
  diff_counter_ = 0;
  return true;
}

bool SosninaADiffCountSEQ::RunImpl() {
  const std::string &str1 = input_.first;
  const std::string &str2 = input_.second;

  std::size_t total_len = std::max(str1.size(), str2.size());
  diff_counter_ = 0;

  for (std::size_t i = 0; i < total_len; i++) {
    if (i >= str1.size() || i >= str2.size() || str1[i] != str2[i]) {
      diff_counter_++;
    }
  }

  return true;
}

bool SosninaADiffCountSEQ::PostProcessingImpl() {
  GetOutput() = diff_counter_;
  return true;
}

}  // namespace sosnina_a_diff_count
