#include "viderman_a_strip_matvec_mult/seq/include/ops_seq.hpp"

#include "viderman_a_strip_matvec_mult/common/include/common.hpp"

namespace viderman_a_strip_matvec_mult {

VidermanAStripMatvecMultSEQ::VidermanAStripMatvecMultSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool VidermanAStripMatvecMultSEQ::ValidationImpl() {
  const InType &input = GetInput();
  const auto &matrix = input.first;
  const auto &vector = input.second;

  if (matrix.empty() && vector.empty()) {
    return true;
  }

  if (matrix.empty() || vector.empty()) {
    return false;
  }

  size_t cols = matrix[0].size();
  for (const auto &row : matrix) {
    if (row.size() != cols) {
      return false;
    }
  }

  if (cols != vector.size()) {
    return false;
  }

  return true;
}

bool VidermanAStripMatvecMultSEQ::PreProcessingImpl() {
  return true;
}

bool VidermanAStripMatvecMultSEQ::RunImpl() {
  const InType &input = GetInput();
  const auto &matrix = input.first;
  const auto &vector = input.second;
  auto &result = GetOutput();

  if (matrix.empty() || vector.empty()) {
    result.clear();
    return true;
  }

  result.resize(matrix.size());
  for (size_t i = 0; i < matrix.size(); ++i) {
    double sum = 0.0;
    for (size_t j = 0; j < matrix[i].size(); ++j) {
      sum += matrix[i][j] * vector[j];
    }
    result[i] = sum;
  }

  return true;
}

bool VidermanAStripMatvecMultSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace viderman_a_strip_matvec_mult
