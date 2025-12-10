#include "safronov_m_sum_values_matrix/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "safronov_m_sum_values_matrix/common/include/common.hpp"

namespace safronov_m_sum_values_matrix {

SafronovMSumValuesMatrixSEQ::SafronovMSumValuesMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool SafronovMSumValuesMatrixSEQ::ValidationImpl() {
  if (GetInput().empty()) {
    return true;
  }
  size_t cols = GetInput()[0].size();
  std::size_t total = 0;
  for (const auto &row : GetInput()) {
    total += row.size();
  }
  return GetOutput().empty() && (cols != 0) && ((cols * GetInput().size()) == total);
}

bool SafronovMSumValuesMatrixSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool SafronovMSumValuesMatrixSEQ::RunImpl() {
  if (GetInput().empty()) {
    return true;
  }
  std::vector<double> vector(GetInput()[0].size());
  for (size_t i = 0; i < GetInput()[0].size(); i++) {
    double summa = 0;
    for (const auto &row : GetInput()) {
      summa += row[i];
    }
    vector[i] = summa;
  }
  GetOutput() = vector;
  return true;
}

bool SafronovMSumValuesMatrixSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace safronov_m_sum_values_matrix
