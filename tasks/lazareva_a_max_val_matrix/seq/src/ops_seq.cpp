#include "lazareva_a_max_val_matrix/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "lazareva_a_max_val_matrix/common/include/common.hpp"

namespace lazareva_a_max_val_matrix {

LazarevaAMaxValMatrixSEQ::LazarevaAMaxValMatrixSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool LazarevaAMaxValMatrixSEQ::ValidationImpl() {
  const auto &input = GetInput();

  return (input.size() >= 2) && GetOutput().empty() && (input[0] > 0) && (input[1] > 0) &&
         (input[0] <= std::numeric_limits<int>::max() / input[1]) &&
         (input.size() == (2 + (static_cast<size_t>(input[0]) * static_cast<size_t>(input[1]))));
}

bool LazarevaAMaxValMatrixSEQ::PreProcessingImpl() {
  GetOutput().clear();
  int n = GetInput()[0];
  GetOutput().reserve(n);

  return true;
}

bool LazarevaAMaxValMatrixSEQ::RunImpl() {
  const auto &input = GetInput();
  int n = input[0];
  int m = input[1];
  const std::vector<int> matrix(input.begin() + 2, input.end());

  for (int i = 0; i < n; i++) {
    int row_start = i * m;
    int row_end = row_start + m;

    int max_val = *std::max_element(matrix.begin() + row_start, matrix.begin() + row_end);
    GetOutput().push_back(max_val);
  }

  return true;
}

bool LazarevaAMaxValMatrixSEQ::PostProcessingImpl() {
  int n = GetInput()[0];
  return !GetOutput().empty() && (GetOutput().size() == static_cast<size_t>(n));
}

}  // namespace lazareva_a_max_val_matrix
