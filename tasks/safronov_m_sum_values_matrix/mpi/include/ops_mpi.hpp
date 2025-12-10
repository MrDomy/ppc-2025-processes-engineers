#pragma once
#include <vector>

#include "safronov_m_sum_values_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace safronov_m_sum_values_matrix {

class SafronovMSumValuesMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SafronovMSumValuesMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  std::vector<double> SummValues(int start, int end);
  bool SendingOutMatrix(int rank);
  std::vector<double> ConversionToVector(int rows, int cols, int rank);
  void ConversionToMatrix(const std::vector<double> &vector, int rows, int cols, int rank);
  static std::vector<int> CalculatingInterval(int size_prcs, int rank, int count_column);
};

}  // namespace safronov_m_sum_values_matrix
