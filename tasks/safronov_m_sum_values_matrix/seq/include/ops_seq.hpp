#pragma once

#include "safronov_m_sum_values_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace safronov_m_sum_values_matrix {

class SafronovMSumValuesMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit SafronovMSumValuesMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace safronov_m_sum_values_matrix
