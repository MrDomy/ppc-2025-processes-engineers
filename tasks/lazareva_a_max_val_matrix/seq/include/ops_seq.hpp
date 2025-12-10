#pragma once

#include "lazareva_a_max_val_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lazareva_a_max_val_matrix {

class LazarevaAMaxValMatrixSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit LazarevaAMaxValMatrixSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace lazareva_a_max_val_matrix
