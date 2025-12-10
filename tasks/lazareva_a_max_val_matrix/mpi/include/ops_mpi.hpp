#pragma once

#include "lazareva_a_max_val_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace lazareva_a_max_val_matrix {

class LazarevaAMaxValMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit LazarevaAMaxValMatrixMPI(const InType &in);

 private:
  int n_ = 0;
  int m_ = 0;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace lazareva_a_max_val_matrix
