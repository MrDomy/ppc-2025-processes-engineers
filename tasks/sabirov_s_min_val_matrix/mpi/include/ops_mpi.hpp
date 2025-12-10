#pragma once

#include "sabirov_s_min_val_matrix/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sabirov_s_min_val_matrix {

class SabirovSMinValMatrixMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit SabirovSMinValMatrixMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace sabirov_s_min_val_matrix
