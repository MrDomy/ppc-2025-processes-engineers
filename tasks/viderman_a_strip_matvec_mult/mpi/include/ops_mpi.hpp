#pragma once

#include "task/include/task.hpp"
#include "viderman_a_strip_matvec_mult/common/include/common.hpp"

namespace viderman_a_strip_matvec_mult {

class VidermanAStripMatvecMultMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit VidermanAStripMatvecMultMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace viderman_a_strip_matvec_mult
