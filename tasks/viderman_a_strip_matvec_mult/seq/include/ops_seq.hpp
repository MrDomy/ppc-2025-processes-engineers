#pragma once

#include "task/include/task.hpp"
#include "viderman_a_strip_matvec_mult/common/include/common.hpp"

namespace viderman_a_strip_matvec_mult {

class VidermanAStripMatvecMultSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VidermanAStripMatvecMultSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace viderman_a_strip_matvec_mult
