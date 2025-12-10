#pragma once

#include "task/include/task.hpp"
#include "vasiliev_m_vec_signs/common/include/common.hpp"

namespace vasiliev_m_vec_signs {

class VasilievMVecSignsSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VasilievMVecSignsSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace vasiliev_m_vec_signs
