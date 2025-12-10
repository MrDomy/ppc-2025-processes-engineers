#pragma once

#include "redkina_a_min_elem_vec/common/include/common.hpp"
#include "task/include/task.hpp"

namespace redkina_a_min_elem_vec {

class RedkinaAMinElemVecSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit RedkinaAMinElemVecSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace redkina_a_min_elem_vec
