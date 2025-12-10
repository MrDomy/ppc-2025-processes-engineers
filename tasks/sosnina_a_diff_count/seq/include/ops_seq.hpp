#pragma once

#include "sosnina_a_diff_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace sosnina_a_diff_count {

class SosninaADiffCountSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }

  explicit SosninaADiffCountSEQ(InType in);

  [[nodiscard]] int GetDiffCount() const;

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  InType input_;
  int diff_counter_ = 0;
};

}  // namespace sosnina_a_diff_count
