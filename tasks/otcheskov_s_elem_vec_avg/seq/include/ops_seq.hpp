#pragma once

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"
#include "task/include/task.hpp"

namespace otcheskov_s_elem_vec_avg {

class OtcheskovSElemVecAvgSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit OtcheskovSElemVecAvgSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace otcheskov_s_elem_vec_avg
