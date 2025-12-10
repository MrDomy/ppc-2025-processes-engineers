#pragma once

#include "morozov_n_sentence_count/common/include/common.hpp"
#include "task/include/task.hpp"

namespace morozov_n_sentence_count {

class MorozovNSentenceCountMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit MorozovNSentenceCountMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  bool validated_ = false;
};

}  // namespace morozov_n_sentence_count
