#include "vasiliev_m_vec_signs/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "vasiliev_m_vec_signs/common/include/common.hpp"

namespace vasiliev_m_vec_signs {

VasilievMVecSignsSEQ::VasilievMVecSignsSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = OutType{};
}

bool VasilievMVecSignsSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool VasilievMVecSignsSEQ::PreProcessingImpl() {
  GetOutput() = 0;
  return true;
}

bool VasilievMVecSignsSEQ::RunImpl() {
  auto &vec = GetInput();
  int alters = 0;

  for (size_t i = 0; i < (vec.size() - 1); i++) {
    if ((vec[i] > 0 && vec[i + 1] < 0) || (vec[i] < 0 && vec[i + 1] > 0)) {
      alters++;
    }
  }

  GetOutput() = alters;
  return true;
}

bool VasilievMVecSignsSEQ::PostProcessingImpl() {
  return GetOutput() >= 0;
}

}  // namespace vasiliev_m_vec_signs
