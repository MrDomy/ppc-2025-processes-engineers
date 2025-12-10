#include "otcheskov_s_elem_vec_avg/seq/include/ops_seq.hpp"

#include <cmath>
#include <cstdint>
#include <numeric>

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"

namespace otcheskov_s_elem_vec_avg {

OtcheskovSElemVecAvgSEQ::OtcheskovSElemVecAvgSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = NAN;
}

bool OtcheskovSElemVecAvgSEQ::ValidationImpl() {
  return (!GetInput().empty() && std::isnan(GetOutput()));
}

bool OtcheskovSElemVecAvgSEQ::PreProcessingImpl() {
  return true;
}

bool OtcheskovSElemVecAvgSEQ::RunImpl() {
  if (GetInput().empty()) {
    return false;
  }

  int64_t sum = std::reduce(GetInput().begin(), GetInput().end(), int64_t{0});
  GetOutput() = static_cast<double>(sum) / static_cast<double>(GetInput().size());
  return !std::isnan(GetOutput());
}

bool OtcheskovSElemVecAvgSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_elem_vec_avg
