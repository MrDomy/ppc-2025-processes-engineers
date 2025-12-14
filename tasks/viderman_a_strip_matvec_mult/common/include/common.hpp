#pragma once

#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "task/include/task.hpp"

namespace viderman_a_strip_matvec_mult {

using InType = std::pair<std::vector<std::vector<double>>, std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::tuple<std::string, double>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace viderman_a_strip_matvec_mult
