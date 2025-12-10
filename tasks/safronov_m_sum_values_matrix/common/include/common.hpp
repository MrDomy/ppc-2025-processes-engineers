#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace safronov_m_sum_values_matrix {

using InType = std::vector<std::vector<double>>;
using OutType = std::vector<double>;
using TestType = std::tuple<std::string, std::vector<std::vector<double>>, std::vector<double>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace safronov_m_sum_values_matrix
