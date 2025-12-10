#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace shilin_n_counting_number_sentences_in_line {

using InType = std::string;
using OutType = int;
using TestType = std::tuple<std::string, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace shilin_n_counting_number_sentences_in_line
