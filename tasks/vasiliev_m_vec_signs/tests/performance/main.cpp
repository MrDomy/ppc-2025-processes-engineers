#include <gtest/gtest.h>

#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "util/include/util.hpp"
#include "vasiliev_m_vec_signs/common/include/common.hpp"
#include "vasiliev_m_vec_signs/mpi/include/ops_mpi.hpp"
#include "vasiliev_m_vec_signs/seq/include/ops_seq.hpp"

namespace vasiliev_m_vec_signs {

class VasilievMVecSignsPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  std::vector<std::pair<std::vector<int>, int>> test_vectors_;
  InType input_data_;
  OutType expected_output_ = 0;

  void SetUp() override {
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_vasiliev_m_vec_signs, "perf_test_vector.txt");
    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Wrong path.");
    }

    test_vectors_.clear();
    std::string line;
    while (std::getline(file, line)) {
      if (line.empty()) {
        continue;
      }

      std::stringstream ss(line);
      std::vector<int> vec;
      int val = 0;
      while (ss >> val) {
        vec.push_back(val);
        ss >> std::ws;
        if (ss.peek() == ';') {
          ss.get();
          break;
        }
      }

      int expected = 0;
      ss >> expected;

      std::vector<int> incr_vec = vec;

      for (int i = 0; i < 10; i++) {
        incr_vec.insert(incr_vec.end(), vec.begin(), vec.end());
      }

      int expected_incr = (expected * 11) + 10;

      test_vectors_.emplace_back(incr_vec, expected_incr);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return expected_output_ == output_data;
  }

  InType GetTestInputData() final {
    input_data_ = test_vectors_[0].first;
    expected_output_ = test_vectors_[0].second;
    return input_data_;
  }
};

TEST_P(VasilievMVecSignsPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, VasilievMVecSignsMPI, VasilievMVecSignsSEQ>(PPC_SETTINGS_vasiliev_m_vec_signs);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = VasilievMVecSignsPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, VasilievMVecSignsPerfTests, kGtestValues, kPerfTestName);

}  // namespace vasiliev_m_vec_signs
