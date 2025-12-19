#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"
#include "viderman_a_strip_matvec_mult/common/include/common.hpp"
#include "viderman_a_strip_matvec_mult/mpi/include/ops_mpi.hpp"
#include "viderman_a_strip_matvec_mult/seq/include/ops_seq.hpp"

namespace viderman_a_strip_matvec_mult {

class VidermanAStripMatvecMultFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<0>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    test_name_ = std::get<0>(params);
    expected_sum_ = std::get<1>(params);
    LoadTestData(test_name_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_vector_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (std::fabs(output_data[i] - expected_vector_[i]) > test_tolerance_) {
        return false;
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  void LoadTestData(const std::string &test_name) {
    std::string filename = "viderman_a_" + test_name + ".txt";
    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_viderman_a_strip_matvec_mult, filename);

    std::ifstream file(abs_path);

    if (!file.is_open()) {
      throw std::runtime_error("Cannot open test file: " + abs_path);
    }

    try {
      std::string line;
      std::getline(file, line);
      line.erase(0, line.find_first_not_of(" \t\r\n"));
      line.erase(line.find_last_not_of(" \t\r\n") + 1);

      if (line != "ROWS") {
        throw std::runtime_error("Expected 'ROWS', got: '" + line + "'");
      }

      std::getline(file, line);
      int rows = 0;
      if (!line.empty()) {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (!line.empty()) {
          rows = std::stoi(line);
        }
      }

      std::getline(file, line);
      line.erase(0, line.find_first_not_of(" \t\r\n"));
      line.erase(line.find_last_not_of(" \t\r\n") + 1);

      if (line != "COLS") {
        throw std::runtime_error("Expected 'COLS', got: '" + line + "'");
      }

      std::getline(file, line);
      int cols = 0;
      if (!line.empty()) {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        if (!line.empty()) {
          cols = std::stoi(line);
        }
      }

      std::getline(file, line);
      line.erase(0, line.find_first_not_of(" \t\r\n"));
      line.erase(line.find_last_not_of(" \t\r\n") + 1);

      if (line != "MATRIX") {
        throw std::runtime_error("Expected 'MATRIX', got: '" + line + "'");
      }

      std::vector<std::vector<double>> matrix;
      if (rows > 0 && cols > 0) {
        for (int i = 0; i < rows; ++i) {
          std::getline(file, line);
          std::istringstream iss(line);
          std::vector<double> row(static_cast<size_t>(cols));
          for (int j = 0; j < cols; ++j) {
            if (!(iss >> row[static_cast<size_t>(j)])) {
              throw std::runtime_error("Failed to read matrix element at row " + std::to_string(i) + ", col " +
                                       std::to_string(j));
            }
          }
          matrix.push_back(row);
        }
      }

      std::getline(file, line);
      line.erase(0, line.find_first_not_of(" \t\r\n"));
      line.erase(line.find_last_not_of(" \t\r\n") + 1);

      if (line != "VECTOR") {
        throw std::runtime_error("Expected 'VECTOR', got: '" + line + "'");
      }

      std::vector<double> vector;
      std::getline(file, line);
      if (cols > 0) {
        std::istringstream iss_vec(line);
        vector.resize(static_cast<size_t>(cols));
        for (int j = 0; j < cols; ++j) {
          if (!(iss_vec >> vector[static_cast<size_t>(j)])) {
            throw std::runtime_error("Failed to read vector element at position " + std::to_string(j));
          }
        }
      }

      std::getline(file, line);
      line.erase(0, line.find_first_not_of(" \t\r\n"));
      line.erase(line.find_last_not_of(" \t\r\n") + 1);

      if (line != "EXPECTED") {
        throw std::runtime_error("Expected 'EXPECTED', got: '" + line + "'");
      }

      std::getline(file, line);
      std::istringstream iss_exp(line);
      expected_vector_.clear();
      double value = 0.0;
      while (iss_exp >> value) {
        expected_vector_.push_back(value);
      }

      if (rows > 0 && expected_vector_.size() != static_cast<size_t>(rows)) {
        throw std::runtime_error("Expected result size mismatch: expected " + std::to_string(rows) + " elements, got " +
                                 std::to_string(expected_vector_.size()));
      }

      test_tolerance_ = 1e-10;

      if (std::getline(file, line)) {
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if (line == "TOLERANCE") {
          if (std::getline(file, line)) {
            test_tolerance_ = std::stod(line);
          }
        }
      }

      input_data_ = std::make_pair(std::move(matrix), std::move(vector));

    } catch (const std::exception &e) {
      throw std::runtime_error(std::string("Error parsing test file '") + abs_path + "': " + e.what());
    }
  }

  InType input_data_;
  OutType expected_vector_;
  std::string test_name_;
  double expected_sum_ = 0.0;
  double test_tolerance_ = 1e-10;
};

namespace {

TEST_P(VidermanAStripMatvecMultFuncTests, MatVecMultFuncTests) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 23> kTestParam = {std::make_tuple("empty", 0.0),
                                             std::make_tuple("1x1_positive", 0.0),
                                             std::make_tuple("1x1_negative", 0.0),
                                             std::make_tuple("square_2x2_simple", 0.0),
                                             std::make_tuple("square_3x3_mixed_signs", 0.0),
                                             std::make_tuple("square_4x4_large_values", 0.0),
                                             std::make_tuple("zero_matrix", 0.0),
                                             std::make_tuple("zero_vector", 0.0),
                                             std::make_tuple("identity_matrix", 0.0),
                                             std::make_tuple("diagonal_matrix", 0.0),
                                             std::make_tuple("rectangular_2x3", 0.0),
                                             std::make_tuple("rectangular_3x2", 0.0),
                                             std::make_tuple("fractions_2x2", 0.0),
                                             std::make_tuple("mpi_uneven_distribution_5rows", 0.0),
                                             std::make_tuple("mpi_more_processes_than_rows", 0.0),
                                             std::make_tuple("all_negative_matrix", 0.0),
                                             std::make_tuple("cancellation_effect", 0.0),
                                             std::make_tuple("precision_accumulation", 0.0),
                                             std::make_tuple("single_row_many_columns", 0.0),
                                             std::make_tuple("many_rows_single_column", 0.0),
                                             std::make_tuple("random_10x10", 0.0),
                                             std::make_tuple("random_5x8", 0.0),
                                             std::make_tuple("random_8x5", 0.0)};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<VidermanAStripMatvecMultMPI, InType>(kTestParam, PPC_SETTINGS_viderman_a_strip_matvec_mult),
    ppc::util::AddFuncTask<VidermanAStripMatvecMultSEQ, InType>(kTestParam, PPC_SETTINGS_viderman_a_strip_matvec_mult));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kFuncTestName = VidermanAStripMatvecMultFuncTests::PrintFuncTestName<VidermanAStripMatvecMultFuncTests>;

INSTANTIATE_TEST_SUITE_P(MatVecMultFuncTests, VidermanAStripMatvecMultFuncTests, kGtestValues, kFuncTestName);

}  // namespace

}  // namespace viderman_a_strip_matvec_mult
