#include "morozov_n_sentence_count/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <iostream>
#include <string>

#include "morozov_n_sentence_count/common/include/common.hpp"

namespace morozov_n_sentence_count {

MorozovNSentenceCountMPI::MorozovNSentenceCountMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool MorozovNSentenceCountMPI::ValidationImpl() {
  validated_ = (!GetInput().empty()) && (GetOutput() == 0);
  return validated_;
}

bool MorozovNSentenceCountMPI::PreProcessingImpl() {
  if (!validated_) {
    return false;
  }
  if (GetInput()[0] == '.' || GetInput()[0] == '!' || GetInput()[0] == '?') {
    GetInput()[0] = ' ';
  }
  return true;
}

bool MorozovNSentenceCountMPI::RunImpl() {
  if (!validated_) {
    return false;
  }
  std::string &input = GetInput();

  int mpi_size = 0;
  int rank = 0;

  std::size_t index_start = 0;
  std::size_t index_end = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::size_t step = input.length() / mpi_size;

  index_start = step * rank;
  index_end = step + index_start;

  if (rank == mpi_size - 1) {
    index_end = input.length();
  }

  std::size_t counter = 0;
  for (std::size_t i = index_start; i < index_end; i++) {
    if ((input[i] == '.' || input[i] == '!' || input[i] == '?') && (input[i - 1] != '.') && (input[i - 1] != '?') &&
        (input[i - 1] != '!')) {
      counter++;
    }
  }

  const std::size_t k_counter = counter;
  std::size_t counter_sum = 0;
  MPI_Reduce(&k_counter, &counter_sum, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(&counter_sum, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  GetOutput() = counter_sum;

  // debug output
  std::cout << std::to_string(rank) + " : " + std::to_string(index_start) + " - " + std::to_string(index_end) +
                   "\nanswer: " + std::to_string(counter_sum) + "\n";

  return true;
}

bool MorozovNSentenceCountMPI::PostProcessingImpl() {
  return validated_;
}

}  // namespace morozov_n_sentence_count
