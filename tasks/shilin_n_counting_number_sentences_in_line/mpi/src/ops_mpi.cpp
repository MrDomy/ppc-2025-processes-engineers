#include "shilin_n_counting_number_sentences_in_line/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "shilin_n_counting_number_sentences_in_line/common/include/common.hpp"

namespace shilin_n_counting_number_sentences_in_line {

ShilinNCountingNumberSentencesInLineMPI::ShilinNCountingNumberSentencesInLineMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  }
  GetOutput() = 0;
}

bool ShilinNCountingNumberSentencesInLineMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    return GetOutput() == 0;
  }
  return true;
}

bool ShilinNCountingNumberSentencesInLineMPI::PreProcessingImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    GetOutput() = 0;
  }
  return true;
}

namespace {

bool IsPunctuation(char c) {
  return c == '.' || c == '!' || c == '?';
}
}  // namespace

int ShilinNCountingNumberSentencesInLineMPI::SkipPunctuationSequence(const std::string &input_str, int start_pos,
                                                                     int end_pos) {
  int pos = start_pos;
  while (pos < end_pos && IsPunctuation(input_str[static_cast<size_t>(pos)])) {
    ++pos;
  }
  return pos;
}

int ShilinNCountingNumberSentencesInLineMPI::CountSentencesFromPosition(const std::string &input_str, int start_pos,
                                                                        int end_pos) {
  int count = 0;
  for (int i = start_pos; i < end_pos; ++i) {
    if (IsPunctuation(input_str[static_cast<size_t>(i)])) {
      count++;
      i = SkipPunctuationSequence(input_str, i + 1, end_pos) - 1;
    }
  }
  return count;
}

int ShilinNCountingNumberSentencesInLineMPI::CountSentencesInChunk(const std::string &input_str, int start_pos,
                                                                   int end_pos, char left_boundary_char) {
  if (start_pos < end_pos && IsPunctuation(left_boundary_char) &&
      IsPunctuation(input_str[static_cast<size_t>(start_pos)])) {
    const int new_start_pos = SkipPunctuationSequence(input_str, start_pos, end_pos);
    return CountSentencesFromPosition(input_str, new_start_pos, end_pos);
  }

  return CountSentencesFromPosition(input_str, start_pos, end_pos);
}

bool ShilinNCountingNumberSentencesInLineMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::string input_str;
  int input_length = 0;

  if (rank == 0) {
    input_str = GetInput();
    input_length = static_cast<int>(input_str.length());
  }

  MPI_Bcast(&input_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (input_length == 0) {
    if (rank == 0) {
      GetOutput() = 0;
    }
    return true;
  }

  int chunk_size = input_length / size;
  int remainder = input_length % size;

  std::vector<int> sendcounts(size);
  std::vector<int> displacements(size);

  for (int i = 0; i < size; ++i) {
    sendcounts[i] = chunk_size + (i < remainder ? 1 : 0);
    displacements[i] = (i * chunk_size) + std::min(i, remainder);
  }

  int local_chunk_size = sendcounts[rank];
  std::string local_chunk(local_chunk_size, '\0');

  MPI_Scatterv(rank == 0 ? input_str.data() : nullptr, sendcounts.data(), displacements.data(), MPI_CHAR,
               local_chunk.data(), local_chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

  std::vector<char> left_boundary_chars(size);
  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      int start_pos = (i * chunk_size) + std::min(i, remainder);
      left_boundary_chars[i] = (start_pos > 0) ? input_str[static_cast<size_t>(start_pos - 1)] : '\0';
    }
  }

  char left_boundary_char = '\0';
  MPI_Scatter(left_boundary_chars.data(), 1, MPI_CHAR, &left_boundary_char, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

  int local_count = CountSentencesInChunk(local_chunk, 0, local_chunk_size, left_boundary_char);

  int global_count = 0;
  MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Bcast(&global_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = global_count;
  return true;
}

bool ShilinNCountingNumberSentencesInLineMPI::PostProcessingImpl() {
  return true;
}

}  // namespace shilin_n_counting_number_sentences_in_line
