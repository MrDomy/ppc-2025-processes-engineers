#include "viderman_a_strip_matvec_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "viderman_a_strip_matvec_mult/common/include/common.hpp"

namespace viderman_a_strip_matvec_mult {

VidermanAStripMatvecMultMPI::VidermanAStripMatvecMultMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool VidermanAStripMatvecMultMPI::ValidationImpl() {
  const InType &input = GetInput();
  const auto &matrix = input.first;
  const auto &vector = input.second;

  if (matrix.empty() && vector.empty()) {
    return true;
  }
  if (matrix.empty() || vector.empty()) {
    return false;
  }

  size_t cols = matrix[0].size();
  for (const auto &row : matrix) {
    if (row.size() != cols) {
      return false;
    }
  }

  if (cols != vector.size()) {
    return false;
  }

  return true;
}

bool VidermanAStripMatvecMultMPI::PreProcessingImpl() {
  return true;
}

bool VidermanAStripMatvecMultMPI::RunImpl() {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const InType &input = GetInput();
  const auto &full_matrix = input.first;
  const auto &full_vector = input.second;
  auto &result = GetOutput();

  if (full_matrix.empty() || full_vector.empty()) {
    result.clear();

    int result_size = 0;
    if (rank == 0) {
      result_size = 0;
      result.resize(0);
    }
    MPI_Bcast(&result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
      result.resize(result_size);
    }

    return true;
  }

  int rows = (rank == 0) ? static_cast<int>(full_matrix.size()) : 0;
  int cols = (rank == 0) ? static_cast<int>(full_vector.size()) : 0;

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rows == 0 || cols == 0) {
    result.clear();

    int result_size = 0;
    if (rank == 0) {
      result_size = 0;
      result.resize(0);
    }
    MPI_Bcast(&result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
      result.resize(result_size);
    }

    return true;
  }

  int rows_per_proc = rows / size;
  int remaining_rows = rows % size;

  int my_row_count = rows_per_proc + (rank < remaining_rows ? 1 : 0);

  std::vector<int> send_counts(size);
  std::vector<int> displacements(size);

  for (int i = 0; i < size; ++i) {
    send_counts[i] = rows_per_proc + (i < remaining_rows ? 1 : 0);
    if (i == 0) {
      displacements[i] = 0;
    } else {
      displacements[i] = displacements[i - 1] + send_counts[i - 1];
    }
  }

  std::vector<double> flat_full_matrix;
  std::vector<int> row_counts_flat(size);
  std::vector<int> displacements_flat(size);

  if (rank == 0) {
    for (const auto &row : full_matrix) {
      flat_full_matrix.insert(flat_full_matrix.end(), row.begin(), row.end());
    }

    for (int i = 0; i < size; ++i) {
      row_counts_flat[i] = send_counts[i] * cols;
      displacements_flat[i] = displacements[i] * cols;
    }
  } else {
    row_counts_flat.resize(size);
    displacements_flat.resize(size);
  }

  MPI_Bcast(row_counts_flat.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displacements_flat.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<double> flat_local_matrix(my_row_count * cols);
  MPI_Scatterv(rank == 0 ? flat_full_matrix.data() : nullptr, row_counts_flat.data(), displacements_flat.data(),
               MPI_DOUBLE, flat_local_matrix.data(), my_row_count * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_vector(cols);
  if (rank == 0) {
    local_vector = full_vector;
  }
  MPI_Bcast(local_vector.data(), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_result(my_row_count);
  for (int i = 0; i < my_row_count; ++i) {
    double sum = 0.0;
    for (int j = 0; j < cols; ++j) {
      sum += flat_local_matrix[i * cols + j] * local_vector[j];
    }
    local_result[i] = sum;
  }

  if (rank == 0) {
    result.resize(rows);
  }

  MPI_Gatherv(local_result.data(), my_row_count, MPI_DOUBLE, rank == 0 ? result.data() : nullptr, send_counts.data(),
              displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    result.resize(rows);
  }
  MPI_Bcast(result.data(), rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

bool VidermanAStripMatvecMultMPI::PostProcessingImpl() {
  return true;
}

}  // namespace viderman_a_strip_matvec_mult
