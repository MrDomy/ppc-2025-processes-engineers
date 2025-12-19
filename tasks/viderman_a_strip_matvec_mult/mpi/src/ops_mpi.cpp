#include "viderman_a_strip_matvec_mult/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "viderman_a_strip_matvec_mult/common/include/common.hpp"

namespace viderman_a_strip_matvec_mult {

VidermanAStripMatvecMultMPI::VidermanAStripMatvecMultMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = InType(in);
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

  const size_t cols = matrix[0].size();
  for (const auto &row : matrix) {
    if (row.size() != cols) {
      return false;
    }
  }

  return cols == vector.size();
}

bool VidermanAStripMatvecMultMPI::PreProcessingImpl() {
  return true;
}

namespace {

std::pair<std::vector<int>, std::vector<int>> CalculateDistribution(int rows, int size) {
  std::vector<int> send_counts(size);
  std::vector<int> displacements(size);

  const int rows_per_proc = rows / size;
  const int remaining_rows = rows % size;
  int displacement = 0;

  for (int i = 0; i < size; ++i) {
    send_counts[i] = rows_per_proc + (i < remaining_rows ? 1 : 0);
    displacements[i] = displacement;
    displacement += send_counts[i];
  }

  return {send_counts, displacements};
}

void HandleEmptyCase(int rank, std::vector<double> &result) {
  result.clear();
  if (rank == 0) {
    result.resize(0);
  }
  int result_size = static_cast<int>(result.size());
  MPI_Bcast(&result_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
  result.resize(static_cast<size_t>(result_size));
}

std::pair<int, int> BroadcastMatrixSizes(int rank, const std::vector<std::vector<double>> &full_matrix,
                                         const std::vector<double> &full_vector) {
  int rows = (rank == 0) ? static_cast<int>(full_matrix.size()) : 0;
  int cols = (rank == 0) ? static_cast<int>(full_vector.size()) : 0;

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return {rows, cols};
}

void PrepareScatterData(int rank, int size, const std::vector<std::vector<double>> &full_matrix,
                        const std::vector<int> &send_counts, const std::vector<int> &displacements, int cols,
                        std::vector<double> &flat_full_matrix, std::vector<int> &scatter_counts,
                        std::vector<int> &scatter_displacements) {
  if (rank == 0) {
    flat_full_matrix.reserve(static_cast<size_t>(full_matrix.size()) * static_cast<size_t>(cols));
    for (const auto &row : full_matrix) {
      flat_full_matrix.insert(flat_full_matrix.end(), row.begin(), row.end());
    }

    for (int i = 0; i < size; ++i) {
      scatter_counts[i] = send_counts[i] * cols;
      scatter_displacements[i] = displacements[i] * cols;
    }
  }
}

std::vector<double> ComputeLocalResult(int my_row_count, int cols, const std::vector<double> &flat_local_matrix,
                                       const std::vector<double> &local_vector) {
  std::vector<double> local_result(static_cast<size_t>(my_row_count), 0.0);
  for (int i = 0; i < my_row_count; ++i) {
    double sum = 0.0;
    const size_t row_offset = static_cast<size_t>(i) * static_cast<size_t>(cols);
    for (int j = 0; j < cols; ++j) {
      sum += flat_local_matrix[row_offset + static_cast<size_t>(j)] * local_vector[static_cast<size_t>(j)];
    }
    local_result[static_cast<size_t>(i)] = sum;
  }
  return local_result;
}

}  // namespace

bool VidermanAStripMatvecMultMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const InType &input = GetInput();
  const auto &full_matrix = input.first;
  const auto &full_vector = input.second;
  auto &result = GetOutput();

  if (full_matrix.empty() || full_vector.empty()) {
    HandleEmptyCase(rank, result);
    return true;
  }

  const auto [rows, cols] = BroadcastMatrixSizes(rank, full_matrix, full_vector);

  if (rows == 0 || cols == 0) {
    HandleEmptyCase(rank, result);
    return true;
  }

  const auto [send_counts, displacements] = CalculateDistribution(rows, size);
  const int my_row_count = send_counts[rank];

  std::vector<double> flat_full_matrix;
  std::vector<int> scatter_counts(size);
  std::vector<int> scatter_displacements(size);

  PrepareScatterData(rank, size, full_matrix, send_counts, displacements, cols, flat_full_matrix, scatter_counts,
                     scatter_displacements);

  MPI_Bcast(scatter_counts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(scatter_displacements.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  const size_t local_matrix_size = static_cast<size_t>(my_row_count) * static_cast<size_t>(cols);
  std::vector<double> flat_local_matrix(local_matrix_size);

  MPI_Scatterv(rank == 0 ? flat_full_matrix.data() : nullptr, scatter_counts.data(), scatter_displacements.data(),
               MPI_DOUBLE, flat_local_matrix.data(), static_cast<int>(local_matrix_size), MPI_DOUBLE, 0,
               MPI_COMM_WORLD);

  std::vector<double> local_vector(static_cast<size_t>(cols));
  if (rank == 0) {
    local_vector = full_vector;
  }
  MPI_Bcast(local_vector.data(), cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<double> local_result = ComputeLocalResult(my_row_count, cols, flat_local_matrix, local_vector);

  if (rank == 0) {
    result.resize(static_cast<size_t>(rows));
  }

  MPI_Gatherv(local_result.data(), my_row_count, MPI_DOUBLE, rank == 0 ? result.data() : nullptr, send_counts.data(),
              displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    result.resize(static_cast<size_t>(rows));
  }
  MPI_Bcast(result.data(), rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

bool VidermanAStripMatvecMultMPI::PostProcessingImpl() {
  return true;
}

}  // namespace viderman_a_strip_matvec_mult
