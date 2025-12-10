#include "lazareva_a_max_val_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "lazareva_a_max_val_matrix/common/include/common.hpp"

namespace lazareva_a_max_val_matrix {

LazarevaAMaxValMatrixMPI::LazarevaAMaxValMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool LazarevaAMaxValMatrixMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int is_valid = 0;

  if (rank == 0) {
    is_valid = static_cast<int>(
        (GetInput().size() >= 2) && GetOutput().empty() && (GetInput()[0] > 0) && (GetInput()[1] > 0) &&
        (GetInput()[0] <= std::numeric_limits<int>::max() / GetInput()[1]) &&
        (GetInput().size() == (2 + (static_cast<size_t>(GetInput()[0]) * static_cast<size_t>(GetInput()[1])))));
  }

  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);

  return static_cast<bool>(is_valid);
}

bool LazarevaAMaxValMatrixMPI::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput().clear();
    n_ = GetInput()[0];
    m_ = GetInput()[1];
  }

  MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput().reserve(n_);
  }

  return true;
}

bool LazarevaAMaxValMatrixMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<int> matrix;
  if (rank == 0) {
    const auto &input = GetInput();
    matrix = std::vector<int>(input.begin() + 2, input.end());
  }

  int rows_per_proc = n_ / size;
  int remainder = n_ % size;

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);

  int offset = 0;
  for (int i = 0; i < size; i++) {
    int rows = rows_per_proc + (i < remainder ? 1 : 0);
    sendcounts[i] = rows * m_;
    displs[i] = offset;
    offset += rows * m_;
  }

  int local_rows = sendcounts[rank] / m_;
  std::vector<int> local_matrix(sendcounts[rank]);
  std::vector<int> local_max(local_rows);

  MPI_Scatterv(rank == 0 ? matrix.data() : nullptr, sendcounts.data(), displs.data(), MPI_INT, local_matrix.data(),
               sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

  for (int i = 0; i < local_rows; i++) {
    auto row_begin = local_matrix.begin() + static_cast<std::vector<int>::difference_type>(i) * m_;
    auto row_end = row_begin + m_;
    local_max[i] = *std::max_element(row_begin, row_end);
  }

  std::vector<int> recvcounts(size);
  std::vector<int> recvdispls(size);

  offset = 0;
  for (int i = 0; i < size; i++) {
    recvcounts[i] = sendcounts[i] / m_;
    recvdispls[i] = offset;
    offset += recvcounts[i];
  }

  std::vector<int> global_max;
  if (rank == 0) {
    global_max.resize(n_);
  }

  MPI_Gatherv(local_max.data(), local_rows, MPI_INT, rank == 0 ? global_max.data() : nullptr, recvcounts.data(),
              recvdispls.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    GetOutput() = global_max;
  }

  return true;
}

bool LazarevaAMaxValMatrixMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    return !GetOutput().empty() && (GetOutput().size() == static_cast<size_t>(n_));
  }

  return true;
}

}  // namespace lazareva_a_max_val_matrix
