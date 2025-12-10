#include "safronov_m_sum_values_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <vector>

#include "safronov_m_sum_values_matrix/common/include/common.hpp"

namespace safronov_m_sum_values_matrix {

SafronovMSumValuesMatrixMPI::SafronovMSumValuesMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool SafronovMSumValuesMatrixMPI::ValidationImpl() {
  if (GetInput().empty()) {
    return true;
  }
  size_t cols = GetInput()[0].size();
  std::size_t total = 0;
  for (const auto &row : GetInput()) {
    total += row.size();
  }
  return GetOutput().empty() && (cols != 0) && ((cols * GetInput().size()) == total);
}

bool SafronovMSumValuesMatrixMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

std::vector<double> SafronovMSumValuesMatrixMPI::SummValues(const int start, const int end) {
  std::vector<double> vec;
  for (int i = start; i <= end; i++) {
    double summa = 0;
    for (const auto &row : GetInput()) {
      summa += row[i];
    }
    vec.push_back(summa);
  }
  return vec;
}

std::vector<int> SafronovMSumValuesMatrixMPI::CalculatingInterval(int size_prcs, int rank, int count_column) {
  std::vector<int> vec(2);
  int whole_part = count_column / size_prcs;
  int real_part = count_column % size_prcs;
  int start = rank * whole_part;
  if ((rank - 1 < real_part) && (rank - 1 != -1)) {
    start += rank;
  } else if (rank != 0) {
    start += real_part;
  }
  int end = start + whole_part - 1;
  if (rank < real_part) {
    end += 1;
  }
  vec[0] = start;
  vec[1] = end;
  return vec;
}

std::vector<double> SafronovMSumValuesMatrixMPI::ConversionToVector(int rows, int cols, int rank) {
  std::vector<double> vector(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
  if (rank == 0) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        vector[(i * cols) + j] = GetInput()[i][j];
      }
    }
  }
  return vector;
}

void SafronovMSumValuesMatrixMPI::ConversionToMatrix(const std::vector<double> &vector, int rows, int cols, int rank) {
  if (rank != 0) {
    GetInput() = std::vector<std::vector<double>>(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        GetInput()[i][j] = vector[(i * cols) + j];
      }
    }
  }
}

bool SafronovMSumValuesMatrixMPI::SendingOutMatrix(int rank) {
  int rows = 0;
  int cols = 0;
  if (rank == 0) {
    rows = static_cast<int>(GetInput().size());
    cols = (rows != 0) ? static_cast<int>(GetInput()[0].size()) : 0;
  }

  MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rows == 0) {
    return true;
  }

  std::vector<double> vector = ConversionToVector(rows, cols, rank);

  MPI_Bcast(vector.data(), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  ConversionToMatrix(vector, rows, cols, rank);

  return false;
}

bool SafronovMSumValuesMatrixMPI::RunImpl() {
  int size = 0;
  int rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  bool flag = SendingOutMatrix(rank);
  if (flag) {
    return true;
  }

  if (rank == 0) {
    int count_column = static_cast<int>(GetInput()[0].size());

    for (int i = 1; i < size; i++) {
      std::vector<int> interval = CalculatingInterval(size, i, count_column);
      MPI_Send(interval.data(), 2, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    std::vector<int> interval = CalculatingInterval(size, 0, count_column);
    std::vector<double> elems = SummValues(interval[0], interval[1]);
    for (double &elem : elems) {
      GetOutput().push_back(elem);
    }

    MPI_Status status;
    for (int i = 1; i < size; i++) {
      int size_elems = 0;
      MPI_Recv(&size_elems, 1, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
      std::vector<double> buf(size_elems);
      MPI_Recv(buf.data(), size_elems, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, &status);
      for (double &elem : buf) {
        GetOutput().push_back(elem);
      }
    }

  } else {
    MPI_Status status;
    std::vector<int> buf(2);
    MPI_Recv(buf.data(), 2, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    std::vector<double> elems = SummValues(buf[0], buf[1]);
    int size_elems = static_cast<int>(elems.size());
    MPI_Send(&size_elems, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    MPI_Send(elems.data(), size_elems, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
  }

  int total_size = 0;
  if (rank == 0) {
    total_size = static_cast<int>(GetOutput().size());
  }

  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    GetOutput().resize(total_size);
  }

  MPI_Bcast(GetOutput().data(), total_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  return true;
}

bool SafronovMSumValuesMatrixMPI::PostProcessingImpl() {
  return true;
}

}  // namespace safronov_m_sum_values_matrix
