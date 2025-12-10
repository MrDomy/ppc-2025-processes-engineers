#include "sabirov_s_min_val_matrix/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "sabirov_s_min_val_matrix/common/include/common.hpp"

namespace sabirov_s_min_val_matrix {

SabirovSMinValMatrixMPI::SabirovSMinValMatrixMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput().clear();
}

bool SabirovSMinValMatrixMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput().empty());
}

bool SabirovSMinValMatrixMPI::PreProcessingImpl() {
  GetOutput().clear();
  GetOutput().reserve(GetInput());
  return true;
}

bool SabirovSMinValMatrixMPI::RunImpl() {
  InType n = GetInput();
  if (n == 0) {
    return false;
  }

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Распределяем строки между процессами
  int rows_per_proc = n / size;
  int remainder = n % size;

  int start_row = (rank * rows_per_proc) + std::min(rank, remainder);
  int num_local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
  int end_row = start_row + num_local_rows;

  // Каждый процесс находит минимумы для своих строк
  std::vector<InType> local_mins;
  local_mins.reserve(num_local_rows);

  auto generate_value = [](int64_t i, int64_t j) -> InType {
    constexpr int64_t kA = 1103515245LL;
    constexpr int64_t kC = 12345LL;
    constexpr int64_t kM = 2147483648LL;
    int64_t seed = ((i % kM) * (100000007LL % kM) + (j % kM) * (1000000009LL % kM)) % kM;
    seed = (seed ^ 42LL) % kM;
    int64_t val = ((kA % kM) * (seed % kM) + kC) % kM;
    return static_cast<InType>((val % 2000001LL) - 1000000LL);
  };

  for (InType i = start_row; i < end_row; i++) {
    InType min_val = generate_value(static_cast<int64_t>(i), 0);
    for (InType j = 1; j < n; j++) {
      InType val = generate_value(static_cast<int64_t>(i), static_cast<int64_t>(j));
      min_val = std::min(min_val, val);
    }
    local_mins.push_back(min_val);
  }

  std::vector<int> recvcounts(size);
  std::vector<int> displs(size);

  for (int i = 0; i < size; i++) {
    int proc_rows = rows_per_proc + (i < remainder ? 1 : 0);
    recvcounts[i] = proc_rows;
    displs[i] = (i * rows_per_proc) + std::min(i, remainder);
  }

  GetOutput().resize(n);

  // Собираем результаты на процессе 0
  if (rank == 0) {
    MPI_Gatherv(local_mins.data(), num_local_rows, MPI_INT, GetOutput().data(), recvcounts.data(), displs.data(),
                MPI_INT, 0, MPI_COMM_WORLD);
  } else {
    MPI_Gatherv(local_mins.data(), num_local_rows, MPI_INT, nullptr, recvcounts.data(), displs.data(), MPI_INT, 0,
                MPI_COMM_WORLD);
  }

  // Рассылаем результат всем процессам
  MPI_Bcast(GetOutput().data(), n, MPI_INT, 0, MPI_COMM_WORLD);

  return !GetOutput().empty() && (GetOutput().size() == static_cast<size_t>(n));
}

bool SabirovSMinValMatrixMPI::PostProcessingImpl() {
  return !GetOutput().empty() && (GetOutput().size() == static_cast<size_t>(GetInput()));
}

}  // namespace sabirov_s_min_val_matrix
