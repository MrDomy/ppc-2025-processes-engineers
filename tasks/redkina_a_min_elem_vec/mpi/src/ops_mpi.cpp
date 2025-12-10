#include "redkina_a_min_elem_vec/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <vector>

#include "redkina_a_min_elem_vec/common/include/common.hpp"

namespace redkina_a_min_elem_vec {

RedkinaAMinElemVecMPI::RedkinaAMinElemVecMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool RedkinaAMinElemVecMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool RedkinaAMinElemVecMPI::PreProcessingImpl() {
  return true;
}

bool RedkinaAMinElemVecMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(GetInput().size());
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int base = n / size;
  const int rem = n % size;
  const int size_l = (rank < rem) ? base + 1 : base;

  std::vector<int> vec_l(size_l);

  std::vector<int> counts(size, 0);
  std::vector<int> displs(size, 0);

  if (rank == 0) {
    for (int i = 0; i < size; ++i) {
      counts[i] = (i < rem) ? base + 1 : base;
    }
    for (int i = 1; i < size; ++i) {
      displs[i] = displs[i - 1] + counts[i - 1];
    }
  }

  MPI_Scatterv(rank == 0 ? GetInput().data() : nullptr, rank == 0 ? counts.data() : nullptr,
               rank == 0 ? displs.data() : nullptr, MPI_INT, size_l > 0 ? vec_l.data() : nullptr, size_l, MPI_INT, 0,
               MPI_COMM_WORLD);

  int min_l = INT_MAX;
  for (const int v : vec_l) {
    min_l = std::min(min_l, v);
  }

  int min_g = 0;
  MPI_Allreduce(&min_l, &min_g, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  GetOutput() = min_g;
  return true;
}

bool RedkinaAMinElemVecMPI::PostProcessingImpl() {
  return true;
}

}  // namespace redkina_a_min_elem_vec
