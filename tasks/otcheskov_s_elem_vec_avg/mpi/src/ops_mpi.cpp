#include "otcheskov_s_elem_vec_avg/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include "otcheskov_s_elem_vec_avg/common/include/common.hpp"

namespace otcheskov_s_elem_vec_avg {

OtcheskovSElemVecAvgMPI::OtcheskovSElemVecAvgMPI(const InType &in) {
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_num_);
  SetTypeOfTask(GetStaticTypeOfTask());
  if (proc_rank_ == 0) {
    GetInput() = in;
  }
  GetOutput() = NAN;
}

bool OtcheskovSElemVecAvgMPI::ValidationImpl() {
  bool is_valid = true;
  if (proc_rank_ == 0) {
    is_valid = !GetInput().empty() && std::isnan(GetOutput());
  }
  MPI_Bcast(&is_valid, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
  return is_valid;
}

bool OtcheskovSElemVecAvgMPI::PreProcessingImpl() {
  return true;
}

bool OtcheskovSElemVecAvgMPI::RunImpl() {
  // передача размера исходного массива
  int total_size = 0;
  if (proc_rank_ == 0) {
    total_size = static_cast<int>(GetInput().size());
  }
  MPI_Bcast(&total_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // распределение данных
  int batch_size = total_size / proc_num_;
  int remainder = total_size % proc_num_;
  int proc_size = batch_size + (proc_rank_ < remainder ? 1 : 0);
  InType local_data(proc_size);
  std::vector<int> displacements;
  std::vector<int> counts;
  if (proc_rank_ == 0) {
    displacements.resize(proc_num_);
    counts.resize(proc_num_);
    int offset = 0;
    for (int i = 0; i < proc_num_; i++) {
      counts[i] = batch_size + (i < remainder ? 1 : 0);
      displacements[i] = offset;
      offset += counts[i];
    }
  }
  MPI_Scatterv(GetInput().data(), counts.data(), displacements.data(), MPI_INT, local_data.data(), proc_size, MPI_INT,
               0, MPI_COMM_WORLD);

  // вычисления среднего элементов вектора
  int64_t local_sum = std::reduce(local_data.begin(), local_data.end(), int64_t{0});
  int64_t total_sum = 0;
  MPI_Allreduce(&local_sum, &total_sum, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  GetOutput() = static_cast<double>(total_sum) / static_cast<double>(total_size);

  return !std::isnan(GetOutput());
}

bool OtcheskovSElemVecAvgMPI::PostProcessingImpl() {
  return true;
}

}  // namespace otcheskov_s_elem_vec_avg
