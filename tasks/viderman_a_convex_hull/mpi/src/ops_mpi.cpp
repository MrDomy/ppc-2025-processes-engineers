#include "viderman_a_convex_hull/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

#include "viderman_a_convex_hull/common/include/common.hpp"

namespace viderman_a_convex_hull {

VidermanAConvexHullMPI::VidermanAConvexHullMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
}

bool VidermanAConvexHullMPI::ValidationImpl() {
  const auto &input = GetInput();
  if (input.width <= 0 || input.height <= 0) {
    return false;
  }

  const int64_t total_pixels = static_cast<int64_t>(input.width) * static_cast<int64_t>(input.height);
  if (static_cast<size_t>(total_pixels) != input.pixels.size()) {
    return false;
  }

  return std::ranges::all_of(input.pixels, [](const auto &pixel) { return pixel == 0 || pixel == 255; });
}

bool VidermanAConvexHullMPI::PreProcessingImpl() {
  return true;
}

bool VidermanAConvexHullMPI::RunImpl() {
  DistributeImageData();
  FindLocalComponents();
  GatherAllFragments();

  if (rank_ == 0) {
    MergeFragmentsOnRank0();
    BuildAllConvexHullsOnRank0();
  }
  BroadcastFinalResult();

  return true;
}

bool VidermanAConvexHullMPI::PostProcessingImpl() {
  return true;
}

void VidermanAConvexHullMPI::DistributeImageData() {
  const auto &input = GetInput();
  if (rank_ == 0) {
    width_ = input.width;
    height_ = input.height;
  }
  MPI_Bcast(&width_, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&height_, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const int rows_per_proc = height_ / size_;
  const int remainder = height_ % size_;
  start_row_ = 0;
  for (int proc_index = 0; proc_index < rank_; ++proc_index) {
    start_row_ += rows_per_proc + (proc_index < remainder ? 1 : 0);
  }
  end_row_ = start_row_ + rows_per_proc + (rank_ < remainder ? 1 : 0);
  local_rows_ = end_row_ - start_row_;

  std::vector<int> send_counts(static_cast<size_t>(size_), 0);
  std::vector<int> displacements(static_cast<size_t>(size_), 0);

  if (rank_ == 0) {
    int current_offset = 0;
    for (int proc_index = 0; proc_index < size_; ++proc_index) {
      const int proc_rows = rows_per_proc + (proc_index < remainder ? 1 : 0);
      send_counts[static_cast<size_t>(proc_index)] = proc_rows * width_;
      displacements[static_cast<size_t>(proc_index)] = current_offset * width_;
      current_offset += proc_rows;
    }
  }

  local_image_.width = width_;
  local_image_.height = local_rows_;
  const size_t local_pixel_count = static_cast<size_t>(local_rows_) * static_cast<size_t>(width_);
  local_image_.pixels.resize(local_pixel_count);

  MPI_Scatterv(rank_ == 0 ? input.pixels.data() : nullptr, send_counts.data(), displacements.data(), MPI_UINT8_T,
               local_image_.pixels.data(), static_cast<int>(local_pixel_count), MPI_UINT8_T, 0, MPI_COMM_WORLD);
}

void VidermanAConvexHullMPI::FindLocalComponents() {
  local_components_.clear();

  const int width = width_;
  const int height = local_rows_;
  const auto &pixels = local_image_.pixels;

  std::vector<bool> visited((static_cast<size_t>(height) * static_cast<size_t>(width)), false);
  const std::vector<Point> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

  for (int row_idx = 0; row_idx < height; ++row_idx) {
    for (int col_idx = 0; col_idx < width; ++col_idx) {
      const size_t idx = (static_cast<size_t>(row_idx) * static_cast<size_t>(width)) + static_cast<size_t>(col_idx);

      if (pixels[idx] == 255 && !visited[idx]) {
        Component comp;
        std::queue<Point> queue;
        queue.emplace(col_idx, row_idx);
        visited[idx] = true;
        const int global_y = start_row_ + row_idx;
        comp.pixels.emplace_back(col_idx, global_y);

        while (!queue.empty()) {
          const Point current = queue.front();
          queue.pop();

          for (const auto &dir : directions) {
            const int nx = current.first + dir.first;
            const int ny = current.second + dir.second;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
              const size_t nidx = (static_cast<size_t>(ny) * static_cast<size_t>(width)) + static_cast<size_t>(nx);
              if (pixels[nidx] == 255 && !visited[nidx]) {
                visited[nidx] = true;
                queue.emplace(nx, ny);
                const int global_ny = start_row_ + ny;
                comp.pixels.emplace_back(nx, global_ny);
              }
            }
          }
        }

        if (!comp.pixels.empty()) {
          local_components_.push_back(std::move(comp));
        }
      }
    }
  }
}

void VidermanAConvexHullMPI::GatherAllFragments() {
  const int local_comp_count = static_cast<int>(local_components_.size());
  std::vector<int> comp_counts(static_cast<size_t>(size_), 0);
  MPI_Gather(&local_comp_count, 1, MPI_INT, comp_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> displacements(static_cast<size_t>(size_), 0);
  int total_comps = 0;

  if (rank_ == 0) {
    for (int i = 1; i < size_; ++i) {
      displacements[static_cast<size_t>(i)] =
          displacements[static_cast<size_t>(i - 1)] + comp_counts[static_cast<size_t>(i - 1)];
    }
    total_comps = displacements[static_cast<size_t>(size_ - 1)] + comp_counts[static_cast<size_t>(size_ - 1)];
  }

  std::vector<int> local_comp_sizes;
  local_comp_sizes.reserve(static_cast<size_t>(local_comp_count));
  for (const auto &comp : local_components_) {
    local_comp_sizes.push_back(static_cast<int>(comp.pixels.size()));
  }

  std::vector<int> all_comp_sizes;
  if (rank_ == 0) {
    all_comp_sizes.resize(static_cast<size_t>(total_comps));
  }

  std::vector<int> recvcounts_comp(static_cast<size_t>(size_), 0);
  if (rank_ == 0) {
    for (int i = 0; i < size_; ++i) {
      recvcounts_comp[static_cast<size_t>(i)] = comp_counts[static_cast<size_t>(i)];
    }
  }

  MPI_Gatherv(local_comp_sizes.data(), local_comp_count, MPI_INT, rank_ == 0 ? all_comp_sizes.data() : nullptr,
              recvcounts_comp.data(), displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> point_displacements(static_cast<size_t>(size_), 0);
  std::vector<int> point_recvcounts(static_cast<size_t>(size_), 0);
  int total_points = 0;

  if (rank_ == 0) {
    int offset = 0;
    for (int proc = 0; proc < size_; ++proc) {
      point_displacements[static_cast<size_t>(proc)] = offset * 2;
      int proc_points = 0;
      for (int i = 0; i < comp_counts[static_cast<size_t>(proc)]; ++i) {
        const int idx = displacements[static_cast<size_t>(proc)] + i;
        proc_points += all_comp_sizes[static_cast<size_t>(idx)];
      }
      point_recvcounts[static_cast<size_t>(proc)] = proc_points * 2;
      offset += proc_points;
      total_points = offset * 2;
    }
  }

  std::vector<int> local_points_data;
  int local_total_points = 0;
  for (const auto &comp : local_components_) {
    local_total_points += static_cast<int>(comp.pixels.size());
  }

  local_points_data.reserve(static_cast<size_t>(local_total_points) * 2);
  for (const auto &comp : local_components_) {
    for (const auto &point : comp.pixels) {
      local_points_data.push_back(point.first);
      local_points_data.push_back(point.second);
    }
  }

  std::vector<int> all_points_data;
  if (rank_ == 0) {
    all_points_data.resize(static_cast<size_t>(total_points));
  }

  MPI_Bcast(point_recvcounts.data(), size_, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(point_displacements.data(), size_, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Gatherv(local_points_data.data(), local_total_points * 2, MPI_INT, rank_ == 0 ? all_points_data.data() : nullptr,
              point_recvcounts.data(), point_displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    all_fragments_.clear();
    int points_offset = 0;

    for (int proc = 0; proc < size_; ++proc) {
      for (int comp_idx = 0; comp_idx < comp_counts[static_cast<size_t>(proc)]; ++comp_idx) {
        Component comp;
        const int comp_size = all_comp_sizes[static_cast<size_t>(displacements[static_cast<size_t>(proc)] + comp_idx)];
        comp.pixels.resize(static_cast<size_t>(comp_size));

        for (int point_idx = 0; point_idx < comp_size; ++point_idx) {
          comp.pixels[static_cast<size_t>(point_idx)] = {all_points_data[static_cast<size_t>(points_offset)],
                                                         all_points_data[static_cast<size_t>(points_offset + 1)]};
          points_offset += 2;
        }

        all_fragments_.push_back(std::move(comp));
      }
    }
  }
}

int64_t VidermanAConvexHullMPI::PointToHash(const Point &p) {
  return (static_cast<int64_t>(p.first) << 32) | static_cast<uint32_t>(p.second);
}

int VidermanAConvexHullMPI::FindRoot(std::vector<int> &parent, int x) {
  while (parent[static_cast<size_t>(x)] != x) {
    parent[static_cast<size_t>(x)] = parent[static_cast<size_t>(parent[static_cast<size_t>(x)])];
    x = parent[static_cast<size_t>(x)];
  }
  return x;
}

void VidermanAConvexHullMPI::UnionSets(std::vector<int> &parent, std::vector<int> &rank, int x, int y) {
  int root_x = FindRoot(parent, x);
  int root_y = FindRoot(parent, y);

  if (root_x != root_y) {
    if (rank[static_cast<size_t>(root_x)] < rank[static_cast<size_t>(root_y)]) {
      parent[static_cast<size_t>(root_x)] = root_y;
    } else if (rank[static_cast<size_t>(root_x)] > rank[static_cast<size_t>(root_y)]) {
      parent[static_cast<size_t>(root_y)] = root_x;
    } else {
      parent[static_cast<size_t>(root_y)] = root_x;
      rank[static_cast<size_t>(root_x)]++;
    }
  }
}

void VidermanAConvexHullMPI::MergeFragmentsOnRank0() {
  if (all_fragments_.empty()) {
    return;
  }

  const int n = static_cast<int>(all_fragments_.size());
  std::vector<int> parent(static_cast<size_t>(n));
  std::vector<int> rank(static_cast<size_t>(n), 0);
  std::iota(parent.begin(), parent.end(), 0);

  std::unordered_map<int64_t, int> point_to_component;

  for (int comp_idx = 0; comp_idx < n; ++comp_idx) {
    for (const auto &point : all_fragments_[static_cast<size_t>(comp_idx)].pixels) {
      point_to_component[PointToHash(point)] = comp_idx;
    }
  }

  for (int comp_idx = 0; comp_idx < n; ++comp_idx) {
    for (const auto &point : all_fragments_[static_cast<size_t>(comp_idx)].pixels) {
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          if (dx == 0 && dy == 0) {
            continue;
          }

          const Point neighbor = {point.first + dx, point.second + dy};
          const auto it = point_to_component.find(PointToHash(neighbor));

          if (it != point_to_component.end()) {
            const int neighbor_comp_idx = it->second;
            if (comp_idx != neighbor_comp_idx) {
              UnionSets(parent, rank, comp_idx, neighbor_comp_idx);
            }
          }
        }
      }
    }
  }

  std::unordered_map<int, std::vector<Point>> merged_points;
  for (int idx = 0; idx < n; ++idx) {
    const int root = FindRoot(parent, idx);
    merged_points[root].insert(merged_points[root].end(), all_fragments_[static_cast<size_t>(idx)].pixels.begin(),
                               all_fragments_[static_cast<size_t>(idx)].pixels.end());
  }

  merged_components_.clear();
  for (auto &[root_id, points] : merged_points) {
    Component merged_comp;
    RemoveDuplicatePoints(points);
    merged_comp.pixels = std::move(points);
    merged_components_.push_back(merged_comp);
  }
}

void VidermanAConvexHullMPI::RemoveDuplicatePoints(std::vector<Point> &points) {
  if (points.empty()) {
    return;
  }
  std::ranges::sort(points);
  const auto last = std::ranges::unique(points).begin();
  points.erase(last, points.end());
}

int64_t VidermanAConvexHullMPI::CrossProduct(const Point &o, const Point &a, const Point &b) {
  return ((static_cast<int64_t>(a.first - o.first) * static_cast<int64_t>(b.second - o.second)) -
          (static_cast<int64_t>(a.second - o.second) * static_cast<int64_t>(b.first - o.first)));
}

std::vector<Point> VidermanAConvexHullMPI::GrahamScan(const std::vector<Point> &points) {
  if (points.size() <= 3) {
    return points;
  }

  std::vector<Point> pts = points;

  size_t pivot_idx = 0;
  for (size_t idx = 1; idx < pts.size(); ++idx) {
    if (pts[idx].second < pts[pivot_idx].second ||
        (pts[idx].second == pts[pivot_idx].second && pts[idx].first < pts[pivot_idx].first)) {
      pivot_idx = idx;
    }
  }

  std::swap(pts[0], pts[pivot_idx]);
  const Point pivot = pts[0];

  std::ranges::sort(pts.begin() + 1, pts.end(), [&pivot](const Point &a, const Point &b) {
    const int64_t cross = CrossProduct(pivot, a, b);
    if (cross != 0) {
      return cross > 0;
    }

    const int64_t dist_a =
        ((static_cast<int64_t>(a.first - pivot.first) * static_cast<int64_t>(a.first - pivot.first)) +
         (static_cast<int64_t>(a.second - pivot.second) * static_cast<int64_t>(a.second - pivot.second)));
    const int64_t dist_b =
        ((static_cast<int64_t>(b.first - pivot.first) * static_cast<int64_t>(b.first - pivot.first)) +
         (static_cast<int64_t>(b.second - pivot.second) * static_cast<int64_t>(b.second - pivot.second)));
    return dist_a < dist_b;
  });

  std::vector<Point> unique_pts;
  unique_pts.push_back(pts[0]);

  for (size_t idx = 1; idx < pts.size(); ++idx) {
    while (idx < pts.size() - 1 && CrossProduct(pivot, pts[idx], pts[idx + 1]) == 0) {
      ++idx;
    }
    unique_pts.push_back(pts[idx]);
  }

  if (unique_pts.size() <= 3) {
    return unique_pts;
  }

  std::vector<Point> hull;
  hull.reserve(unique_pts.size());

  hull.push_back(unique_pts[0]);
  hull.push_back(unique_pts[1]);
  hull.push_back(unique_pts[2]);

  for (size_t idx = 3; idx < unique_pts.size(); ++idx) {
    while (hull.size() >= 2 && CrossProduct(hull[hull.size() - 2], hull.back(), unique_pts[idx]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(unique_pts[idx]);
  }

  return hull;
}

void VidermanAConvexHullMPI::BuildAllConvexHullsOnRank0() {
  OutType final_result;

  for (auto &component : merged_components_) {
    if (component.pixels.size() >= 3) {
      component.hull = GrahamScan(component.pixels);
    } else {
      component.hull = component.pixels;
    }
    final_result.push_back(component);
  }

  GetOutput() = final_result;
}

void VidermanAConvexHullMPI::BroadcastFinalResult() {
  OutType &output = GetOutput();

  int total_components = 0;
  if (rank_ == 0) {
    total_components = static_cast<int>(output.size());
  }

  MPI_Bcast(&total_components, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (total_components == 0) {
    if (rank_ != 0) {
      output.clear();
    }
    return;
  }

  std::vector<int> component_sizes(static_cast<size_t>(total_components) * 2);
  std::vector<int> displacements(static_cast<size_t>(total_components) + 1, 0);
  std::vector<int> all_data;

  if (rank_ == 0) {
    int total_data_size = 0;
    for (int comp_idx = 0; comp_idx < total_components; ++comp_idx) {
      component_sizes[static_cast<size_t>(comp_idx) * 2] =
          static_cast<int>(output[static_cast<size_t>(comp_idx)].pixels.size());
      component_sizes[static_cast<size_t>(comp_idx) * 2 + 1] =
          static_cast<int>(output[static_cast<size_t>(comp_idx)].hull.size());
      displacements[static_cast<size_t>(comp_idx) + 1] = displacements[static_cast<size_t>(comp_idx)] +
                                                         (component_sizes[static_cast<size_t>(comp_idx) * 2] * 2) +
                                                         (component_sizes[static_cast<size_t>(comp_idx) * 2 + 1] * 2);
      total_data_size = displacements[static_cast<size_t>(comp_idx) + 1];
    }

    all_data.resize(static_cast<size_t>(total_data_size));
    int offset = 0;
    for (int comp_idx = 0; comp_idx < total_components; ++comp_idx) {
      for (const auto &point : output[static_cast<size_t>(comp_idx)].pixels) {
        all_data[static_cast<size_t>(offset++)] = point.first;
        all_data[static_cast<size_t>(offset++)] = point.second;
      }
      for (const auto &point : output[static_cast<size_t>(comp_idx)].hull) {
        all_data[static_cast<size_t>(offset++)] = point.first;
        all_data[static_cast<size_t>(offset++)] = point.second;
      }
    }
  }

  MPI_Bcast(component_sizes.data(), static_cast<int>(total_components * 2), MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displacements.data(), static_cast<int>(total_components + 1), MPI_INT, 0, MPI_COMM_WORLD);

  int total_data_size = 0;
  if (rank_ == 0) {
    total_data_size = displacements[static_cast<size_t>(total_components)];
  }
  MPI_Bcast(&total_data_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ != 0) {
    all_data.resize(static_cast<size_t>(total_data_size));
  }

  MPI_Bcast(all_data.data(), total_data_size, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ != 0) {
    output.clear();
    output.reserve(static_cast<size_t>(total_components));

    int data_offset = 0;
    for (int comp_idx = 0; comp_idx < total_components; ++comp_idx) {
      Component comp;

      const int num_pixels = component_sizes[static_cast<size_t>(comp_idx) * 2];
      comp.pixels.resize(static_cast<size_t>(num_pixels));
      for (int point_idx = 0; point_idx < num_pixels; ++point_idx) {
        comp.pixels[static_cast<size_t>(point_idx)] = {all_data[static_cast<size_t>(data_offset)],
                                                       all_data[static_cast<size_t>(data_offset + 1)]};
        data_offset += 2;
      }

      const int num_hull = component_sizes[static_cast<size_t>(comp_idx) * 2 + 1];
      comp.hull.resize(static_cast<size_t>(num_hull));
      for (int hull_idx = 0; hull_idx < num_hull; ++hull_idx) {
        comp.hull[static_cast<size_t>(hull_idx)] = {all_data[static_cast<size_t>(data_offset)],
                                                    all_data[static_cast<size_t>(data_offset + 1)]};
        data_offset += 2;
      }

      output.push_back(comp);
    }
  }
}

}  // namespace viderman_a_convex_hull
