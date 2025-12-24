#include "viderman_a_convex_hull/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <queue>
#include <ranges>
#include <unordered_map>
#include <utility>
#include <vector>

#include "viderman_a_convex_hull/common/include/common.hpp"

namespace viderman_a_convex_hull {

namespace {

void ProcessPixelCell(const std::vector<uint8_t> &pixels, std::vector<bool> &visited, int col_idx, int row_idx,
                      int width, int height, int start_row, Component &comp) {
  const size_t idx = (static_cast<size_t>(row_idx) * static_cast<size_t>(width)) + static_cast<size_t>(col_idx);

  if (pixels[idx] != 255 || visited[idx]) {
    return;
  }

  const std::vector<Point> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  std::queue<Point> queue;
  queue.emplace(col_idx, row_idx);
  visited[idx] = true;
  const int global_y = start_row + row_idx;
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
          const int global_ny = start_row + ny;
          comp.pixels.emplace_back(nx, global_ny);
        }
      }
    }
  }
}

void BuildPointToComponentMap(const std::vector<Component> &all_fragments,
                              std::unordered_map<int64_t, int> &point_to_component) {
  const int n = static_cast<int>(all_fragments.size());
  for (int comp_idx = 0; comp_idx < n; ++comp_idx) {
    for (const auto &point : all_fragments[static_cast<size_t>(comp_idx)].pixels) {
      const int64_t hash = (static_cast<int64_t>(point.first) << 32) | static_cast<uint32_t>(point.second);
      point_to_component[hash] = comp_idx;
    }
  }
}

void CheckAndMergeNeighbors(const Point &point, int comp_idx,
                            const std::unordered_map<int64_t, int> &point_to_component, std::vector<int> &parent,
                            std::vector<int> &rank) {
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      if (dx == 0 && dy == 0) {
        continue;
      }

      const Point neighbor = {point.first + dx, point.second + dy};
      const int64_t hash = (static_cast<int64_t>(neighbor.first) << 32) | static_cast<uint32_t>(neighbor.second);
      const auto it = point_to_component.find(hash);

      if (it != point_to_component.end()) {
        const int neighbor_comp_idx = it->second;
        if (comp_idx != neighbor_comp_idx) {
          VidermanAConvexHullMPI::UnionSets(parent, rank, comp_idx, neighbor_comp_idx);
        }
      }
    }
  }
}

std::vector<Point> RemoveCollinearPoints(const std::vector<Point> &pts, const Point &pivot) {
  std::vector<Point> unique_pts;
  unique_pts.push_back(pts[0]);

  for (size_t idx = 1; idx < pts.size(); ++idx) {
    while (idx < pts.size() - 1 && VidermanAConvexHullMPI::CrossProduct(pivot, pts[idx], pts[idx + 1]) == 0) {
      ++idx;
    }
    unique_pts.push_back(pts[idx]);
  }

  return unique_pts;
}

Point FindPivot(std::vector<Point> &pts) {
  size_t pivot_idx = 0;
  for (size_t idx = 1; idx < pts.size(); ++idx) {
    if (pts[idx].second < pts[pivot_idx].second ||
        (pts[idx].second == pts[pivot_idx].second && pts[idx].first < pts[pivot_idx].first)) {
      pivot_idx = idx;
    }
  }
  std::swap(pts[0], pts[pivot_idx]);
  return pts[0];
}

void PrepareComponentDataForBroadcast(const OutType &output, std::vector<int> &component_sizes,
                                      std::vector<int> &displacements, std::vector<int> &all_data) {
  const int total_components = static_cast<int>(output.size());
  int total_data_size = 0;

  for (int comp_idx = 0; comp_idx < total_components; ++comp_idx) {
    const size_t base_idx = static_cast<size_t>(comp_idx) * 2U;
    component_sizes[base_idx] = static_cast<int>(output[static_cast<size_t>(comp_idx)].pixels.size());
    component_sizes[base_idx + 1U] = static_cast<int>(output[static_cast<size_t>(comp_idx)].hull.size());

    const int pixels_data_size = component_sizes[base_idx] * 2;
    const int hull_data_size = component_sizes[base_idx + 1U] * 2;
    displacements[static_cast<size_t>(comp_idx) + 1U] =
        displacements[static_cast<size_t>(comp_idx)] + pixels_data_size + hull_data_size;
    total_data_size = displacements[static_cast<size_t>(comp_idx) + 1U];
  }

  all_data.resize(static_cast<size_t>(total_data_size));
  int offset = 0;
  for (int comp_idx = 0; comp_idx < total_components; ++comp_idx) {
    for (const auto &point : output[static_cast<size_t>(comp_idx)].pixels) {
      all_data[static_cast<size_t>(offset)] = point.first;
      all_data[static_cast<size_t>(offset) + 1U] = point.second;
      offset += 2;
    }
    for (const auto &point : output[static_cast<size_t>(comp_idx)].hull) {
      all_data[static_cast<size_t>(offset)] = point.first;
      all_data[static_cast<size_t>(offset) + 1U] = point.second;
      offset += 2;
    }
  }
}

void ReconstructComponentsFromData(OutType &output, int total_components, const std::vector<int> &component_sizes,
                                   const std::vector<int> &all_data) {
  output.clear();
  output.reserve(static_cast<size_t>(total_components));

  int data_offset = 0;
  for (int comp_idx = 0; comp_idx < total_components; ++comp_idx) {
    Component comp;

    const size_t base_idx = static_cast<size_t>(comp_idx) * 2U;
    const int num_pixels = component_sizes[base_idx];
    comp.pixels.resize(static_cast<size_t>(num_pixels));
    for (int point_idx = 0; point_idx < num_pixels; ++point_idx) {
      comp.pixels[static_cast<size_t>(point_idx)] = {all_data[static_cast<size_t>(data_offset)],
                                                     all_data[static_cast<size_t>(data_offset) + 1U]};
      data_offset += 2;
    }

    const int num_hull = component_sizes[base_idx + 1U];
    comp.hull.resize(static_cast<size_t>(num_hull));
    for (int hull_idx = 0; hull_idx < num_hull; ++hull_idx) {
      comp.hull[static_cast<size_t>(hull_idx)] = {all_data[static_cast<size_t>(data_offset)],
                                                  all_data[static_cast<size_t>(data_offset) + 1U]};
      data_offset += 2;
    }

    output.push_back(std::move(comp));
  }
}

}  // namespace

VidermanAConvexHullMPI::VidermanAConvexHullMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
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
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);

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

  const std::vector<uint8_t> *input_pixels = (rank_ == 0) ? &input.pixels : nullptr;
  MPI_Scatterv(input_pixels != nullptr ? input_pixels->data() : nullptr, send_counts.data(), displacements.data(),
               MPI_UINT8_T, local_image_.pixels.data(), static_cast<int>(local_pixel_count), MPI_UINT8_T, 0,
               MPI_COMM_WORLD);
}

void VidermanAConvexHullMPI::FindLocalComponents() {
  local_components_.clear();

  const int width = width_;
  const int height = local_rows_;
  const auto &pixels = local_image_.pixels;

  std::vector<bool> visited(static_cast<size_t>(height) * static_cast<size_t>(width), false);

  for (int row_idx = 0; row_idx < height; ++row_idx) {
    for (int col_idx = 0; col_idx < width; ++col_idx) {
      Component comp;
      ProcessPixelCell(pixels, visited, col_idx, row_idx, width, height, start_row_, comp);
      if (!comp.pixels.empty()) {
        local_components_.push_back(std::move(comp));
      }
    }
  }
}

void VidermanAConvexHullMPI::GatherComponentCounts(std::vector<int> &comp_counts, std::vector<int> &displacements,
                                                   int &total_comps) {
  const int local_comp_count = static_cast<int>(local_components_.size());
  MPI_Gather(&local_comp_count, 1, MPI_INT, comp_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    for (int i = 1; i < size_; ++i) {
      displacements[static_cast<size_t>(i)] =
          displacements[static_cast<size_t>(i - 1)] + comp_counts[static_cast<size_t>(i - 1)];
    }
    total_comps = displacements[static_cast<size_t>(size_ - 1)] + comp_counts[static_cast<size_t>(size_ - 1)];
  }
}

void VidermanAConvexHullMPI::GatherComponentSizes(const std::vector<int> &comp_counts,
                                                  const std::vector<int> &displacements,
                                                  std::vector<int> &all_comp_sizes) {
  std::vector<int> local_comp_sizes;
  local_comp_sizes.reserve(static_cast<size_t>(local_components_.size()));
  for (const auto &comp : local_components_) {
    local_comp_sizes.push_back(static_cast<int>(comp.pixels.size()));
  }

  const int local_comp_count = static_cast<int>(local_components_.size());
  std::vector<int> recvcounts_comp(static_cast<size_t>(size_), 0);
  if (rank_ == 0) {
    for (int i = 0; i < size_; ++i) {
      recvcounts_comp[static_cast<size_t>(i)] = comp_counts[static_cast<size_t>(i)];
    }
  }

  MPI_Gatherv(local_comp_sizes.data(), local_comp_count, MPI_INT, rank_ == 0 ? all_comp_sizes.data() : nullptr,
              recvcounts_comp.data(), displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);
}

std::vector<int> VidermanAConvexHullMPI::PackLocalPoints() {
  std::vector<int> local_points;
  int total_points = 0;
  for (const auto &comp : local_components_) {
    total_points += static_cast<int>(comp.pixels.size());
  }

  local_points.reserve(static_cast<size_t>(total_points) * 2U);
  for (const auto &comp : local_components_) {
    for (const auto &point : comp.pixels) {
      local_points.push_back(point.first);
      local_points.push_back(point.second);
    }
  }
  return local_points;
}

void VidermanAConvexHullMPI::ComputeDisplacementsAndCounts(const std::vector<int> &comp_counts,
                                                           const std::vector<int> &displacements,
                                                           const std::vector<int> &all_comp_sizes,
                                                           std::vector<int> &point_displacements,
                                                           std::vector<int> &point_recvcounts) const {
  point_displacements.assign(static_cast<size_t>(size_), 0);
  point_recvcounts.assign(static_cast<size_t>(size_), 0);

  if (rank_ != 0) {
    return;
  }

  int offset = 0;
  for (int proc = 0; proc < size_; ++proc) {
    point_displacements[static_cast<size_t>(proc)] = offset * 2;
    int proc_points = 0;
    for (int i = 0; i < comp_counts[static_cast<size_t>(proc)]; ++i) {
      int idx = displacements[static_cast<size_t>(proc)] + i;
      proc_points += all_comp_sizes[static_cast<size_t>(idx)];
    }
    point_recvcounts[static_cast<size_t>(proc)] = proc_points * 2;
    offset += proc_points;
  }
}

void VidermanAConvexHullMPI::UnpackAllPoints(const std::vector<int> &all_points_data,
                                             const std::vector<int> &comp_counts, const std::vector<int> &displacements,
                                             const std::vector<int> &all_comp_sizes) {
  all_fragments_.clear();
  int points_offset = 0;

  for (int proc = 0; proc < size_; ++proc) {
    for (int comp_idx = 0; comp_idx < comp_counts[static_cast<size_t>(proc)]; ++comp_idx) {
      Component comp;
      int idx = displacements[static_cast<size_t>(proc)] + comp_idx;
      int comp_size = all_comp_sizes[static_cast<size_t>(idx)];
      comp.pixels.resize(static_cast<size_t>(comp_size));

      for (int point_idx = 0; point_idx < comp_size; ++point_idx) {
        comp.pixels[static_cast<size_t>(point_idx)] = {all_points_data[static_cast<size_t>(points_offset)],
                                                       all_points_data[static_cast<size_t>(points_offset) + 1]};
        points_offset += 2;
      }
      all_fragments_.push_back(std::move(comp));
    }
  }
}

void VidermanAConvexHullMPI::GatherComponentPoints(const std::vector<int> &comp_counts,
                                                   const std::vector<int> &displacements,
                                                   const std::vector<int> &all_comp_sizes, int total_points) {
  auto local_points_data = PackLocalPoints();

  std::vector<int> point_displacements(static_cast<size_t>(size_), 0);
  std::vector<int> point_recvcounts(static_cast<size_t>(size_), 0);
  ComputeDisplacementsAndCounts(comp_counts, displacements, all_comp_sizes, point_displacements, point_recvcounts);

  std::vector<int> all_points_data;
  if (rank_ == 0) {
    all_points_data.resize(static_cast<size_t>(total_points) * 2);
  }

  MPI_Bcast(point_recvcounts.data(), size_, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(point_displacements.data(), size_, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Gatherv(local_points_data.data(), static_cast<int>(local_points_data.size()), MPI_INT,
              rank_ == 0 ? all_points_data.data() : nullptr, point_recvcounts.data(), point_displacements.data(),
              MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    UnpackAllPoints(all_points_data, comp_counts, displacements, all_comp_sizes);
  }
}

void VidermanAConvexHullMPI::GatherAllFragments() {
  std::vector<int> comp_counts(static_cast<size_t>(size_), 0);
  std::vector<int> displacements(static_cast<size_t>(size_), 0);
  int total_comps = 0;

  GatherComponentCounts(comp_counts, displacements, total_comps);

  std::vector<int> all_comp_sizes;
  if (rank_ == 0) {
    all_comp_sizes.resize(static_cast<size_t>(total_comps));
  }

  GatherComponentSizes(comp_counts, displacements, all_comp_sizes);

  int total_points = 0;
  if (rank_ == 0) {
    int offset = 0;
    for (int proc = 0; proc < size_; ++proc) {
      for (int i = 0; i < comp_counts[static_cast<size_t>(proc)]; ++i) {
        const int idx = displacements[static_cast<size_t>(proc)] + i;
        offset += all_comp_sizes[static_cast<size_t>(idx)];
      }
    }
    total_points = offset * 2;
  }

  GatherComponentPoints(comp_counts, displacements, all_comp_sizes, total_points);
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
      ++rank[static_cast<size_t>(root_x)];
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
  std::ranges::iota(parent, 0);

  std::unordered_map<int64_t, int> point_to_component;
  BuildPointToComponentMap(all_fragments_, point_to_component);

  for (int comp_idx = 0; comp_idx < n; ++comp_idx) {
    for (const auto &point : all_fragments_[static_cast<size_t>(comp_idx)].pixels) {
      CheckAndMergeNeighbors(point, comp_idx, point_to_component, parent, rank);
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
    merged_components_.push_back(std::move(merged_comp));
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
  return (static_cast<int64_t>(a.first - o.first) * static_cast<int64_t>(b.second - o.second)) -
         (static_cast<int64_t>(a.second - o.second) * static_cast<int64_t>(b.first - o.first));
}

std::vector<Point> VidermanAConvexHullMPI::GrahamScan(const std::vector<Point> &points) {
  if (points.size() <= 3) {
    return points;
  }

  std::vector<Point> pts = points;
  const Point pivot = FindPivot(pts);

  std::ranges::sort(pts.begin() + 1, pts.end(), [&pivot](const Point &a, const Point &b) {
    const int64_t cross = CrossProduct(pivot, a, b);
    if (cross != 0) {
      return cross > 0;
    }

    const int64_t dist_a =
        (static_cast<int64_t>(a.first - pivot.first) * static_cast<int64_t>(a.first - pivot.first)) +
        (static_cast<int64_t>(a.second - pivot.second) * static_cast<int64_t>(a.second - pivot.second));
    const int64_t dist_b =
        (static_cast<int64_t>(b.first - pivot.first) * static_cast<int64_t>(b.first - pivot.first)) +
        (static_cast<int64_t>(b.second - pivot.second) * static_cast<int64_t>(b.second - pivot.second));
    return dist_a < dist_b;
  });

  std::vector<Point> unique_pts = RemoveCollinearPoints(pts, pivot);

  if (unique_pts.size() <= 3) {
    return unique_pts;
  }

  std::vector<Point> hull;
  hull.reserve(unique_pts.size());

  hull.push_back(unique_pts[0]);
  hull.push_back(unique_pts[1]);
  hull.push_back(unique_pts[2]);

  for (size_t idx = 3; idx < unique_pts.size(); ++idx) {
    while (hull.size() >= 2 && CrossProduct(hull[hull.size() - 2U], hull.back(), unique_pts[idx]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(unique_pts[idx]);
  }

  return hull;
}

void VidermanAConvexHullMPI::BuildAllConvexHullsOnRank0() {
  OutType final_result;

  for (auto &component : merged_components_) {
    if (component.pixels.size() >= 3U) {
      component.hull = GrahamScan(component.pixels);
    } else {
      component.hull = component.pixels;
    }
    final_result.push_back(std::move(component));
  }

  GetOutput() = std::move(final_result);
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

  std::vector<int> component_sizes(static_cast<size_t>(total_components) * 2U, 0);
  std::vector<int> displacements(static_cast<size_t>(total_components) + 1U, 0);
  std::vector<int> all_data;

  if (rank_ == 0) {
    PrepareComponentDataForBroadcast(output, component_sizes, displacements, all_data);
  }

  MPI_Bcast(component_sizes.data(), total_components * 2, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displacements.data(), total_components + 1, MPI_INT, 0, MPI_COMM_WORLD);

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
    ReconstructComponentsFromData(output, total_components, component_sizes, all_data);
  }
}

}  // namespace viderman_a_convex_hull
