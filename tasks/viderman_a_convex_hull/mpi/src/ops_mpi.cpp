#include "viderman_a_convex_hull/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "util/include/util.hpp"
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

  if (static_cast<size_t>(input.width * input.height) != input.pixels.size()) {
    return false;
  }
  for (const auto &pixel : input.pixels) {
    if (pixel != 0 && pixel != 255) {
      return false;
    }
  }

  return true;
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
  int rows_per_proc = height_ / size_;
  int remainder = height_ % size_;
  start_row_ = 0;
  for (int i = 0; i < rank_; ++i) {
    start_row_ += rows_per_proc + (i < remainder ? 1 : 0);
  }
  end_row_ = start_row_ + rows_per_proc + (rank_ < remainder ? 1 : 0);
  local_rows_ = end_row_ - start_row_;
  std::vector<int> send_counts(size_, 0);
  std::vector<int> displacements(size_, 0);

  if (rank_ == 0) {
    int current_offset = 0;
    for (int i = 0; i < size_; ++i) {
      int proc_rows = rows_per_proc + (i < remainder ? 1 : 0);
      send_counts[i] = proc_rows * width_;
      displacements[i] = current_offset * width_;
      current_offset += proc_rows;
    }
  }
  local_image_.width = width_;
  local_image_.height = local_rows_;
  size_t local_pixel_count = static_cast<size_t>(local_rows_) * static_cast<size_t>(width_);
  local_image_.pixels.resize(local_pixel_count);
  MPI_Scatterv(rank_ == 0 ? input.pixels.data() : nullptr, send_counts.data(), displacements.data(), MPI_UINT8_T,
               local_image_.pixels.data(), static_cast<int>(local_pixel_count), MPI_UINT8_T, 0, MPI_COMM_WORLD);
}

void VidermanAConvexHullMPI::FindLocalComponents() {
  local_components_.clear();

  int width = width_;
  int height = local_rows_;
  const auto &pixels = local_image_.pixels;
  std::vector<bool> visited(static_cast<size_t>(height) * static_cast<size_t>(width), false);
  const std::vector<Point> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
      if (pixels[idx] == 255 && !visited[idx]) {
        Component comp;
        std::queue<Point> queue;
        queue.emplace(x, y);
        visited[idx] = true;
        int global_y = start_row_ + y;
        comp.pixels.emplace_back(x, global_y);
        while (!queue.empty()) {
          Point current = queue.front();
          queue.pop();
          for (const auto &dir : directions) {
            int nx = current.first + dir.first;
            int ny = current.second + dir.second;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
              size_t nidx = static_cast<size_t>(ny) * static_cast<size_t>(width) + static_cast<size_t>(nx);

              if (pixels[nidx] == 255 && !visited[nidx]) {
                visited[nidx] = true;
                queue.emplace(nx, ny);
                int global_ny = start_row_ + ny;
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
  int local_count = static_cast<int>(local_components_.size());
  std::vector<int> component_counts(size_, 0);

  MPI_Gather(&local_count, 1, MPI_INT, component_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    int total_fragments = std::accumulate(component_counts.begin(), component_counts.end(), 0);
    all_fragments_.clear();
    all_fragments_.reserve(total_fragments);
    all_fragments_.insert(all_fragments_.end(), local_components_.begin(), local_components_.end());
    for (int src = 1; src < size_; ++src) {
      if (component_counts[src] > 0) {
        for (int i = 0; i < component_counts[src]; ++i) {
          int comp_size;
          MPI_Recv(&comp_size, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          std::vector<int> point_data(comp_size * 2);
          MPI_Recv(point_data.data(), comp_size * 2, MPI_INT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          Component comp;
          comp.pixels.resize(comp_size);
          for (int p = 0; p < comp_size; ++p) {
            comp.pixels[p] = {point_data[p * 2], point_data[p * 2 + 1]};
          }

          all_fragments_.push_back(comp);
        }
      }
    }
  } else {
    for (const auto &comp : local_components_) {
      int comp_size = static_cast<int>(comp.pixels.size());
      MPI_Send(&comp_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      std::vector<int> point_data;
      point_data.reserve(comp_size * 2);
      for (const auto &point : comp.pixels) {
        point_data.push_back(point.first);
        point_data.push_back(point.second);
      }

      MPI_Send(point_data.data(), comp_size * 2, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }
  }
}

int64_t VidermanAConvexHullMPI::PointToHash(const Point &p) const {
  return (static_cast<int64_t>(p.first) << 32) | static_cast<uint32_t>(p.second);
}

int VidermanAConvexHullMPI::FindRoot(std::vector<int> &parent, int x) {
  while (parent[x] != x) {
    parent[x] = parent[parent[x]];
    x = parent[x];
  }
  return x;
}

void VidermanAConvexHullMPI::UnionSets(std::vector<int> &parent, std::vector<int> &rank, int x, int y) {
  int rootX = FindRoot(parent, x);
  int rootY = FindRoot(parent, y);

  if (rootX != rootY) {
    if (rank[rootX] < rank[rootY]) {
      parent[rootX] = rootY;
    } else if (rank[rootX] > rank[rootY]) {
      parent[rootY] = rootX;
    } else {
      parent[rootY] = rootX;
      rank[rootX]++;
    }
  }
}

void VidermanAConvexHullMPI::MergeFragmentsOnRank0() {
  if (all_fragments_.empty()) {
    return;
  }

  int n = static_cast<int>(all_fragments_.size());
  std::vector<int> parent(n);
  std::vector<int> rank(n, 0);
  std::iota(parent.begin(), parent.end(), 0);
  std::unordered_map<int64_t, int> point_to_component;

  for (int comp_idx = 0; comp_idx < n; ++comp_idx) {
    for (const auto &point : all_fragments_[comp_idx].pixels) {
      point_to_component[PointToHash(point)] = comp_idx;
    }
  }
  for (int comp_idx = 0; comp_idx < n; ++comp_idx) {
    for (const auto &point : all_fragments_[comp_idx].pixels) {
      for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
          if (dx == 0 && dy == 0) {
            continue;
          }

          Point neighbor = {point.first + dx, point.second + dy};
          auto it = point_to_component.find(PointToHash(neighbor));

          if (it != point_to_component.end()) {
            int neighbor_comp_idx = it->second;
            if (comp_idx != neighbor_comp_idx) {
              UnionSets(parent, rank, comp_idx, neighbor_comp_idx);
            }
          }
        }
      }
    }
  }
  std::unordered_map<int, std::vector<Point>> merged_points;
  for (int i = 0; i < n; ++i) {
    int root = FindRoot(parent, i);
    merged_points[root].insert(merged_points[root].end(), all_fragments_[i].pixels.begin(),
                               all_fragments_[i].pixels.end());
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
  std::sort(points.begin(), points.end());
  auto last = std::unique(points.begin(), points.end());
  points.erase(last, points.end());
}

long long VidermanAConvexHullMPI::CrossProduct(const Point &O, const Point &A, const Point &B) const {
  return static_cast<long long>(A.first - O.first) * static_cast<long long>(B.second - O.second) -
         static_cast<long long>(A.second - O.second) * static_cast<long long>(B.first - O.first);
}

std::vector<Point> VidermanAConvexHullMPI::GrahamScan(const std::vector<Point> &points) {
  if (points.size() <= 3) {
    return points;
  }
  std::vector<Point> pts = points;
  size_t pivot_idx = 0;
  for (size_t i = 1; i < pts.size(); ++i) {
    if (pts[i].second < pts[pivot_idx].second ||
        (pts[i].second == pts[pivot_idx].second && pts[i].first < pts[pivot_idx].first)) {
      pivot_idx = i;
    }
  }
  std::swap(pts[0], pts[pivot_idx]);
  Point pivot = pts[0];
  std::sort(pts.begin() + 1, pts.end(), [&pivot, this](const Point &a, const Point &b) {
    long long cross = CrossProduct(pivot, a, b);
    if (cross != 0) {
      return cross > 0;
    }
    long long dist_a = static_cast<long long>(a.first - pivot.first) * (a.first - pivot.first) +
                       static_cast<long long>(a.second - pivot.second) * (a.second - pivot.second);
    long long dist_b = static_cast<long long>(b.first - pivot.first) * (b.first - pivot.first) +
                       static_cast<long long>(b.second - pivot.second) * (b.second - pivot.second);
    return dist_a < dist_b;
  });
  std::vector<Point> unique_pts;
  unique_pts.push_back(pts[0]);

  for (size_t i = 1; i < pts.size(); ++i) {
    while (i < pts.size() - 1 && CrossProduct(pivot, pts[i], pts[i + 1]) == 0) {
      ++i;
    }
    unique_pts.push_back(pts[i]);
  }

  if (unique_pts.size() <= 3) {
    return unique_pts;
  }
  std::vector<Point> hull;
  hull.reserve(unique_pts.size());

  hull.push_back(unique_pts[0]);
  hull.push_back(unique_pts[1]);
  hull.push_back(unique_pts[2]);

  for (size_t i = 3; i < unique_pts.size(); ++i) {
    while (hull.size() >= 2 && CrossProduct(hull[hull.size() - 2], hull.back(), unique_pts[i]) <= 0) {
      hull.pop_back();
    }
    hull.push_back(unique_pts[i]);
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
  for (int comp_idx = 0; comp_idx < total_components; ++comp_idx) {
    Component comp;

    if (rank_ == 0) {
      comp = output[comp_idx];
    }
    int num_points = static_cast<int>(comp.pixels.size());
    MPI_Bcast(&num_points, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank_ != 0) {
      comp.pixels.resize(num_points);
    }
    if (num_points > 0) {
      std::vector<int> points_data(num_points * 2);

      if (rank_ == 0) {
        for (int p = 0; p < num_points; ++p) {
          points_data[p * 2] = comp.pixels[p].first;
          points_data[p * 2 + 1] = comp.pixels[p].second;
        }
      }

      MPI_Bcast(points_data.data(), num_points * 2, MPI_INT, 0, MPI_COMM_WORLD);

      if (rank_ != 0) {
        for (int p = 0; p < num_points; ++p) {
          comp.pixels[p] = {points_data[p * 2], points_data[p * 2 + 1]};
        }
      }
    }
    int hull_size = static_cast<int>(comp.hull.size());
    MPI_Bcast(&hull_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (hull_size > 0) {
      std::vector<int> hull_data(hull_size * 2);

      if (rank_ == 0) {
        for (int h = 0; h < hull_size; ++h) {
          hull_data[h * 2] = comp.hull[h].first;
          hull_data[h * 2 + 1] = comp.hull[h].second;
        }
      }

      MPI_Bcast(hull_data.data(), hull_size * 2, MPI_INT, 0, MPI_COMM_WORLD);

      if (rank_ != 0) {
        comp.hull.resize(hull_size);
        for (int h = 0; h < hull_size; ++h) {
          comp.hull[h] = {hull_data[h * 2], hull_data[h * 2 + 1]};
        }
      }
    } else if (rank_ != 0) {
      comp.hull = comp.pixels;
    }

    if (rank_ != 0) {
      if (comp_idx == 0) {
        output.clear();
        output.reserve(total_components);
      }
      output.push_back(comp);
    }
  }
}

}  // namespace viderman_a_convex_hull
