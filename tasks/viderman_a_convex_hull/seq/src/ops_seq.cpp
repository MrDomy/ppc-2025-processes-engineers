#include "viderman_a_convex_hull/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "viderman_a_convex_hull/common/include/common.hpp"

namespace viderman_a_convex_hull {

namespace {

void ProcessPixelCell(const std::vector<uint8_t> &pixels, std::vector<bool> &visited, int col_idx, int row_idx,
                      int width, int height, Component &comp) {
  const size_t idx = (static_cast<size_t>(row_idx) * static_cast<size_t>(width)) + static_cast<size_t>(col_idx);

  if (pixels[idx] != 255 || visited[idx]) {
    return;
  }

  const std::vector<Point> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
  std::queue<Point> queue;
  queue.emplace(col_idx, row_idx);
  visited[idx] = true;
  comp.pixels.emplace_back(col_idx, row_idx);

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
          comp.pixels.emplace_back(nx, ny);
        }
      }
    }
  }
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

std::vector<Point> RemoveCollinearPoints(const std::vector<Point> &pts, const Point &pivot) {
  std::vector<Point> unique_pts;
  unique_pts.push_back(pts[0]);

  for (size_t idx = 1; idx < pts.size(); ++idx) {
    while (idx < pts.size() - 1 && VidermanAConvexHullSEQ::CrossProduct(pivot, pts[idx], pts[idx + 1]) == 0) {
      ++idx;
    }
    unique_pts.push_back(pts[idx]);
  }

  return unique_pts;
}

}  // namespace

VidermanAConvexHullSEQ::VidermanAConvexHullSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool VidermanAConvexHullSEQ::ValidationImpl() {
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

bool VidermanAConvexHullSEQ::PreProcessingImpl() {
  return true;
}

int64_t VidermanAConvexHullSEQ::CrossProduct(const Point &o, const Point &a, const Point &b) {
  return (static_cast<int64_t>(a.first - o.first) * static_cast<int64_t>(b.second - o.second)) -
         (static_cast<int64_t>(a.second - o.second) * static_cast<int64_t>(b.first - o.first));
}

void VidermanAConvexHullSEQ::RemoveDuplicatePoints(std::vector<Point> &points) {
  if (points.empty()) {
    return;
  }

  std::ranges::sort(points);
  const auto last = std::ranges::unique(points).begin();
  points.erase(last, points.end());
}

std::vector<Point> VidermanAConvexHullSEQ::BuildConvexHull(const std::vector<Point> &points) {
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

std::vector<Component> VidermanAConvexHullSEQ::FindConnectedComponents(const ImageData &image) {
  const int width = image.width;
  const int height = image.height;
  const auto &pixels = image.pixels;

  std::vector<bool> visited(static_cast<size_t>(width) * static_cast<size_t>(height), false);
  std::vector<Component> components;

  for (int row_idx = 0; row_idx < height; ++row_idx) {
    for (int col_idx = 0; col_idx < width; ++col_idx) {
      Component comp;
      ProcessPixelCell(pixels, visited, col_idx, row_idx, width, height, comp);
      if (!comp.pixels.empty()) {
        components.push_back(std::move(comp));
      }
    }
  }

  return components;
}

bool VidermanAConvexHullSEQ::RunImpl() {
  const auto &input_image = GetInput();

  std::vector<Component> components = FindConnectedComponents(input_image);

  for (auto &component : components) {
    if (component.pixels.size() >= 3) {
      RemoveDuplicatePoints(component.pixels);
      component.hull = BuildConvexHull(component.pixels);
    } else {
      component.hull = component.pixels;
    }
  }

  GetOutput() = std::move(components);
  return true;
}

bool VidermanAConvexHullSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace viderman_a_convex_hull
