#include "viderman_a_convex_hull/seq/include/ops_seq.hpp"

#include <algorithm>
#include <queue>
#include <stack>
#include <utility>
#include <vector>

#include "util/include/util.hpp"
#include "viderman_a_convex_hull/common/include/common.hpp"

namespace viderman_a_convex_hull {

VidermanAConvexHullSEQ::VidermanAConvexHullSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool VidermanAConvexHullSEQ::ValidationImpl() {
  const auto &input = GetInput();

  if (input.width <= 0 || input.height <= 0) {
    return false;
  }

  if (static_cast<size_t>(input.width * input.height) != input.pixels.size()) {
    return false;
  }

  return true;
}

bool VidermanAConvexHullSEQ::PreProcessingImpl() {
  return true;
}

long long VidermanAConvexHullSEQ::CrossProduct(const Point &O, const Point &A, const Point &B) const {
  return static_cast<long long>(A.first - O.first) * static_cast<long long>(B.second - O.second) -
         static_cast<long long>(A.second - O.second) * static_cast<long long>(B.first - O.first);
}

void VidermanAConvexHullSEQ::RemoveDuplicatePoints(std::vector<Point> &points) {
  if (points.empty()) {
    return;
  }

  std::sort(points.begin(), points.end());
  auto last = std::unique(points.begin(), points.end());
  points.erase(last, points.end());
}

std::vector<Point> VidermanAConvexHullSEQ::BuildConvexHull(const std::vector<Point> &points) {
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

std::vector<Component> VidermanAConvexHullSEQ::FindConnectedComponents(const ImageData &image) {
  int width = image.width;
  int height = image.height;
  const auto &pixels = image.pixels;

  std::vector<bool> visited(static_cast<size_t>(width) * static_cast<size_t>(height), false);
  std::vector<Component> components;

  const std::vector<std::pair<int, int>> directions = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      size_t idx = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);

      if (pixels[idx] == 255 && !visited[idx]) {
        Component comp;
        std::queue<Point> q;

        q.emplace(x, y);
        visited[idx] = true;
        comp.pixels.emplace_back(x, y);

        while (!q.empty()) {
          Point current = q.front();
          q.pop();

          for (const auto &dir : directions) {
            int nx = current.first + dir.first;
            int ny = current.second + dir.second;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
              size_t nidx = static_cast<size_t>(ny) * static_cast<size_t>(width) + static_cast<size_t>(nx);

              if (pixels[nidx] == 255 && !visited[nidx]) {
                visited[nidx] = true;
                q.emplace(nx, ny);
                comp.pixels.emplace_back(nx, ny);
              }
            }
          }
        }

        if (!comp.pixels.empty()) {
          components.push_back(std::move(comp));
        }
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
