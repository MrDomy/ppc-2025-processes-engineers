#pragma once

#include <cstdint>
#include <vector>

#include "task/include/task.hpp"
#include "viderman_a_convex_hull/common/include/common.hpp"

namespace viderman_a_convex_hull {

class VidermanAConvexHullSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VidermanAConvexHullSEQ(const InType &in);

  [[nodiscard]] static int64_t CrossProduct(const Point &o, const Point &a, const Point &b);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static std::vector<Component> FindConnectedComponents(const ImageData &image);
  static std::vector<Point> BuildConvexHull(const std::vector<Point> &points);

  static void RemoveDuplicatePoints(std::vector<Point> &points);
};

}  // namespace viderman_a_convex_hull
