#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "task/include/task.hpp"
#include "viderman_a_convex_hull/common/include/common.hpp"

namespace viderman_a_convex_hull {

class VidermanAConvexHullMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }

  explicit VidermanAConvexHullMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void DistributeImageData();
  void FindLocalComponents();
  void GatherAllFragments();
  void MergeFragmentsOnRank0();
  void BuildAllConvexHullsOnRank0();
  void BroadcastFinalResult();

  std::vector<Point> GrahamScan(const std::vector<Point> &points);
  [[nodiscard]] int64_t CrossProduct(const Point &o, const Point &a, const Point &b) const;

  int FindRoot(std::vector<int> &parent, int x);
  void UnionSets(std::vector<int> &parent, std::vector<int> &rank, int x, int y);

  void RemoveDuplicatePoints(std::vector<Point> &points);
  [[nodiscard]] int64_t PointToHash(const Point &p) const;

  int rank_ = 0;
  int size_ = 1;

  int width_ = 0;
  int height_ = 0;
  int start_row_ = 0;
  int end_row_ = 0;
  int local_rows_ = 0;

  ImageData local_image_;
  std::vector<Component> local_components_;

  std::vector<Component> all_fragments_;
  std::vector<Component> merged_components_;
};

}  // namespace viderman_a_convex_hull
