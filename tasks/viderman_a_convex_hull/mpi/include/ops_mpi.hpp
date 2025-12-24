#pragma once

#include <cstdint>
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

  [[nodiscard]] static int64_t CrossProduct(const Point &o, const Point &a, const Point &b);
  static void UnionSets(std::vector<int> &parent, std::vector<int> &rank, int x, int y);

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

  static std::vector<Point> GrahamScan(const std::vector<Point> &points);
  static int FindRoot(std::vector<int> &parent, int x);

  static void RemoveDuplicatePoints(std::vector<Point> &points);
  [[nodiscard]] static int64_t PointToHash(const Point &p);

  void GatherComponentCounts(std::vector<int> &comp_counts, std::vector<int> &displacements, int &total_comps);
  void GatherComponentSizes(const std::vector<int> &comp_counts, const std::vector<int> &displacements,
                            std::vector<int> &all_comp_sizes);
  std::vector<int> PackLocalPoints();
  void ComputeDisplacementsAndCounts(const std::vector<int> &comp_counts, const std::vector<int> &displacements,
                                     const std::vector<int> &all_comp_sizes, std::vector<int> &point_displacements,
                                     std::vector<int> &point_recvcounts) const;
  void UnpackAllPoints(const std::vector<int> &all_points_data, const std::vector<int> &comp_counts,
                       const std::vector<int> &displacements, const std::vector<int> &all_comp_sizes);
  void GatherComponentPoints(const std::vector<int> &comp_counts, const std::vector<int> &displacements,
                             const std::vector<int> &all_comp_sizes, int total_points);

  int rank_{0};
  int size_{1};

  int width_{0};
  int height_{0};
  int start_row_{0};
  int end_row_{0};
  int local_rows_{0};

  ImageData local_image_;
  std::vector<Component> local_components_;

  std::vector<Component> all_fragments_;
  std::vector<Component> merged_components_;
};

}  // namespace viderman_a_convex_hull
