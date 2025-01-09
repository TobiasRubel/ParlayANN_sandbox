#include <cstdint>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cassert>

#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include "utils/types.h"
#include "utils/euclidian_point.h"
#include "utils/point_range.h"
#include "utils/graph.h"
#include "vamana/neighbors.h"

using index_type = uint32_t;
using value_type = float;
using point_type = parlayANN::Euclidian_Point<value_type>;
using point_range_type = parlayANN::PointRange<point_type>;

// Variation of beam search to allow custom functions called on vertices/edges
// PointFunc is called on vertices when they are visited (ie. expanded)
// EdgeFunc is called on vertices when they are added to the frontier (ie. seen)
template<typename indexType, typename Point, typename PointRange,
         typename QPoint, typename QPointRange, class GT,
         typename PointFunc, typename EdgeFunc>
std::pair<std::pair<parlay::sequence<std::pair<indexType, typename Point::distanceType>>,
                    parlay::sequence<std::pair<indexType, typename Point::distanceType>>>,
          size_t>
custom_beam_search(const GT &G,
                     const Point p,  const PointRange &Points,
                     const QPoint qp, const QPointRange &Q_Points,
                     const parlay::sequence<indexType> starting_points,
                     const parlayANN::QueryParams &QP,
                     bool use_filtering = false
                     ) {
  using dtype = typename Point::distanceType;
  using id_dist = std::pair<indexType, dtype>;
  int beamSize = QP.beamSize;

  if (starting_points.size() == 0) {
    std::cout << "beam search expects at least one start point" << std::endl;
    abort();
  }

  // compare two (node_id,distance) pairs, first by distance and then id if
  // equal
  using distanceType = typename Point::distanceType;
  auto less = [&](id_dist a, id_dist b) {
    return a.second < b.second || (a.second == b.second && a.first < b.first);
  };

  // used as a hash filter (can give false negative -- i.e. can say
  // not in table when it is)
  int bits = std::max<int>(10, std::ceil(std::log2(beamSize * beamSize)) - 2);
  std::vector<indexType> hash_filter(1 << bits, -1);
  auto has_been_seen = [&](indexType a) -> bool {
    int loc = parlay::hash64_2(a) & ((1 << bits) - 1);
    if (hash_filter[loc] == a) return true;
    hash_filter[loc] = a;
    return false;
  };

  // Frontier maintains the closest points found so far and its size
  // is always at most beamSize.  Each entry is a (id,distance) pair.
  // Initialized with starting points and kept sorted by distance.
  std::vector<id_dist> frontier;
  frontier.reserve(beamSize);
  for (auto q : starting_points) {
    frontier.push_back(id_dist(q, Points[q].distance(p)));
    has_been_seen(q);
    //PointFunc(q);
  }
  std::sort(frontier.begin(), frontier.end(), less);

  // The subset of the frontier that has not been visited
  // Use the first of these to pick next vertex to visit.
  std::vector<id_dist> unvisited_frontier(beamSize);
  for (int i=0; i < frontier.size(); i++)
    unvisited_frontier[i] = frontier[i];

  // maintains sorted set of visited vertices (id-distance pairs)
  std::vector<id_dist> visited;
  visited.reserve(2 * beamSize);

  // counters
  size_t dist_cmps = starting_points.size();
  size_t full_dist_cmps = starting_points.size();
  int remain = frontier.size();
  int num_visited = 0;

  // used as temporaries in the loop
  std::vector<id_dist> new_frontier(2 * std::max<size_t>(beamSize,starting_points.size()) +
                                    G.max_degree());
  std::vector<id_dist> candidates;
  candidates.reserve(G.max_degree() + beamSize);
  std::vector<indexType> filtered;
  filtered.reserve(G.max_degree());
  std::vector<indexType> pruned;
  pruned.reserve(G.max_degree());

  dtype filter_threshold_sum = 0.0;
  int filter_threshold_count = 0;
  dtype filter_threshold;

  // offset into the unvisited_frontier vector (unvisited_frontier[offset] is the next to visit)
  int offset = 0;

  // The main loop.  Terminate beam search when the entire frontier
  // has been visited or have reached max_visit.
  while (remain > offset && num_visited < QP.limit) {
    // the next node to visit is the unvisited frontier node that is closest to p
    id_dist current = unvisited_frontier[offset];
    G[current.first].prefetch();
    // add to visited set
    auto position = std::upper_bound(visited.begin(), visited.end(), current, less);
    visited.insert(position, current);
    num_visited++;
    bool frontier_full = frontier.size() == beamSize;
    PointFunc(current.first);

    // if using filtering based on lower quality distances measure, then maintain the average
    // of low quality distance to the last point in the frontier (if frontier is full)
    if (use_filtering && frontier_full) {
      filter_threshold_sum += Q_Points[frontier.back().first].distance(qp);
      filter_threshold_count++;
      filter_threshold = filter_threshold_sum / filter_threshold_count;
    }

    // keep neighbors that have not been visited (using approximate
    // hash). Note that if a visited node is accidentally kept due to
    // approximate hash it will be removed below by the union.
    pruned.clear();
    filtered.clear();
    long num_elts = std::min<long>(G[current.first].size(), QP.degree_limit);
    for (indexType i=0; i<num_elts; i++) {
      auto a = G[current.first][i];
      if (has_been_seen(a) || Points[a].same_as(p)) continue;  // skip if already seen
      Q_Points[a].prefetch();
      pruned.push_back(a);
    }
    dist_cmps += pruned.size();

    // filter using low-quality distance
    if (use_filtering && frontier_full) {
      for (auto a : pruned) {
        if (frontier_full && Q_Points[a].distance(qp) >= filter_threshold) continue;
        filtered.push_back(a);
        Points[a].prefetch();
      }
    } else std::swap(filtered, pruned);

    // Further remove if distance is greater than current
    // furthest distance in current frontier (if full).
    distanceType cutoff = (frontier_full
                           ? frontier[frontier.size() - 1].second
                           : (distanceType)std::numeric_limits<int>::max());
    for (auto a : filtered) {
      distanceType dist = Points[a].distance(p);
      full_dist_cmps++;
      // skip if frontier not full and distance too large
      if (dist >= cutoff) continue;
      candidates.push_back(std::pair{a, dist});
      EdgeFunc(current.first, a);
    }
    // If candidates insufficently full then skip rest of step until sufficiently full.
    // This improves performance for higher accuracies (e.g. beam sizes of 100+)
    if (candidates.size() == 0 || 
        (QP.limit >= 2 * beamSize &&
         candidates.size() < beamSize/8 &&
         offset + 1 < remain)) {
      offset++;
      continue;
    }
    offset = 0;

    // sort the candidates by distance from p,
    // and remove any duplicates (to be robust for neighbor lists with duplicates)
    std::sort(candidates.begin(), candidates.end(), less);
    auto candidates_end = std::unique(candidates.begin(), candidates.end(),
                                      [] (auto a, auto b) {return a.first == b.first;});

    // union the frontier and candidates into new_frontier, both are sorted
    auto new_frontier_size =
      std::set_union(frontier.begin(), frontier.end(), candidates.begin(),
                     candidates_end, new_frontier.begin(), less) -
      new_frontier.begin();
    candidates.clear();
    
    // trim to at most beam size
    new_frontier_size = std::min<size_t>(beamSize, new_frontier_size);

    // if a k is given (i.e. k != 0) then trim off entries that have a
    // distance greater than cut * current-kth-smallest-distance.
    // Only used during query and not during build.
    if (QP.k > 0 && new_frontier_size > QP.k && Points[0].is_metric())
      new_frontier_size = std::max<indexType>(
        (std::upper_bound(new_frontier.begin(),
                          new_frontier.begin() + new_frontier_size,
                          std::pair{0, QP.cut * new_frontier[QP.k].second}, less) -
         new_frontier.begin()), frontier.size());

    // copy new_frontier back to the frontier
    frontier.clear();
    for (indexType i = 0; i < new_frontier_size; i++)
      frontier.push_back(new_frontier[i]);

    // get the unvisited frontier
    remain = (std::set_difference(frontier.begin(),
                                  frontier.begin() + std::min<long>(frontier.size(), QP.beamSize),
                                  visited.begin(),
                                  visited.end(),
                                  unvisited_frontier.begin(), less) -
              unvisited_frontier.begin());
  }

  return std::make_pair(std::make_pair(parlay::to_sequence(frontier),
                                       parlay::to_sequence(visited)),
                        full_dist_cmps);
}