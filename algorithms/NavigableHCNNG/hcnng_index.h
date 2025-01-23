// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <math.h>

#include <algorithm>
#include <queue>
#include <random>
#include <set>
#include <utility>

#include "../utils/graph.h"
#include "clusterEdge.h"
#include "hcnng_utils.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"

namespace parlayANN {

template <typename Point, typename PointRange, typename indexType>
struct hcnng_index {
  using distanceType = typename Point::distanceType;
  using edge = std::pair<indexType, indexType>;
  using pid = std::pair<indexType, distanceType>;
  using GraphI = Graph<indexType>;
  using PR = PointRange;

  using utils = cluster_utils<Point, PointRange, indexType>;

  hcnng_index() {}

  static std::pair<size_t, size_t> greedyShape(
      indexType p, indexType r, PR &Points, GraphI &G,
      parlay::sequence<size_t> &active_indices) {
    auto curr = r;
    std::vector<indexType> path;
    std::set<indexType> visited;
    bool backtracked = false;
    indexType first_backtrack = utils::kNullId;
    indexType last_node_on_path = utils::kNullId;
    path.push_back(r);
    // greedy search with backtracking
    while (curr != p) {
      if (path.back() != curr) path.push_back(curr);
      visited.insert(curr);
      auto min = utils::kNullDist;
      indexType next = utils::kNullId;
      for (size_t i = 0; i < G[curr].size(); i++) {
        auto nbh = G[curr][i];
        if (std::find(active_indices.begin(), active_indices.end(), nbh) !=
                active_indices.end() &&
            visited.find(nbh) == visited.end()) {
          auto dist = Points[p].distance(Points[nbh]);
          if (dist < min) {
            min = dist;
            next = nbh;
          }
        }
      }
      if (next == utils::kNullId) {
        if (path.size() == 0) {
          std::cout << "Error: no path found" << std::endl;
          break;
        }
        path.pop_back();
        curr = path.back();
        if (!backtracked) {
          first_backtrack = curr;
          backtracked = true;
          break;
        }
      } else {
        curr = next;
      }
    }
    // last_node_on_path is the node incident to p on the path
    // last_node_on_path = path[path.size() - 2];

    // if back-tracking happened, then add edge from the backtrack node to p and
    // remove the edge from the last node on the path to p
    if (backtracked) {
      G[first_backtrack].append_neighbor(p);
      return std::make_pair(first_backtrack, p);
      // for (size_t i = 0; i < G[last_node_on_path].size(); i++) {
      //   if (G[last_node_on_path][i] == p) {
      //     G[last_node_on_path][i] =
      //     G[last_node_on_path][G[last_node_on_path].size() - 1];
      //     G[last_node_on_path].update_neighbors(parlay::make_slice(G[last_node_on_path],
      //     G[last_node_on_path].size() - 1)); break;
      //   }
      // }
    }
    return std::make_pair(utils::kNullId, utils::kNullId);
  }

  void build_index(GraphI &G, PR &Points, long cluster_rounds,
                   long cluster_size, long MSTDeg, bool multi_pivot, bool prune, bool prune_all, double alpha,
                   bool mst_k, long prune_degree, bool vamana_long_range, double top_level_pct, long top_level_leaders) {
    cluster<Point, PointRange, indexType> C;
    C.START_POINTS.push_back(0);
    C.MSTDeg = MSTDeg;
    C.MULTI_PIVOT = multi_pivot;
    C.ALPHA = alpha;
    C.MAX_CLUSTER_SIZE=cluster_size;
    C.MAX_MERGED_CLUSTER_SIZE=cluster_size;
		C.TOP_LEVEL_NUM_LEADERS = top_level_leaders;
    std::cout << "Set MSTDeg to: " << MSTDeg << " MultiPivot to: " << multi_pivot << std::endl;

    if (mst_k) {
      std::cout << "Using MSTk" << std::endl;
      C.LEAF_ALG = "MSTk";
    } else {
      std::cout << "Using VamanaLeaf" << std::endl;
      C.LEAF_ALG = "VamanaLeaf";
    }
    C.multiple_clustertrees(G, Points, cluster_size, cluster_rounds);

    if (vamana_long_range) {
      std::cout << "Total start points = " << C.START_POINTS.size() << std::endl;
      std::cout << "Adding long range edges using Vamana" << std::endl;
      BuildParams BP;
      BP.R = 40;
      BP.L = 200;
      BP.alpha = alpha;
      BP.num_passes = 3;
      BP.single_batch = 0;

      if (top_level_pct > 0) {
        std::mt19937 prng(0);
        auto ids = parlay::tabulate(Points.size(), [&](uint32_t i) { return i; });
        std::vector<uint32_t> extra_pts(G.size() * top_level_pct);
        std::sample(ids.begin(), ids.end(), extra_pts.begin(), extra_pts.size(), prng);
        std::cout << "Sampled an extra: " << extra_pts.size() << " start points." << std::endl;
        for (auto top : extra_pts)  {
          C.START_POINTS.push_back(top);
        }
      }

      // Remove duplicates from start points (points could be added as
      // start points multiple times in different replicas).
      C.START_POINTS = parlay::remove_duplicates(C.START_POINTS);
      auto edges = run_vamana_on_indices(C.START_POINTS, Points, BP);
      utils::process_edges(G, std::move(edges));
      utils::remove_all_duplicates(G);
    }

    C.START_POINTS.clear();
    if (prune) {
      parlay::parallel_for(0, G.size(), [&](size_t i) {
        if (prune_all || G[i].size() > prune_degree) {
          utils::robustPrune(i, Points, G, alpha, prune_degree);
        }
      });
    }
  }
};

}  // namespace parlayANN
