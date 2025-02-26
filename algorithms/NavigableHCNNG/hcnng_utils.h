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

#pragma once

#include <math.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <queue>
#include <random>
#include <set>

#include "../utils/graph.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"

#include "../vamana/index.h"

namespace parlayANN {

template <typename Point, typename PointRange, typename indexType>
struct cluster_utils {
  using distanceType = typename Point::distanceType;
  using edge = std::pair<indexType, indexType>;
  using pid = std::pair<indexType, distanceType>;
  using GraphI = Graph<indexType>;
  using PR = PointRange;

  static constexpr indexType kNullId = std::numeric_limits<indexType>::max();
  static constexpr distanceType kNullDist =
      std::numeric_limits<distanceType>::max();
  using labelled_edge = std::pair<edge, distanceType>;
  static constexpr labelled_edge kNullEdge = {{kNullId, kNullId}, kNullDist};

  // robustPrune routine as found in DiskANN paper, with the exception that the
  // new candidate set is added to the field new_nbhs instead of directly
  // replacing the out_nbh of p
  static void robustPrune(indexType p, PR &Points, GraphI &G, double alpha,
                          size_t degree) {
    // add out neighbors of p to the candidate set.
    std::vector<pid> candidates;
    for (size_t i = 0; i < G[p].size(); i++) {
      candidates.push_back(std::make_pair(G[p][i], Points[G[p][i]].distance(Points[p])));
    }

    // Sort the candidate set in reverse order according to distance from p.
    auto less = [&](pid a, pid b) { 
      return a.second < b.second || (a.second == b.second && a.first < b.first);
    };
    std::sort(candidates.begin(), candidates.end(), less);

    // remove any duplicates
    auto new_end =std::unique(candidates.begin(), candidates.end(),
			      [&] (auto x, auto y) {return x.first == y.first;});
    candidates = std::vector(candidates.begin(), new_end);

    std::vector<int> new_nbhs;
    new_nbhs.reserve(degree);

    size_t candidate_idx = 0;
    while (new_nbhs.size() < degree && candidate_idx < candidates.size()) {
      // Don't need to do modifications.
      indexType p_star = candidates[candidate_idx].first;
      candidate_idx++;
      if (p_star == p || p_star == kNullId) continue;

      new_nbhs.push_back(p_star);

      for (size_t i = candidate_idx; i < candidates.size(); i++) {
        indexType p_prime = candidates[i].first;
        if (p_prime != kNullId) {
          distanceType dist_starprime =
              Points[p_star].distance(Points[p_prime]);
          distanceType dist_pprime = candidates[i].second;
          if (alpha * dist_starprime <= dist_pprime)
            candidates[i].first = kNullId;
        }
      }
    }
    G[p].update_neighbors(new_nbhs);
  }

  // inserts each edge after checking for duplicates
  static void process_edges(GraphI &G, parlay::sequence<edge> edges) {
    long maxDeg = G.max_degree();
    auto grouped = parlay::group_by_key(edges);
    parlay::parallel_for(0, grouped.size(), [&](size_t i) {
      int32_t index = grouped[i].first;
      for (auto c : grouped[i].second) {
        if (G[index].size() < maxDeg) {
          G[index].append_neighbor(c);
        } else {
          remove_edge_duplicates(index, G);
          G[index].append_neighbor(c);
        }
      }
    });
  }

  static void remove_edge_duplicates(indexType p, GraphI &G) {
    parlay::sequence<indexType> points;
    for (indexType i = 0; i < G[p].size(); i++) {
      points.push_back(G[p][i]);
    }
    auto np = parlay::remove_duplicates(points);
    G[p].update_neighbors(np);
  }

  static void remove_all_duplicates(GraphI &G) {
    parlay::parallel_for(0, G.size(),
                         [&](size_t i) { remove_edge_duplicates(i, G); });
  }

};

template <typename PR, typename Seq>
auto run_vamana_on_indices(Seq &seq, PR &all_points, BuildParams &BP, bool parallel=true) {
  using indexType = uint32_t;
  Graph<indexType> G(BP.R, seq.size());

  using edge = std::pair<uint32_t, uint32_t>;
  parlay::sequence<edge> edges;

  for (size_t pass=0; pass<1; ++pass) {
    //std::sort(seq.begin(), seq.end());

    std::mt19937 prng(pass);
    std::shuffle(seq.begin(), seq.end(), prng);
  
    PR points = PR(all_points, seq);

    using findex = knn_index<PR, PR, indexType>;
    findex I(BP);
    stats<unsigned int> BuildStats(G.size());
    if (parallel) {
      I.build_index(G, points, points, BuildStats, /*sort_neighbors=*/true, /*print=*/false);
    } else {
      I.build_index_seq(G, points, points, BuildStats, /*sort_neighbors=*/true, /*print=*/false);
    }
    for (size_t i=0; i < G.size(); ++i) {
      size_t our_ind = seq[i];
      if (our_ind == 0) {
        std::cout << "Got point 0" << std::endl;
      }
      for (size_t j=0; j < G[i].size(); ++j) {
        auto neighbor_ind = seq[G[i][j]];
        edges.push_back(std::make_pair(our_ind, neighbor_ind));
      }
      if (G[i].size() > BP.R) {
        std::cout << "point: " << i << " exceeded max degree... " << std::endl;
        exit(0);
      }
    }
  }

  return edges;
}

template <typename PR, typename Seq>
auto run_quadprune_on_indices(Seq &seq, PR &all_points, BuildParams &BP, bool parallel=true) {
  using indexType = uint32_t;
  Graph<indexType> G(BP.R, seq.size());

  using edge = std::pair<uint32_t, uint32_t>;
  parlay::sequence<edge> edges;

  PR points = PR(all_points, seq);
  using findex = knn_index<PR, PR, indexType>;
  findex I(BP);
  stats<unsigned int> BuildStats(G.size());
  I.robust_prune_index(G, points, points, BuildStats, true, false);
  for (size_t i=0; i < G.size(); ++i) {
    size_t our_ind = seq[i];
    for (size_t j=0; j < G[i].size(); ++j) {
      auto neighbor_ind = seq[G[i][j]];
      edges.push_back(std::make_pair(our_ind, neighbor_ind));
    }
  }
  return edges;
}

template <typename PR, typename Seq>
auto distmat_quadprune(Seq &seq, PR &all_points, BuildParams &BP, bool parallel=true) {
  // parlay::internal::timer t;
  // std::cout << "Generating distance matrix" << std::endl;
  // t.start();
  using indexType = uint32_t;
  Graph<indexType> G(BP.R, seq.size());

  using edge = std::pair<uint32_t, uint32_t>;
  parlay::sequence<edge> edges;

  PR points = PR(all_points, seq);
  using findex = knn_index<PR, PR, indexType>;
  findex I(BP);
  stats<unsigned int> BuildStats(G.size());

  using distanceType = typename PR::Point::distanceType;
  auto dist_mat = new distanceType[seq.size() * seq.size()];
  // for (size_t i = 0; i < seq.size(); ++i) {
  //   for (size_t j = i + 1; j < seq.size(); ++j) {
  //     dist_mat[i * seq.size() + j] = dist_mat[j * seq.size() + i] = points[i].distance(points[j]);
  //   }
  // }
  for (size_t a = 0; a < 2; a++) {
    size_t i_min = a * seq.size() / 2, i_max = (a + 1) * seq.size() / 2;
    for (size_t b = a; b < 2; b++) {
      size_t j_min = b * seq.size() / 2, j_max = (b + 1) * seq.size() / 2;
      for (size_t i = i_min; i < i_max; ++i) {
        for (size_t j = std::max(i + 1, j_min); j < j_max; ++j) {
          dist_mat[i * seq.size() + j] = dist_mat[j * seq.size() + i] = points[i].distance(points[j]);
        }
      }
    }
  }
  // std::cout << "Distance matrix generated: " << t.next_time() << std::endl;

  I.distmat_robust_prune(G, points, points, dist_mat, BuildStats, true, false);
  // std::cout << "Pruning done: " << t.next_time() << std::endl;
  for (size_t i=0; i < G.size(); ++i) {
    size_t our_ind = seq[i];
    for (size_t j=0; j < G[i].size(); ++j) {
      auto neighbor_ind = seq[G[i][j]];
      edges.push_back(std::make_pair(our_ind, neighbor_ind));
    }
  }
  delete[] dist_mat;
  return edges;
}

template <typename PR, typename Seq>
auto run_fastprune(Seq &seq, PR &all_points, BuildParams &BP, bool parallel=true) {
  using indexType = uint32_t;
  Graph<indexType> G(BP.R, seq.size());

  using edge = std::pair<uint32_t, uint32_t>;
  parlay::sequence<edge> edges;

  PR points = PR(all_points, seq);
  using findex = knn_index<PR, PR, indexType>;
  findex I(BP);
  stats<unsigned int> BuildStats(G.size());
  // parlay::internal::timer t;
  // t.start();
  using distanceType = typename PR::Point::distanceType;
  auto dist_mat = new distanceType[seq.size() * seq.size()];
  for (size_t a = 0; a < 2; a++) {
    size_t i_min = a * seq.size() / 2, i_max = (a + 1) * seq.size() / 2;
    for (size_t b = a; b < 2; b++) {
      size_t j_min = b * seq.size() / 2, j_max = (b + 1) * seq.size() / 2;
      for (size_t i = i_min; i < i_max; ++i) {
        for (size_t j = std::max(i + 1, j_min); j < j_max; ++j) {
          dist_mat[i * seq.size() + j] = dist_mat[j * seq.size() + i] = points[i].distance(points[j]);
        }
      }
    }
  }
  // std::cout << "Distance matrix generated: " << t.next_time() << std::endl;

  I.distmat_fastprune(G, points, points, dist_mat, BuildStats, true, false);
  // std::cout << "Pruning done: " << t.next_time() << std::endl;
  for (size_t i=0; i < G.size(); ++i) {
    size_t our_ind = seq[i];
    for (size_t j=0; j < G[i].size(); ++j) {
      auto neighbor_ind = seq[G[i][j]];
      edges.push_back(std::make_pair(our_ind, neighbor_ind));
    }
  }
  delete[] dist_mat;
  return edges;
}

template <typename PR, typename Seq>
auto run_nnprune(uint32_t source ,Seq &seq, PR &all_points, BuildParams &BP, bool parallel=true) {
  using indexType = uint32_t;
  Graph<indexType> G(BP.R, seq.size());

  using edge = std::pair<uint32_t, uint32_t>;
  parlay::sequence<edge> edges;

  PR points = PR(all_points, seq);
  using findex = knn_index<PR, PR, indexType>;
  findex I(BP);
  stats<unsigned int> BuildStats(G.size());

  size_t source_ind = 0;
  for (size_t i = 0; i < seq.size(); i++) {
    if ((size_t)seq[i] == source) {
      source_ind = i;
      break;
    }
  }
  I.nn_prune_index(G, source_ind, points, points, BuildStats, true, false);
  // std::cout << "Pruning done: " << t.next_time() << std::endl;
  for (size_t i=0; i < G.size(); ++i) {
    size_t our_ind = seq[i];
    for (size_t j=0; j < G[i].size(); ++j) {
      auto neighbor_ind = seq[G[i][j]];
      edges.push_back(std::make_pair(our_ind, neighbor_ind));
    }
  }
  return edges;
}

class SpinLock {
public:
    // boilerplate to make it 'copyable'. but we just clear the spinlock. there is never a use case to copy a locked spinlock
    SpinLock() { }
    SpinLock(const SpinLock&) { }
    SpinLock& operator=(const SpinLock&) { spinner.clear(std::memory_order_relaxed); return *this; }

    bool tryLock() {
        return !spinner.test_and_set(std::memory_order_acquire);
    }

    void lock() {
        while (spinner.test_and_set(std::memory_order_acquire)) {
            // spin
            // stack overflow says adding 'cpu_relax' instruction may improve performance
        }
    }

    void unlock() {
        spinner.clear(std::memory_order_release);
    }

private:
    std::atomic_flag spinner = ATOMIC_FLAG_INIT;
};


struct DisjointSet {
  parlay::sequence<int> parent;
  parlay::sequence<int> rank;
  size_t N;

  DisjointSet(size_t size) {
    N = size;
    parent = parlay::sequence<int>(N);
    rank = parlay::sequence<int>(N);
    parlay::parallel_for(0, N, [&](size_t i) {
      parent[i] = i;
      rank[i] = 0;
    });
  }

  void _union(int x, int y) {
    int xroot = parent[x];
    int yroot = parent[y];
    int xrank = rank[x];
    int yrank = rank[y];
    if (xroot == yroot)
      return;
    else if (xrank < yrank)
      parent[xroot] = yroot;
    else {
      parent[yroot] = xroot;
      if (xrank == yrank) rank[xroot] = rank[xroot] + 1;
    }
  }

  int find(int x) {
    if (parent[x] == x) return x;
    int c = x;
    while (parent[c] != c) {
      c = parent[c];
    }
    while (x != c) {
      int s = parent[x];
      parent[x] = c;
      x = s;
    }
    return c;
  }

  void flatten() {
    for (int i = 0; i < N; i++) find(i);
  }

  bool is_full() {
    flatten();
    parlay::sequence<bool> truthvals(N);
    parlay::parallel_for(
        0, N, [&](size_t i) { truthvals[i] = (parent[i] == parent[0]); });
    auto ff = [&](bool a) { return not a; };
    auto filtered = parlay::filter(truthvals, ff);
    if (filtered.size() == 0) return true;
    return false;
  }
};

} // end namespace
