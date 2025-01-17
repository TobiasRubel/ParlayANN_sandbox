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
#include <functional>
#include <queue>
#include <random>
#include <set>
#include "../utils/simhash.h"
#include "../utils/graph.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"

namespace parlayANN {

//using Floyd's algorithm
parlay::sequence<size_t> select_k_random(parlay::sequence<size_t> &active_indices,
                  parlay::random &rnd, size_t k) {
  parlay::sequence<size_t> selected_indices;
  auto n = active_indices.size();
  for (size_t i = n-k; i < n; i++) {
    size_t index = rnd.ith_rand(i) % i;
    //if index not in selected_indices push_back, else push_back i
    if (std::find(selected_indices.begin(), selected_indices.end(), index) == selected_indices.end())
      selected_indices.push_back(index);
    else 
      selected_indices.push_back(i);
    }
  return selected_indices;
}

template <typename Point, typename PointRange, typename indexType>
struct karycluster {
  using distanceType = typename Point::distanceType;
  using edge = std::pair<indexType, indexType>;
  using labelled_edge = std::pair<edge, distanceType>;
  using GraphI = Graph<indexType>;
  using PR = PointRange;

  karycluster() {}

  int generate_index(int N, int i) {
    return (N * (N - 1) - (N - i) * (N - i - 1)) / 2;
  }

  template <typename F>
  void randomkary_clustering(GraphI &G, PR &Points,
                         parlay::sequence<size_t>& active_indices, parlay::random &rnd,
                         size_t cluster_size, F g,
                         long MSTDeg, size_t num_pivots) {

    if (active_indices.size() <= cluster_size)
            g(G, Points, active_indices, MSTDeg);

    else {
      auto pivots = select_k_random(active_indices, rnd, num_pivots);
      //print pivots
      // std::cout << "selected pivots: ";;
      // for (size_t i = 0; i < pivots.size(); i++) {
      //  std::cout << pivots[i] << " ";
      // }
      // std::cout << std::endl;
      parlay::sequence<parlay::sequence<size_t>> buckets(num_pivots);
      // for (size_t i = 0; i < active_indices.size(); i++) {
      //   size_t min_dist = std::numeric_limits<size_t>::max();
      //   size_t min_pivot = 0;
      //   for (size_t j = 0; j < num_pivots; j++) {
      //     size_t dist = Points[active_indices[i]].distance(Points[pivots[j]]);
      //     if (dist < min_dist) {
      //       min_dist = dist;
      //       min_pivot = j;
      //     } else if (dist == min_dist) {
      //       if (rnd.ith_rand(i) % 2 == 0) {
      //         min_pivot = j;
      //       }
      //     }
      //   }
      //   buckets[min_pivot].push_back(active_indices[i]);
      // }
      //assign pivots for each point in parallel
      auto pivot_assignments = parlay::tabulate(active_indices.size(), [&](size_t i) {
        size_t min_dist = std::numeric_limits<size_t>::max();
        size_t min_pivot = 0;
        for (size_t j = 0; j < num_pivots; j++) {
          size_t dist = Points[active_indices[i]].distance(Points[pivots[j]]);
          if (dist < min_dist) {
            min_dist = dist;
            min_pivot = j;
          } else if (dist == min_dist) {
            if (rnd.ith_rand(i) % 2 == 0) {
              min_pivot = j;
            }
          }
        }
        return min_pivot;
      });
      //assign points to buckets
      for (size_t i = 0; i < active_indices.size(); i++) {
        buckets[pivot_assignments[i]].push_back(active_indices[i]);
      }

      //std::cout << "generated buckets" << std::endl;
      // create a new random number generator for each bucket
      parlay::sequence<parlay::random> newrands = parlay::tabulate(num_pivots, [&](size_t i) {
        return rnd.fork(i);
      });
      // recurse on all subproblems in parallel
      parlay::parallel_for(0, num_pivots, [&](size_t i) {
        if (buckets[i].size() > 0) {
          randomkary_clustering(G, Points, buckets[i], newrands[i], cluster_size, g, MSTDeg, num_pivots);
        }
      });
  }
}

  template <typename F>
  void simhash_clustering_wrapper(GraphI &G, PR &Points, size_t cluster_size,
                                 F f, long MSTDeg) {
    indexType num_pivots = 32;
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, Points.size());
    parlay::random rnd(uni(rng));
    auto active_indices =
        parlay::tabulate(Points.size(), [&](size_t i) { return i; });
    randomkary_clustering(G, Points, active_indices, rnd, cluster_size, f, MSTDeg, num_pivots);
  }

  template <typename F>
  void multiple_clustertrees(GraphI &G, PR &Points, long cluster_size,
                             long num_clusters, F f, long MSTDeg) {
    for (long i = 0; i < num_clusters; i++) {
      simhash_clustering_wrapper(G, Points, cluster_size, f, MSTDeg);
      std::cout << "Built cluster " << i+1 << " of " << num_clusters << std::endl;
    }
  }
};
                        

} // namespace parlayANN