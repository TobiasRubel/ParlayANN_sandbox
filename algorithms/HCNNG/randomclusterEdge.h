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
#include "../utils/graph.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"

namespace parlayANN {

template <typename Point, typename PointRange, typename indexType>
struct randompartition {
  using distanceType = typename Point::distanceType;
  using edge = std::pair<indexType, indexType>;
  using labelled_edge = std::pair<edge, distanceType>;
  using GraphI = Graph<indexType>;
  using PR = PointRange;

  randompartition() {}

  int generate_index(int N, int i) {
    return (N * (N - 1) - (N - i) * (N - i - 1)) / 2;
  }

  static void permute(parlay::sequence<size_t>& U,parlay::random& rnd) {
    for (size_t i = 0; i < U.size(); i++) {
        size_t j = i + rnd.ith_rand(i) % (U.size() - i);
        std::swap(U[i], U[j]);
    }
  }
 
 template <typename F>
  void random_partition_clustering(GraphI &G, PR &Points,
                         parlay::sequence<size_t>& active_indices,
                         parlay::random& gen, size_t cluster_size, F g,
                         long MSTDeg) {

    if (active_indices.size() <= cluster_size)
            g(G, Points, active_indices, MSTDeg);

    else {

      permute(active_indices,gen);
      parlay::sequence<size_t> left_sp;
      parlay::sequence<size_t> right_sp;
      for (size_t i = 0; i < active_indices.size(); i++) {
        if (i < active_indices.size() / 2)
          left_sp.push_back(active_indices[i]);
        else
          right_sp.push_back(active_indices[i]);
      }

      //std::cout << "Left size: " << left_sp.size() << " Right size: " << right_sp.size() << std::endl;
      // recurse on the two halves
      parlay::par_do(
          [&]() {
            random_partition_clustering(G, Points, left_sp, gen, cluster_size, g, MSTDeg);
          },
          [&]() {
            random_partition_clustering(G, Points, right_sp, gen, cluster_size, g, MSTDeg);
          });

    }
  
  }

  template <typename F>
  void random_partition_clustering_wrapper(GraphI &G, PR &Points,
                                 size_t cluster_size, F f, long MSTDeg) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, Points.size());
    parlay::random rnd(uni(rng));
    auto active_indices =
        parlay::tabulate(Points.size(), [&](size_t i) { return i; });

    random_partition_clustering(G, Points, active_indices, rnd, cluster_size, f, MSTDeg);
    }
  template <typename F>
  void multiple_clustertrees(GraphI &G, PR &Points, long cluster_size,
                             long num_clusters, F f, long MSTDeg) {
    for (long i = 0; i < num_clusters; i++) {
      random_partition_clustering_wrapper(G, Points, cluster_size, f, MSTDeg);
      std::cout << "Built cluster " << i+1 << " of " << num_clusters << std::endl;
    }
  }
};

} // end namespace
