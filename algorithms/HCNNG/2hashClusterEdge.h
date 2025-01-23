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
struct hashcluster2 {
  using distanceType = typename Point::distanceType;
  using edge = std::pair<indexType, indexType>;
  using labelled_edge = std::pair<edge, distanceType>;
  using GraphI = Graph<indexType>;
  using PR = PointRange;

  hashcluster2() {}

  int generate_index(int N, int i) {
    return (N * (N - 1) - (N - i) * (N - i - 1)) / 2;
  } 
 
 template <typename F>
  void hash_clustering_2(GraphI &G, PR &Points,
                         parlay::sequence<size_t>& active_indices,
                         parlay::random_generator& gen, std::normal_distribution<float>& dis, size_t cluster_size, F g,
                         long MSTDeg, size_t depth) {

    if (active_indices.size() <= cluster_size)
            g(G, Points, active_indices, MSTDeg);

    else {
      size_t d = Points.dimension();
      auto pivot = parlay::tabulate(d, [&](size_t i) {
        auto r = gen[(depth*d)+i];
        return dis(r);
        });

      auto left_sp = parlay::filter(
          parlay::make_slice(active_indices), [&](size_t ind) {
            float dot = 0;
            for (size_t k = 0; k < d; k++) {
              dot += Points[ind][k] * pivot[k];
            }
            return dot < 0 ? 0 : 1;
          });
      auto right_sp = parlay::filter(
          parlay::make_slice(active_indices), [&](size_t ind) {
            float dot = 0;
            for (size_t k = 0; k < d; k++) {
              dot += Points[ind][k] * pivot[k];
            }
            return dot >= 0 ? 0 : 1;
          });
      //std::cout << "Left size: " << left_sp.size() << " Right size: " << right_sp.size() << std::endl;
      // recurse on the two halves
      parlay::par_do(
          [&]() {
            hash_clustering_2(G, Points, left_sp, gen, dis, cluster_size, g, MSTDeg, depth+1);
          },
          [&]() {
            hash_clustering_2(G, Points, right_sp, gen, dis, cluster_size, g, MSTDeg, depth+1);
          });

    }
  
  }

  template <typename F>
  void hash_clustering_2_wrapper(GraphI &G, PR &Points,
                                 size_t cluster_size, F f, long MSTDeg, size_t repl) {
    auto active_indices =
        parlay::tabulate(Points.size(), [&](size_t i) { return i; });
    parlay::random_generator gen(repl);
    std::normal_distribution<float> dis(0.0, 1.0);

    
    hash_clustering_2(G, Points, active_indices, gen, dis, cluster_size, f, MSTDeg,1);
    }
  template <typename F>
  void multiple_clustertrees(GraphI &G, PR &Points, long cluster_size,
                             long num_clusters, F f, long MSTDeg) {
    for (long i = 0; i < num_clusters; i++) {
      hash_clustering_2_wrapper(G, Points, cluster_size, f, MSTDeg,i);
      std::cout << "Built cluster " << i+1 << " of " << num_clusters << std::endl;
    }
  }
};

} // end namespace
