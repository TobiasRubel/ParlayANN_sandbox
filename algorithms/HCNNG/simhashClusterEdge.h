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

template <typename Point, typename PointRange, typename indexType>
struct simhashcluster {
  using distanceType = typename Point::distanceType;
  using edge = std::pair<indexType, indexType>;
  using labelled_edge = std::pair<edge, distanceType>;
  using GraphI = Graph<indexType>;
  using PR = PointRange;

  simhashcluster() {}

  int generate_index(int N, int i) {
    return (N * (N - 1) - (N - i) * (N - i - 1)) / 2;
  }

  template <typename F>
  void simhash_clustering(GraphI &G, PR &Points,
                         parlay::sequence<size_t>& active_indices,
                         size_t cluster_size, F g,
                         long MSTDeg, size_t num_bits) {

    if (active_indices.size() <= cluster_size)
            g(G, Points, active_indices, MSTDeg);

    else {
      auto d = Points.dimension();
      auto H = simhash<Point, PointRange, indexType>(num_bits, d);
      //hash each point, adding it to a subproblem based on the hash
      parlay::sequence<parlay::sequence<size_t>> buckets(1 << num_bits);
      // parlay::parallel_for(0, active_indices.size(), [&](size_t i) {
      //   auto h = H.hash(Points[active_indices[i]]);
      //   buckets[h].push_back(active_indices[i]);
      // });
      for (size_t i = 0; i < active_indices.size(); i++) {
        auto h = H.hash(Points[active_indices[i]]);
        buckets[h].push_back(active_indices[i]);
      }

      std::cout << "generated buckets" << std::endl;
      std::cout << "number of populated buckets: " << parlay::count_if(buckets, [&](auto x) {return x.size() > 0;}) << std::endl;

      // recurse on all subproblems in parallel
      parlay::parallel_for(0, 1 << num_bits, [&](size_t i) {
        if (buckets[i].size() > 0) {
          simhash_clustering(G, Points, buckets[i], cluster_size, g, MSTDeg, num_bits);
        }
      });
  }
}

  template <typename F>
  void simhash_clustering_wrapper(GraphI &G, PR &Points, size_t cluster_size,
                                 F f, long MSTDeg) {
    indexType num_bits = 4; // do not modify this to be much bigger unless you also fix the representation of the hash (which is a vector and grows exponentially with num_bits)
    auto n = Points.size();
    auto active_indices = parlay::tabulate(n, [&](size_t i) { return i; });
    //testing whether simhash clustering works
    // std::cout << "Building simhash cluster" << std::endl;
    // auto d = Points.dimension();
    // std::cout << "Dimension: " << d << std::endl;
    // std::cout << "Number of bits: " << num_bits << std::endl;
    // auto H = simhash<Point, PointRange, indexType>(num_bits, d);
    // std::cout << "Hashing points" << std::endl;
    // auto p0 = H.hash(Points[0]);
    // auto p1 = H.hash(Points[1]);
    // std::cout << "p0 = " << p0 << std::endl;
    // std::cout << "p1 = " << p1 << std::endl;
    // std::cout << "Hashed points" << std::endl;
    //exit(0);


    simhash_clustering(G, Points, active_indices, cluster_size, f, MSTDeg, num_bits);
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