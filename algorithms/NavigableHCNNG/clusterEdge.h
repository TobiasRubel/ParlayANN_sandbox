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
#include "hcnng_utils.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "topn.h"

namespace parlayANN {

std::atomic<size_t> leaf_count = 0;

std::pair<size_t, size_t> select_two_random(parlay::sequence<uint32_t> &ids,
                                            parlay::random &rnd) {
  size_t first_index = rnd.ith_rand(0) % ids.size();
  size_t second_index_unshifted = rnd.ith_rand(1) % (ids.size() - 1);
  size_t second_index = (second_index_unshifted < first_index)
                            ? second_index_unshifted
                            : (second_index_unshifted + 1);

  return {ids[first_index], ids[second_index]};
}

template <typename Point, typename PointRange, typename indexType>
struct cluster {
  using distanceType = typename Point::distanceType;
  using edge = std::pair<indexType, indexType>;
  using labelled_edge = std::pair<edge, distanceType>;
  using GraphI = Graph<indexType>;
  using PR = PointRange;
  using Bucket = parlay::sequence<uint32_t>;

  cluster() {}

  int generate_index(int N, int i) {
    return (N * (N - 1) - (N - i) * (N - i - 1)) / 2;
  }

  template <typename F>
  void recurse(GraphI &G, PR &Points, parlay::sequence<uint32_t> &ids,
               parlay::random &rnd, size_t cluster_size, F f, indexType first,
               indexType second) {
    // Split points based on which of the two points are closer.
    auto closer_first =
        parlay::filter(parlay::make_slice(ids), [&](size_t ind) {
          distanceType dist_first = Points[ind].distance(Points[first]);
          distanceType dist_second = Points[ind].distance(Points[second]);
          return dist_first <= dist_second;
        });

    auto closer_second =
        parlay::filter(parlay::make_slice(ids), [&](size_t ind) {
          distanceType dist_first = Points[ind].distance(Points[first]);
          distanceType dist_second = Points[ind].distance(Points[second]);
          return dist_second < dist_first;
        });

    auto left_rnd = rnd.fork(0);
    auto right_rnd = rnd.fork(1);

    parlay::par_do(
        [&]() {
          random_clustering(G, Points, closer_first, left_rnd, cluster_size, f);
        },
        [&]() {
          random_clustering(G, Points, closer_second, right_rnd, cluster_size,
                            f);
        });
  }

  template <typename F>
  void random_clustering(GraphI &G, PR &Points, parlay::sequence<uint32_t> &ids,
                         parlay::random &rnd, size_t cluster_size, F g) {
    if (ids.size() <= cluster_size)
      g(G, Points, ids, MSTDeg);
    else {
      auto [f, s] = select_two_random(ids, rnd);
      if (Points[f] == Points[s]) {
        parlay::sequence<uint32_t> closer_first;
        parlay::sequence<uint32_t> closer_second;
        for (int i = 0; i < ids.size(); i++) {
          if (i < ids.size() / 2)
            closer_first.push_back(ids[i]);
          else
            closer_second.push_back(ids[i]);
        }
        auto left_rnd = rnd.fork(0);
        auto right_rnd = rnd.fork(1);
        parlay::par_do(
            [&]() {
              random_clustering(G, Points, closer_first, left_rnd, cluster_size,
                                g);
            },
            [&]() {
              random_clustering(G, Points, closer_second, right_rnd,
                                cluster_size, g);
            });
      } else {
        recurse(G, Points, ids, rnd, cluster_size, g, f, s);
      }
    }
  }

  template <typename F>
  void random_clustering_wrapper(GraphI &G, PR &Points, size_t cluster_size,
                                 F f) {
    std::random_device rd;
    std::mt19937 rng(seed);
    seed = parlay::hash64(seed);
    std::uniform_int_distribution<int> uni(0, Points.size());
    parlay::random rnd(uni(rng));
    auto ids = parlay::tabulate(Points.size(), [&](uint32_t i) { return i; });
    random_clustering(G, Points, ids, rnd, cluster_size, f);
  }

  // Returns a collection of leaf buckets.
  std::vector<Bucket> RecursivelySketch(PR &Points, Bucket &ids,
                                        long cluster_size, int depth,
                                        int fanout) {
    //		std::cout << "In recursively sketch, ids.size = " << ids.size() << "
    //cluster size = " << cluster_size << std::endl;
    if (ids.size() <= cluster_size) {
      return {ids};
    }

    // sample leaders
    size_t num_leaders =
        depth == 0 ? TOP_LEVEL_NUM_LEADERS : ids.size() * FRACTION_LEADERS;
    num_leaders = std::min<size_t>(num_leaders, MAX_NUM_LEADERS);
    num_leaders = std::max<size_t>(num_leaders, 3);
    Bucket leaders(num_leaders);
    std::mt19937 prng(seed);
    seed = parlay::hash64(seed);
    std::sample(ids.begin(), ids.end(), leaders.begin(), leaders.size(), prng);

    //		std::cout << "after sampling: leaders size " << leaders.size() <<
    //std::endl;
    auto leader_points = PointRange(Points, leaders);
    std::vector<Bucket> clusters(leaders.size());
    //		std::cout << "Computing clusters" << std::endl;

    {  // less readable than map + zip + flatten, but at least it's as efficient
       // as possible for fanout = 1
      parlay::sequence<std::pair<uint32_t, uint32_t>> flat(ids.size() * fanout);

      parlay::parallel_for(0, ids.size(), [&](size_t i) {
        uint32_t point_id = ids[i];
        auto cl =
            ClosestLeaders(Points, leader_points, point_id, fanout).Take();
        for (int j = 0; j < fanout; ++j) {
          flat[i * fanout + j] = std::make_pair(cl[j].second, point_id);
        }
      });

      auto pclusters = parlay::group_by_index(flat, leaders.size());
      // copy clusters from parlay::sequence to std::vector
      parlay::parallel_for(0, pclusters.size(), [&](size_t i) {
        clusters[i] = Bucket(pclusters[i].begin(), pclusters[i].end());
      });
    }
    //		std::cout << "Assigned to closest leaders" << std::endl;

    leaders.clear();
    // TODO: clear leader_points PR.

    std::vector<Bucket> buckets;
    std::sort(
        clusters.begin(), clusters.end(),
        [&](const auto &b1, const auto &b2) { return b1.size() > b2.size(); });
    // Merge
    while (!clusters.empty() && clusters.back().size() < MIN_CLUSTER_SIZE) {
      if (buckets.empty() || clusters.back().size() + buckets.back().size() >
                                 MAX_MERGED_CLUSTER_SIZE) {
        buckets.emplace_back();
      }
      // merge small clusters together -- and already store them in the return
      // buckets this will add some stupid long range edges. but hopefully this
      // is better than having some isolated nodes. another fix could be to do a
      // brute-force comparison with all points in the bucket one recursion
      // level higher Caveat --> we have to hold the data structures for the
      // graph already
      for (const auto id : clusters.back()) {
        buckets.back().push_back(id);
      }
      clusters.pop_back();
    }
    //		std::cout << "Done merging" << std::endl;

    // recurse on clusters
    SpinLock bucket_lock;
    parlay::parallel_for(
        0, clusters.size(),
        [&](size_t cluster_id) {
          std::vector<Bucket> recursive_buckets;
          if (depth > MAX_DEPTH ||
              (depth > CONCERNING_DEPTH &&
               clusters[cluster_id].size() >
                   TOO_SMALL_SHRINKAGE_FRACTION * ids.size())) {
            // Base case for duplicates and near-duplicates. Split the buckets
            // randomly
            auto ids_copy = clusters[cluster_id];
            std::mt19937 prng(seed + depth + ids.size());
            std::shuffle(ids_copy.begin(), ids_copy.end(), prng);
            for (size_t i = 0; i < ids_copy.size(); i += MAX_CLUSTER_SIZE) {
              auto &new_bucket = recursive_buckets.emplace_back();
              for (size_t j = 0; j < MAX_CLUSTER_SIZE; ++j) {
                new_bucket.push_back(ids_copy[j]);
              }
            }
          } else {
            // The normal case
            recursive_buckets =
                RecursivelySketch(Points, clusters[cluster_id], cluster_size,
                                  depth + 1, /*fanout=*/1);
          }

          bucket_lock.lock();
          buckets.insert(buckets.end(), recursive_buckets.begin(),
                         recursive_buckets.end());
          bucket_lock.unlock();
        },
        1);

    return buckets;
  }

  template <typename F>
  auto recursively_sketch_wrapper(GraphI &G, PR &Points, size_t cluster_size,
                                  F f) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, Points.size());
    parlay::random rnd(uni(rng));
    auto ids = parlay::tabulate(Points.size(), [&](uint32_t i) { return i; });
    auto buckets = RecursivelySketch(Points, ids, cluster_size, 0, FANOUT);
    std::cout << "Computed buckets!" << std::endl;
    // Build on each bucket.
    parlay::parallel_for(0, buckets.size(),
                         [&](size_t i) { f(G, Points, buckets[i], MSTDeg); });
  }

  template <typename F>
  void multiple_clustertrees(GraphI &G, PR &Points, long cluster_size,
                             long num_clusters, F f) {
    for (long i = 0; i < num_clusters; i++) {
      if (MULTI_PIVOT) {
        recursively_sketch_wrapper(G, Points, cluster_size, f);
      } else {
        random_clustering_wrapper(G, Points, cluster_size, f);
      }
      std::cout << "Built cluster " << i << " of " << num_clusters << std::endl;
      std::cout << "Leaf count: " << leaf_count << std::endl;
    }
  }

  size_t seed = 555;
  double FRACTION_LEADERS = 0.005;
  size_t TOP_LEVEL_NUM_LEADERS = 950;
  size_t MAX_NUM_LEADERS = 1500;
  size_t MAX_CLUSTER_SIZE = 5000;
  size_t MIN_CLUSTER_SIZE = 50;
  size_t MAX_MERGED_CLUSTER_SIZE = 2500;
  int REPETITIONS = 3;
  int FANOUT = 1;
  int MAX_DEPTH = 14;
  int CONCERNING_DEPTH = 10;
  double TOO_SMALL_SHRINKAGE_FRACTION = 0.8;
  size_t MSTDeg = 3;
  // Set to true to do k-way pivoting
  bool MULTI_PIVOT = false;
};

}  // namespace parlayANN
