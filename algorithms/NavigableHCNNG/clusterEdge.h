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
#include <vector>

#include "../utils/beamSearch.h"
#include "../utils/graph.h"
#include "../utils/types.h"
#include "../vamana/index.h"
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

  using utils = cluster_utils<Point, PointRange, indexType>;

  cluster() {}

  int generate_index(int N, int i) {
    return (N * (N - 1) - (N - i) * (N - i - 1)) / 2;
  }

  void recurse(GraphI &G, PR &Points, parlay::sequence<uint32_t> &ids,
               parlay::random &rnd, size_t cluster_size, indexType first,
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
          random_clustering(G, Points, closer_first, left_rnd, cluster_size);
        },
        [&]() {
          random_clustering(G, Points, closer_second, right_rnd, cluster_size);
        });
  }

  void random_clustering(GraphI &G, PR &Points, parlay::sequence<uint32_t> &ids,
                         parlay::random &rnd, size_t cluster_size) {
    if (ids.size() <= cluster_size) {
      RunLeaf(G, Points, ids);
    } else {
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
              random_clustering(G, Points, closer_first, left_rnd,
                                cluster_size);
            },
            [&]() {
              random_clustering(G, Points, closer_second, right_rnd,
                                cluster_size);
            });
      } else {
        recurse(G, Points, ids, rnd, cluster_size, f, s);
      }
    }
  }

  void random_clustering_wrapper(GraphI &G, PR &Points, size_t cluster_size) {
    std::random_device rd;
    std::mt19937 rng(SEED);
    SEED = parlay::hash64(SEED);
    std::uniform_int_distribution<int> uni(0, Points.size());
    parlay::random rnd(uni(rng));
    auto ids = parlay::tabulate(Points.size(), [&](uint32_t i) { return i; });
    parlay::internal::timer t;
    t.start();
    random_clustering(G, Points, ids, rnd, cluster_size);
    t.next("tree time");
  }

  std::vector<Bucket> PruneSLINK(PR &Points, Bucket &ids, size_t num_leaders,
                                 size_t seed) {
    // sample leaders
    std::cout << "Leader sampling seed = " << seed << std::endl;

    // size_t num_leaders = TOP_LEVEL_NUM_LEADERS;
    Bucket leaders(num_leaders);
    std::mt19937 prng(seed);
    // size_t next_seed = parlay::hash64(seed);
    std::sample(ids.begin(), ids.end(), leaders.begin(), leaders.size(), prng);
    std::vector<Bucket> clusters(leaders.size() * FANOUT);
    auto leader_points = PointRange(Points, leaders);
    // create graph on leaders to be ANN index...
    GraphI G(32, leaders.size());
    auto leader_ids =
        parlay::tabulate(num_leaders, [&](uint32_t i) { return i; });

    // this uses QuadPrune to build the index
    // DistMatQuadPrune(G, leader_points,leader_ids);
    // this uses vamana to build index
    stats<indexType> sbuild(size_t(leaders.size()));
    BuildParams BPb(20, 32, ALPHA, 2, false);
    using findex = knn_index<PointRange, PointRange, indexType>;
    findex I(BPb);
    I.build_index(G, leader_points, leader_points, sbuild);
    QueryParams QP((long)3 * (FANOUT + 1), 3 * (FANOUT + 2), (double)ALPHA,
                   (long)leaders.size(),
                   3 * (FANOUT + 2));  // how to set this with fanout?
    stats<indexType> s((size_t)Points.size());
    // std::cout << "built graph on leaders..." << std::endl;
    //  get nearest neighbors for each point
    auto bss = beamSearchZero(Points, G, leader_points, s, QP);

    // now we'll try to assign points to buckets
    // for now we will use a greedy strategy:

    parlay::sort_inplace(
        bss, [](auto const &a, auto const &b) { return a.second < b.second; });

    std::vector<int> matched(Points.size(), 0);

    for (size_t i = 0; i < bss.size(); i++) {
      // Each edge is stored as: ((source, target), distance)
      auto [edge, dist] = bss[i];
      auto [source, target] = edge;

      // Check if this source is not matched and that the target's cluster is
      // not full.
      if ((matched[source] < FANOUT) &&
          clusters[target].size() < MAX_CLUSTER_SIZE) {
        auto offset = matched[source];
        clusters[(offset * num_leaders) + target].push_back(source);
        ++matched[source];
      }
    }
    auto unmatched =
        parlay::filter(ids, [&](auto i) { return matched[i] < FANOUT; });
    // If any points remain unmatched after 10 rounds, randomly assign them.

    if (unmatched.size() != 0) {
      std::cout << "After matching, " << unmatched.size()
                << " points remain unmatched. Randomly assigning them."
                << std::endl;
      std::mt19937 rng(seed +
                       12345);  // use a different seed for random assignment
      std::uniform_int_distribution<int> bucketDist(0, num_leaders);
      for (auto i : unmatched) {
        auto offset = matched[i];
        int bucket = bucketDist(rng);
        clusters[(offset * num_leaders) + bucket].push_back(i);
        ++matched[i];
      }
    }
    std::cout << "matching complete..." << std::endl;

    std::cout << "merging small clusters..." << std::endl;
    for (size_t cid = 0; cid < clusters.size(); cid++) {
      // Get the leader associated with this bucket.
      size_t leader = cid % num_leaders;

      // Check the bucket we want to merge, not the leader’s primary bucket.
      if (clusters[cid].size() > MIN_CLUSTER_SIZE) continue;
      // std::cout << "merging " << cid << " with leader id " << leader << " and
      // size " << clusters[cid].size() << std::endl;
      std::vector<bool> seen(num_leaders, false);
      // seen[leader] = true;
      std::priority_queue<std::pair<distanceType, int>,
                          std::vector<std::pair<distanceType, int>>,
                          std::greater<>>
          frontier;

      // Push the leader
      frontier.push({(distanceType)0, leader});

      while (!frontier.empty()) {
        auto [dist, cand] = frontier.top();
        // std:: cout << "considering " << cand << std::endl;
        frontier.pop();
        if (seen[cand]) continue;
        seen[cand] = true;

        // search all fanout buckets for mergable cluster
        for (size_t fan = 0; fan < FANOUT; fan++) {
          size_t candclustid = (fan * num_leaders) + cand;
          size_t csize = clusters[candclustid].size();
          if ((candclustid != cid) && (csize != 0) &&
              (csize + clusters[cid].size() <= MAX_CLUSTER_SIZE)) {
            // Merge without duplicates (todo: more efficiently)
            for (auto elem : clusters[cid]) {
              if (std::find(clusters[candclustid].begin(),
                            clusters[candclustid].end(),
                            elem) == clusters[candclustid].end()) {
                clusters[candclustid].push_back(elem);
              }
            }
            clusters[cid].clear();
            while (!frontier.empty()) frontier.pop();
            break;
          }
        }

        // Push the neighbors of cand into the frontier.
        for (auto nbrid = 0; nbrid < G[cand].size(); nbrid++) {
          auto nbr = G[cand][nbrid];
          if (!seen[nbr]) {
            distanceType ndist =
                leader_points[leader].distance(leader_points[nbr]);
            frontier.push({ndist, nbr});
          }
        }
      }
    }

    // for (auto i = 0; i < 20; i++) {
    //   std::cout << clusters[i].size() << " ";
    // }
    return clusters;
  }

  auto PruneSLINK_wrapper(GraphI &G, PR &Points, size_t num_clusters,
                          size_t cluster_size) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, Points.size());
    parlay::random rnd(uni(rng));
    auto ids = parlay::tabulate(Points.size(), [&](uint32_t i) { return i; });
    parlay::internal::timer t;
    t.start();
    SEED = parlay::hash64(SEED);
    auto buckets = PruneSLINK(Points, ids, num_clusters, SEED);
    t.next("buckets time");
    std::cout << "Computed buckets!" << std::endl;
    // Build on each bucket.
    parlay::parallel_for(0, buckets.size(), [&](size_t i) {
      if (buckets[i].size() != 0) RunLeaf(G, Points, buckets[i]);
    });
    t.next("build leaf time");
  }

  // Returns a collection of leaf buckets.
  std::vector<Bucket> RecursivelySketch(PR &Points, Bucket &ids,
                                        long cluster_size, int depth,
                                        int fanout, size_t seed, int fanout_levels, std::vector<int> &fanout_scheme) {
    int granularity = 5000;
    if (ids.size() <= cluster_size) {
      return {ids};
    }

    // sample leaders
    size_t num_leaders =
        (depth == 0) ? TOP_LEVEL_NUM_LEADERS : ids.size() * FRACTION_LEADERS;
    num_leaders = std::min<size_t>(num_leaders, MAX_NUM_LEADERS);
    num_leaders = std::max<size_t>(num_leaders, 3);
    lock.lock();
    COMPARISONS += num_leaders * ids.size();
    lock.unlock();
    Bucket leaders(num_leaders);
    std::mt19937 prng(seed);
    size_t next_seed = parlay::hash64(seed);
    std::sample(ids.begin(), ids.end(), leaders.begin(), leaders.size(), prng);

    //		std::cout << "after sampling: leaders size " << leaders.size()
    //<< std::endl;
    auto leader_points = PointRange(Points, leaders);
    (depth < fanout_scheme.size()) ? fanout = fanout_scheme[depth] : std::min<int>(fanout, (int) num_leaders);
    //fanout = std::min<int>(fanout, (int) num_leaders);
    // if (depth == 0) fanout = 6;
    // if (depth == 1) fanout = std::min<int>(3, (int) num_leaders);
    // if (depth == 2) fanout = std::min<int>(2, (int) num_leaders);
    parlay::internal::timer t;
    t.start();
    parlay::sequence<std::pair<uint32_t, uint32_t>> flat(ids.size() * fanout);
    if (ids.size() < granularity) {
      for (size_t i = 0; i < ids.size(); i++) {
        uint32_t point_id = ids[i];
        auto cl = closest_leaders(Points, leader_points, point_id, fanout);
        for (int j = 0; j < fanout; ++j) {
          flat[i * fanout + j] = std::make_pair(cl[j].first, point_id);
        }
      }
    } else{
      parlay::parallel_for(0, ids.size(), [&](size_t i) {
        uint32_t point_id = ids[i];
        auto cl = closest_leaders(Points, leader_points, point_id, fanout);
        for (int j = 0; j < fanout; ++j) {
          flat[i * fanout + j] = std::make_pair(cl[j].first, point_id);
        }
      });
    }
    parlay::sequence<Bucket> clusters = parlay::group_by_index(flat, leaders.size());
    if (depth == 0) {
      t.next("first level assign time");
    }

    leaders.clear();

    std::vector<Bucket> buckets;
    std::sort(
        clusters.begin(), clusters.end(),
        [&](const auto &b1, const auto &b2) { return b1.size() > b2.size(); });
    // Merge

    while (clusters.size() > 1 && clusters.back().size() < MIN_CLUSTER_SIZE) {
      size_t num_to_merge = 1;
      size_t merged_size = clusters.back().size();
      // Try to merge additional clusters from the end without exceeding MAX_MERGED_CLUSTER_SIZE.
      while (num_to_merge < clusters.size() &&
            merged_size < MIN_CLUSTER_SIZE &&
            merged_size + clusters[clusters.size() - num_to_merge - 1].size() <= MAX_MERGED_CLUSTER_SIZE) {
        merged_size += clusters[clusters.size() - num_to_merge - 1].size();
        num_to_merge++;
      }
      
      // Create a new bucket with the computed merged size.
      auto new_bucket = parlay::sequence<uint32_t>::uninitialized(merged_size);
      size_t pos = 0;
      // Merge the last 'num_to_merge' clusters in order:
      // Copy clusters from the index clusters.size() - num_to_merge up to clusters.size()-1.
      for (size_t i = clusters.size() - num_to_merge; i < clusters.size(); i++) {
        size_t cluster_size = clusters[i].size();
        std::memcpy(new_bucket.begin() + pos,
                    clusters[i].begin(),
                    cluster_size * sizeof(uint32_t));
        pos += cluster_size;
      }
      
      // Remove the merged clusters from clusters.
      for (size_t i = 0; i < num_to_merge; i++) {
        clusters.pop_back();
      }
      //deduplicate the new bucket 
      new_bucket = parlay::remove_duplicates(new_bucket);

      //std::unordered_set<uint32_t> seen;
      
      buckets.push_back(std::move(new_bucket));
    }


    // recurse on clusters
    parlay::sequence<std::vector<Bucket>> rec_buckets(clusters.size());
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
              size_t start = i;
              size_t end = std::min(start + MAX_CLUSTER_SIZE, ids_copy.size());
              for (size_t j = i; j < end; ++j) {
                new_bucket.push_back(ids_copy[j]);
              }
            }
          } else {
            auto local_fanout = 1;
            (depth < fanout_levels) ? local_fanout = fanout : local_fanout = 1;
            // The normal case
            recursive_buckets = RecursivelySketch(
                Points, clusters[cluster_id], cluster_size, depth + 1,
                /*fanout=*/local_fanout, next_seed + cluster_id, fanout_levels, fanout_scheme);
          }
          rec_buckets[cluster_id] = std::move(recursive_buckets);
        },
        1);

    for (size_t cluster_id = 0; cluster_id < clusters.size(); ++cluster_id) {
      while (!rec_buckets[cluster_id].empty()) {
        size_t index = rec_buckets[cluster_id].size();
        buckets.push_back(std::move(rec_buckets[cluster_id][index - 1]));
        rec_buckets[cluster_id].pop_back();
      }
    }

    return buckets;
  }

  auto recursively_sketch_wrapper(GraphI &G, PR &Points, size_t cluster_size) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, Points.size());
    parlay::random rnd(uni(rng));
    auto ids = parlay::tabulate(Points.size(), [&](uint32_t i) { return i; });
    parlay::internal::timer t;

    int fanout_per = FANOUT_PER_LEVEL;
    int fanout_levels = 0;
    auto local_fanout = FANOUT;
    if (fanout_per != 0) {
      // identify the number of levels to fanout at the per_level_fanout without exceeding fanout.
      // we are using a constant rate of fanout, which may not be ideal. For power users, we may want to allow
      // custom fanout distributions.
      // there may also be some way of dynamically adjusting the fanout within the RecursivelySketch method based on
      // distributional properties of the data. 
      local_fanout = fanout_per;
      int fprod = fanout_per;
      while (fprod*fanout_per <= FANOUT) {
        fprod *= fanout_per;
        fanout_levels++;
      }
    } 
    std::cout << "number of levels to fanout: " << fanout_levels + 1 << std::endl;
    std::cout << "fanout per level: " << local_fanout << std::endl;
    std::cout << "total fanout: " << std::pow(local_fanout,(fanout_levels+1)) << std::endl;
    std::cout << "total fanout allowance: " << FANOUT << std::endl;

    t.start();
    auto buckets =
        RecursivelySketch(Points, ids, cluster_size, 0, local_fanout, SEED, fanout_levels, FANOUT_SCHEME);
    SEED = parlay::hash64(SEED);
    t.next("buckets time");
    std::cout << "Computed buckets!" << std::endl;
    // Build on each bucket. // note the hack of checking size. This is a hack to get around the RecursivelySketch routine producing empty clusters... Looking into it.
    parlay::parallel_for(0, buckets.size(),
                         [&](size_t i) { if (buckets[i].size() >= MIN_CLUSTER_SIZE) RunLeaf(G, Points, buckets[i]); });
    t.next("build leaf time");
  }

  void multiple_clustertrees(GraphI &G, PR &Points, long cluster_size,
                             long num_clusters) {
    for (long i = 0; i < num_clusters; i++) {
      if (MULTI_PIVOT) {
        recursively_sketch_wrapper(G, Points, cluster_size);
      } else {
        // random_clustering_wrapper(G, Points, cluster_size);
        double t = (double)i / (num_clusters - 1);
        // size_t num_leaders = num_clusters > 1
        //   ? TOP_LEVEL_NUM_LEADERS / 5 + t * (TOP_LEVEL_NUM_LEADERS -
        //   TOP_LEVEL_NUM_LEADERS / 5) : TOP_LEVEL_NUM_LEADERS;
        size_t num_leaders = TOP_LEVEL_NUM_LEADERS;
        std::cout << "TOP_LEVEL_NUM_LEADERS: " << TOP_LEVEL_NUM_LEADERS
                  << std::endl;
        std::cout << "t: " << t << std::endl;
        std::cout << "Building cluster with " << num_leaders << " leaders"
                  << std::endl;
        PruneSLINK_wrapper(G, Points, num_leaders, cluster_size);
      }
      std::cout << "Built cluster " << i << " of " << num_clusters << std::endl;
      std::cout << "Leaf count: " << leaf_count << std::endl;
      std::cout << "# of comparisons: " << COMPARISONS << std::endl;
    }
  }

  /*  Leaf code  */
  // parameters dim and K are just to interface with the cluster tree code

  void RunLeaf(GraphI &G, PR &Points,
               parlay::sequence<uint32_t> &active_indices) {
    if (LEAF_ALG == "VamanaLeaf") {
      VamanaLeaf(G, Points, active_indices);
    } else if (LEAF_ALG == "MSTk") {
      MSTk(G, Points, active_indices);
    } else if (LEAF_ALG == "QuadPrune") {
      QuadPrune(G, Points, active_indices);
    } else if (LEAF_ALG == "DistMatQuadPrune") {
      DistMatQuadPrune(G, Points, active_indices);
    } else {
      std::cout << "Unknown leaf method: " << LEAF_ALG << std::endl;
      exit(0);
    }
  }

  void VamanaLeaf(GraphI &G, PR &Points,
                  parlay::sequence<uint32_t> &active_indices) {
    BuildParams BP;
    BP.R = MST_DEG;
    BP.L = MST_DEG * 2;
    BP.alpha = ALPHA;
    BP.num_passes = 2;
    BP.single_batch = 0;

    auto edges =
        run_vamana_on_indices(active_indices, Points, BP, /*parallel=*/true);
    utils::process_edges(G, std::move(edges));

    lock.lock();
    START_POINTS.push_back(active_indices[0]);
    lock.unlock();

    leaf_count++;
  }

  void QuadPrune(GraphI &G, PR &Points,
                 parlay::sequence<uint32_t> &active_indices) {
    BuildParams BP;
    BP.R = MST_DEG;
    BP.alpha = ALPHA;
    auto edges = run_quadprune_on_indices(active_indices, Points, BP);
    utils::process_edges(G, std::move(edges));

    lock.lock();
    START_POINTS.push_back(active_indices[0]);
    lock.unlock();

    leaf_count++;
  }

  void DistMatQuadPrune(GraphI &G, PR &Points,
                        parlay::sequence<uint32_t> &active_indices) {
    BuildParams BP;
    BP.R = MST_DEG;
    BP.alpha = ALPHA;
    auto edges = distmat_quadprune(active_indices, Points, BP);
    utils::process_edges(G, std::move(edges));

    if (active_indices.size() > 0) {
      lock.lock();
      START_POINTS.push_back(active_indices[0]);
      lock.unlock();
      leaf_count++;
    }
  }

  // parameters dim and K are just to interface with the cluster tree code
  void MSTk(GraphI &G, PR &Points, parlay::sequence<uint32_t> &active_indices) {
    lock.lock();
    START_POINTS.push_back(active_indices[0]);

    double extra_fraction = 0.01;

    // std::mt19937 prng(parlay::hash64(active_indices[0]));
    // std::sample(active_indices.begin(), active_indices.end(),
    // leaders.begin(), leaders.size(), prng);
    lock.unlock();

    // preprocessing for Kruskal's
    size_t N = active_indices.size();
    long dim = Points.dimension();
    DisjointSet disjset(N);
    size_t m = 10;
    auto less = [&](labelled_edge a, labelled_edge b) {
      return a.second < b.second;
    };
    parlay::sequence<labelled_edge> candidate_edges(N * m, utils::kNullEdge);
    parlay::parallel_for(0, N, [&](size_t i) {
      std::priority_queue<labelled_edge, std::vector<labelled_edge>,
                          decltype(less)>
          Q(less);
      for (indexType j = i + 1; j < N; j++) {
        distanceType dist_ij =
            Points[active_indices[i]].distance(Points[active_indices[j]]);
        if (Q.size() >= m) {
          distanceType topdist = Q.top().second;
          if (dist_ij < topdist) {
            labelled_edge e;
            e = std::make_pair(std::make_pair(i, j), dist_ij);
            Q.pop();
            Q.push(e);
          }
        } else {
          labelled_edge e;
          e = std::make_pair(std::make_pair(i, j), dist_ij);
          Q.push(e);
        }
      }
      indexType limit = std::min(Q.size(), m);
      for (indexType j = 0; j < limit; j++) {
        candidate_edges[i * m + j] = Q.top();
        Q.pop();
      }
    });

    parlay::sort_inplace(candidate_edges, less);

    auto degrees =
        parlay::tabulate(active_indices.size(), [&](size_t i) { return 0; });
    parlay::sequence<edge> MST_edges = parlay::sequence<edge>();
    // modified Kruskal's algorithm
    for (indexType i = 0; i < candidate_edges.size(); i++) {
      // Since we sorted, any null edges form the suffix.
      if (candidate_edges[i].second == utils::kNullDist) break;
      labelled_edge e_l = candidate_edges[i];
      edge e = e_l.first;
      if ((disjset.find(e.first) != disjset.find(e.second)) &&
          (degrees[e.first] < MST_DEG) && (degrees[e.second] < MST_DEG)) {
        MST_edges.push_back(
            std::make_pair(active_indices[e.first], active_indices[e.second]));
        MST_edges.push_back(
            std::make_pair(active_indices[e.second], active_indices[e.first]));
        degrees[e.first] += 1;
        degrees[e.second] += 1;
        disjset._union(e.first, e.second);
      }
      if (i % N == 0) {
        if (disjset.is_full()) {
          break;
        }
      }
    }
    utils::process_edges(G, std::move(MST_edges));
    leaf_count++;
  }

  size_t SEED = 555;
  //double FRACTION_LEADERS = 0.005;
  double FRACTION_LEADERS = 0.0005;
  size_t TOP_LEVEL_NUM_LEADERS = 950;
  size_t MAX_NUM_LEADERS = 1500;
  size_t MAX_CLUSTER_SIZE = 5000;
  size_t MIN_CLUSTER_SIZE = 500;
  size_t MAX_MERGED_CLUSTER_SIZE = 2500;
  int MAX_DEPTH = 14;
  int CONCERNING_DEPTH = 10;
  double TOO_SMALL_SHRINKAGE_FRACTION = 0.8;
  size_t MST_DEG = 3;
  int FANOUT = 1;
  int FANOUT_PER_LEVEL = 1;
  // Set to true to do k-way pivoting
  bool MULTI_PIVOT = false;
  double ALPHA = 1;
  std::string LEAF_ALG = "VamanaLeaf";

  // 0 = all at the top level; 1 = n, n -1, ... 1 
  int PIVOT_STRAT = 0;

  std::vector<int> FANOUT_SCHEME = {};

  // Horrible hacks. Fix.
  SpinLock lock;
  parlay::sequence<uint32_t> START_POINTS;
  size_t COMPARISONS = 0;
};

}  // namespace parlayANN
