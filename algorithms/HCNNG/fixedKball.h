// adapted from Lars' code: https://github.com/larsgottesbueren/gp-ann/blob/main/src/knn_graph.h

#pragma once

#include <math.h>

#include <algorithm>
#include <functional>
#include <queue>
#include <random>
#include <set>
#include "fixedKballdefs.h"
#include "topn.h"
#include "../utils/graph.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
namespace parlayANN {

template <typename Point, typename PointRange, typename indexType>
struct kballcluster {
  using distanceType = typename Point::distanceType;
  using edge = std::pair<indexType, indexType>;
  using labelled_edge = std::pair<edge, distanceType>;
  using GraphI = Graph<indexType>;
  using PR = PointRange;
  using Bucket = std::vector<indexType>;

    int seed = 555;
    double FRACTION_LEADERS = 0.005;
    size_t TOP_LEVEL_NUM_LEADERS = 950;
    size_t MAX_NUM_LEADERS = 1500;
    size_t MAX_CLUSTER_SIZE = 1000;
    size_t MIN_CLUSTER_SIZE = 50;
    size_t MAX_MERGED_CLUSTER_SIZE = 500;
    int REPETITIONS = 3;
    int FANOUT = 3;
    int MAX_DEPTH = 14;
    int CONCERNING_DEPTH = 10;
    double TOO_SMALL_SHRINKAGE_FRACTION = 0.8;

    bool quiet = false;

    Timer timer;

  kballcluster() {}

  int generate_index(int N, int i) {
    return (N * (N - 1) - (N - i) * (N - i - 1)) / 2;
  }
    PR ExtractPoints(PR& points, const Bucket& ids) {
        PointSet bucket_points;
        bucket_points.d = points.d;
        bucket_points.n = ids.size();
        for (auto id : ids) {
            float* P = points.GetPoint(id);
            for (size_t d = 0; d < points.d; ++d) {
                bucket_points.coordinates.push_back(P[d]);
            }
        }
        return bucket_points;
    }

    std::vector<Bucket> RecursivelySketch(PR& points, const Bucket& ids, int depth, int fanout) {
        if (ids.size() <= MAX_CLUSTER_SIZE) {
            return { ids };
        }

        if (depth == 0) {
            timer.Start();
        }

        // sample leaders
        size_t num_leaders = depth == 0 ? TOP_LEVEL_NUM_LEADERS : ids.size() * FRACTION_LEADERS;
        num_leaders = std::min<size_t>(num_leaders, MAX_NUM_LEADERS);
        num_leaders = std::max<size_t>(num_leaders, 3);
        Bucket leaders(num_leaders);
        std::mt19937 prng(seed);
        std::sample(ids.begin(), ids.end(), leaders.begin(), leaders.size(), prng);

        PointSet leader_points = ExtractPoints(points, leaders);
        std::vector<Bucket> clusters(leaders.size());

        { // less readable than map + zip + flatten, but at least it's as efficient as possible for fanout = 1
            parlay::sequence<std::pair<indexType, indexType>> flat(ids.size() * fanout);

            parlay::parallel_for(0, ids.size(), [&](size_t i) {
                indexType point_id = ids[i];
                auto cl = ClosestLeaders(points, leader_points, point_id, fanout).Take();
                for (int j = 0; j < fanout; ++j) {
                    flat[i * fanout + j] = std::make_pair(cl[j].second, point_id);
                }
            });

            auto pclusters = parlay::group_by_index(flat, leaders.size());
            // copy clusters from parlay::sequence to std::vector
            parlay::parallel_for(0, pclusters.size(), [&](size_t i) { clusters[i] = Bucket(pclusters[i].begin(), pclusters[i].end()); });
        }

        leaders.clear();
        leaders.shrink_to_fit();
        leader_points.Drop();

        if (depth == 0) {
            double time = timer.Stop();
            if (!quiet)
                std::cout << "Closest leaders on top level took " << time << std::endl;
        }

        std::vector<Bucket> buckets;
        std::sort(clusters.begin(), clusters.end(), [&](const auto& b1, const auto& b2) { return b1.size() > b2.size(); });
        while (!clusters.empty() && clusters.back().size() < MIN_CLUSTER_SIZE) {
            if (buckets.empty() || clusters.back().size() + buckets.back().size() > MAX_MERGED_CLUSTER_SIZE) {
                buckets.emplace_back();
            }
            // merge small clusters together -- and already store them in the return buckets
            // this will add some stupid long range edges. but hopefully this is better than having some isolated nodes.
            // another fix could be to do a brute-force comparison with all points in the bucket one recursion level higher
            // Caveat --> we have to hold the data structures for the graph already
            for (const auto id : clusters.back()) {
                buckets.back().push_back(id);
            }
            clusters.pop_back(); 
        }

        // recurse on clusters
        SpinLock bucket_lock;
        parlay::parallel_for(
                0, clusters.size(),
                [&](size_t cluster_id) {
                    std::vector<Bucket> recursive_buckets;
                    if (depth > MAX_DEPTH || (depth > CONCERNING_DEPTH && clusters[cluster_id].size() > TOO_SMALL_SHRINKAGE_FRACTION * ids.size())) {
                        // Base case for duplicates and near-duplicates. Split the buckets randomly
                        auto ids_copy = clusters[cluster_id];
                        std::mt19937 prng(seed + depth + ids.size());
                        std::shuffle(ids_copy.begin(), ids_copy.end(), prng);
                        for (size_t i = 0; i < ids_copy.size(); i += MAX_CLUSTER_SIZE) {
                            auto& new_bucket = recursive_buckets.emplace_back();
                            for (size_t j = 0; j < MAX_CLUSTER_SIZE; ++j) {
                                new_bucket.push_back(ids_copy[j]);
                            }
                        }
                    } else {
                        // The normal case
                        recursive_buckets = RecursivelySketch(points, clusters[cluster_id], depth + 1, /*fanout=*/1);
                    }

                    bucket_lock.lock();
                    buckets.insert(buckets.end(), recursive_buckets.begin(), recursive_buckets.end());
                    bucket_lock.unlock();
                },
                1);

        return buckets;
    }


  template <typename F>
  void fixedkball_clustering(GraphI &G, PR &Points, size_t cluster_size,
                                 F f, long MSTDeg) {
        Bucket all_ids(points.n);
        std::iota(all_ids.begin(), all_ids.end(), 0);
        std::vector<Bucket> buckets;
        for (int rep = 0; rep < REPETITIONS; ++rep) {
            std::cout << "Sketching rep " << rep << std::endl;
            Timer timer2;
            timer2.Start();
            std::vector<Bucket> new_buckets = RecursivelySketch(points, all_ids, 0, FANOUT);
            std::cout << "Finished sketching rep. It took " << timer2.Stop() << " seconds." << std::endl;
            buckets.insert(buckets.end(), new_buckets.begin(), new_buckets.end());
        }
        std::cout << "Start bucket brute force" << std::endl;
        //return BruteForceBuckets(points, buckets, num_neighbors);
        for (bucket in buckets) {
            //create parlay seq from bucket
            parlay::sequence<size_t> active_indices(bucket.begin(), bucket.end());
            g(G, Points, active_indices, MSTDeg);
        }


  }

  template <typename F>
  void multiple_clustertrees(GraphI &G, PR &Points, long cluster_size,
                             long num_clusters, F f, long MSTDeg) {
    for (long i = 0; i < num_clusters; i++) {
      fixedkball_clustering(G, Points, cluster_size, f, MSTDeg);
      std::cout << "Built cluster " << i+1 << " of " << num_clusters << std::endl;
    }
  }
};

} // end namespace
