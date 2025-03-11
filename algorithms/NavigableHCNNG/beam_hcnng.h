#pragma once

#include <vector>
#include <mutex>
#include <random>
#include <limits>

#include "parlay/sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "parlay/internal/get_time.h"

#include "../utils/types.h"
#include "../utils/stats.h"
#include "../utils/graph.h"
#include "../utils/point_range.h"
#include "../vamana/index.h"
#include "hcnng_utils.h"

struct alignas(64) PaddedMutex {
    std::mutex m;
};

template <typename index_t, typename PointRangeType>
parlay::sequence<parlay::sequence<index_t>> cluster_prune(PointRangeType &points, parlay::sequence<index_t> &ids, size_t max_degree) {
    using value_t = typename PointRangeType::Point::distanceType;

    // Compute the distance matrix
    auto distances = parlay::sequence<value_t>::uninitialized(ids.size() * ids.size());
    for (size_t i = 0; i < ids.size(); i++) {
        distances[i * ids.size() + i] = 0;
        for (size_t j = i + 1; j < ids.size(); j++) {
            distances[i * ids.size() + j] = distances[j * ids.size() + i] = points[ids[i]].distance(points[ids[j]]);
        }
    }

    // Perform all-to-all prune using the distance matrix
    parlay::sequence<parlay::sequence<index_t>> neighbors(ids.size());
    for (size_t i = 0; i < ids.size(); i++) {
        parlay::sequence<index_t> candidates;
        candidates.reserve(ids.size() - 1);
        for (size_t j = 0; j < ids.size(); j++) {
            if (j != i) candidates.push_back(j);
        }
        std::sort(candidates.begin(), candidates.end(), [&] (index_t a, index_t b) { return distances[i * ids.size() + a] < distances[i * ids.size() + b]; });
        
        parlay::sequence<index_t> pruned_indices;
        pruned_indices.reserve(max_degree);
        for (size_t j = 0; j < max_degree && j < candidates.size(); j++) {
            index_t candidate = candidates[j];
            bool add = true;
            for (size_t k = 0; k < pruned_indices.size(); k++) {
                value_t dist = distances[i * ids.size() + candidate];
                if (dist < distances[i * ids.size() + pruned_indices[k]]) {
                    add = false;
                    break;
                }
            }
            if (add) pruned_indices.push_back(candidate);
        }

        neighbors[i].reserve(pruned_indices.size());
        for (size_t j = 0; j < pruned_indices.size(); j++) {
            neighbors[i].push_back(ids[pruned_indices[j]]);
        }
    }
    
    return neighbors;
}

template <typename GraphType, typename PointRangeType>
void beam_clusters_vamana(GraphType &graph, PointRangeType &points, size_t target_cluster_size, size_t max_cluster_size, size_t fanout) {
    using PointType = typename PointRangeType::Point;
    using index_t = uint32_t;
    using value_t = typename PointRangeType::Point::distanceType;

    parlay::internal::timer timer;
    timer.start();

    // Sample initial leaders
    auto indices = parlay::tabulate<index_t>(points.size(), [&] (size_t i) { return i; });
    parlay::random_shuffle(indices);
    size_t num_initial_leaders = points.size() / (target_cluster_size * fanout);
    auto leader_ids = parlay::tabulate(num_initial_leaders, [&] (size_t i) { return indices[i]; });
    std::cout << "Sampled " << num_initial_leaders << " initial leaders" << std::endl;

    // Create an index on the leaders
    static constexpr size_t max_degree = 32;
    static constexpr size_t beam_width = 64;
    static constexpr double alpha = 1.2;
    PointRangeType leader_points(points, leader_ids);
    GraphType leader_graph(max_degree, leader_ids.size());
    parlayANN::stats<index_t> build_stats(leader_ids.size());
    parlayANN::BuildParams build_params(max_degree, beam_width, alpha, 2, false);
    parlayANN::knn_index<PointRangeType, PointRangeType, index_t> index(build_params);
    index.build_index(leader_graph, leader_points, leader_points, build_stats);
    std::cout << "Built index on leaders" << std::endl;

    // Query the index for the closest leaders to each point
    static constexpr size_t neighbor_clusters = 5;
    static constexpr double cut = 1.35;
    parlayANN::stats<index_t> query_stats(points.size());
    parlayANN::QueryParams query_params(neighbor_clusters, beam_width, cut, points.size(), max_degree);
    auto beams = parlayANN::qsearchAll<PointRangeType, PointRangeType, PointRangeType, index_t>(points, points, points, leader_graph, leader_points, leader_points, leader_points, query_stats, leader_ids[0], query_params);
    std::cout << "Ran beam search for each point" << std::endl;
    
    // Assign points to clusters
    std::vector<PaddedMutex> cluster_locks(leader_ids.size());
    std::vector<parlay::sequence<index_t>> clusters(leader_ids.size());
    parlay::parallel_for(0, points.size(), [&] (size_t i) {
        for (size_t j = 0; j < fanout && j < beams[i].size(); j++) {
            index_t leader = beams[i][j];
            std::lock_guard<std::mutex> lock(cluster_locks[leader].m);
            clusters[leader].push_back(i);
        }
    });
    std::cout << "Assigned points to clusters" << std::endl;
    std::cout << "Smallest cluster: " << std::min_element(clusters.begin(), clusters.end(), [] (const auto& a, const auto& b) { return a.size() < b.size(); })->size() << std::endl;
    std::cout << "Largest cluster: " << std::max_element(clusters.begin(), clusters.end(), [] (const auto& a, const auto& b) { return a.size() < b.size(); })->size() << std::endl;
    std::cout << "Num clusters over limit: " << std::count_if(clusters.begin(), clusters.end(), [&] (const auto& c) { return c.size() > max_cluster_size; }) << std::endl;
    std::cout << "Clusters computed in " << timer.next_time() << " seconds" << std::endl;

    // Perform pruning within each cluster
    size_t cluster_max_degree = graph.max_degree() / 8;
    std::vector<PaddedMutex> adjlist_locks(points.size());
    parlay::sequence<parlay::sequence<index_t>> neighbors(points.size());
    parlay::parallel_for(0, leader_ids.size(), [&] (size_t i) {
        auto cluster_neighbors = cluster_prune(points, clusters[i], cluster_max_degree);
        for (size_t j = 0; j < clusters[i].size(); j++) {
            index_t point = clusters[i][j];
            std::lock_guard<std::mutex> lock(adjlist_locks[point].m);
            neighbors[point] = std::move(cluster_neighbors[j]);
        }
    }, 1);
    std::cout << "Pruned " << cluster_max_degree << " candidates from clusters in " << timer.next_time() << " seconds" << std::endl;

    // Add candidates from beam and prune
    parlay::random_generator gen;
    std::uniform_int_distribution<index_t> dis(0, std::numeric_limits<index_t>::max());
    parlay::parallel_for(0, points.size(), [&] (size_t i) {
        auto r = gen[i];
        neighbors[i].reserve(graph.max_degree());
        for (size_t j = fanout; j < graph.max_degree() && j < beams[i].size(); j++) {
            index_t leader_id = beams[i][j];
            index_t candidate = clusters[leader_id][dis(r) % clusters[leader_id].size()];

            value_t dist_to_point = points[i].distance(points[candidate]);
            bool add = true;
            for (size_t k = 0; k < neighbors[i].size(); k++) {
                value_t dist_to_neighbor = points[i].distance(points[neighbors[i][k]]);
                if (dist_to_point < dist_to_neighbor) {
                    add = false;
                    break;
                }
            }
            if (add) neighbors[i].push_back(candidate);
        }
    }, 1);
    std::cout << "Pruned " << graph.max_degree() - cluster_max_degree << " candidates from beam in " << timer.next_time() << " seconds" << std::endl;

    // Update the graph
    auto edges = parlay::flatten(
        parlay::tabulate(points.size(), [&] (index_t i) {
            return parlay::map(neighbors[i], [&] (index_t j) { return std::make_pair(i, j); });
        })
    );
    parlayANN::cluster_utils<PointType, PointRangeType, index_t>::process_edges(graph, std::move(edges));
    // parlay::parallel_for(0, points.size(), [&] (size_t i) {
    //     graph[i].update_neighbors(neighbors[i]);
    // });
}