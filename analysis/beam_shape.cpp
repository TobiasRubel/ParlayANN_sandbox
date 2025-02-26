#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <unordered_map>
#include <atomic>

#include <parlay/sequence.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/random.h>

#include "custom_beam_search.h"

using index_t = uint32_t;
using value_t = float;
using point_t = parlayANN::Euclidian_Point<value_t>;
using point_range_t = parlayANN::PointRange<point_t>;

value_t dot_product(const value_t *a, const value_t *b, size_t len) {
    value_t result = 0.0;
    for (size_t i = 0; i < len; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// std::tuple<value_t, value_t, value_t, bool> get_sighting(point_t &source, point_t &query, point_t &current) {
//     auto distance_from_query = query.distance(current);
//     std::vector<value_t> from_source;
//     from_source.reserve(query.params.dims);
//     for (int k = 0; k < query.params.dims; k++) {
//         from_source.push_back(current[k] - source[k]);
//     }
//     value_t dot = dot_product(from_source.data(), d.data(), query.params.dims);
//     std::vector<value_t> projection;
//     projection.reserve(query.params.dims);
//     for (int k = 0; k < query.params.dims; k++) {
//         projection.push_back(query[k] + dot * d[k] / d_dot_d);
//     }
//     value_t dist_target_to_proj = 0.0;
//     for (int k = 0; k < query.params.dims; k++) {
//         value_t diff = query[k] - projection[k];
//         dist_target_to_proj += diff * diff;
//     }
//     value_t dist_proj_to_seen = 0.0;
//     for (int k = 0; k < query.params.dims; k++) {
//         value_t diff = projection[k] - current[k];
//         dist_proj_to_seen += diff * diff;
//     }
//     return std::make_tuple(distance_from_query, dist_target_to_proj, dist_proj_to_seen, false);
// }

// Load a graph, then run multiple custom_beam_searches on it with random starting points,
// counting the number of times each edge is visited, which we'll call its centrality.
// Then output a file with both the length and centrality of each edge.
int main(int argc, char* argv[]) {
    char g_file[] = "/ssd1/anndata/ANNbench_/data/sift/graphs/sift_vamana_R32_L64_alpha1.2_k10_nthreads64.graph";
    char p_file[] = "/ssd1/anndata/ANNbench_/data/sift/sift_base.fbin";
    char o_file[] = "../../output/beam_shape.bin";
    const size_t num_queries = 100;

    std::cout << "Loading graph from " << g_file << std::endl;
    parlayANN::Graph<index_t> graph(g_file);
    std::cout << "Loading base points from " << p_file << std::endl;
    point_range_t points(p_file);
    if (graph.size() != points.size()) {
        std::cerr << "Graph and point range sizes do not match. Aborting..." << std::endl;
        return 1;
    }
    const size_t max_edges = graph.size() * graph.max_degree();

    auto BP = parlayANN::QueryParams(
        10, // k
        64, // beam size
        1.35, // cut
        graph.size(), // limit
        graph.max_degree() // degree limit
    );

    std::cout << "Generating random queries..." << std::flush;
    parlay::random_generator gen;
    std::uniform_int_distribution<index_t> dis(0, points.size() - 1);
    auto queries = parlay::tabulate<std::pair<index_t, index_t>>(num_queries,
        [&] (size_t i) {
            auto r = gen[i];
            return std::make_pair(dis(r), dis(r));
        }
    );
    std::cout << "Done" << std::endl;

    // Run beam search with random start and query points, tracking centrality
    std::cout << "Running beam search..." << std::flush;
    auto sightings = parlay::sequence<std::vector<std::tuple<value_t, value_t, value_t, bool>>>(num_queries);
    parlay::parallel_for(0, num_queries, [&] (size_t i) {
        index_t start_point = queries[i].first, query_point = queries[i].second;
        parlay::sequence<index_t> starting_points(1, start_point);
        std::unordered_map<index_t, std::tuple<value_t, value_t, value_t, bool>> seen;
        
        std::vector<value_t> d;
        d.reserve(points[query_point].params.dims);
        for (int j = 0; j < points[query_point].params.dims; j++) {
            d.push_back(points[query_point][j] - points[start_point][j]);
        }
        value_t d_dot_d = dot_product(d.data(), d.data(), points[query_point].params.dims);

        custom_beam_search(
            graph, points[query_point], points, points[query_point], points,
            starting_points, BP,
            [&] (index_t i) {
                auto it = seen.find(i);
                if (it != seen.end()) {
                    std::get<3>(it->second) = true;
                }
            },
            [&] (index_t i, index_t j) {
                auto distance_from_query = points[query_point].distance(points[j]);
                std::vector<value_t> from_source;
                from_source.reserve(points[query_point].params.dims);
                for (int k = 0; k < points[query_point].params.dims; k++) {
                    from_source.push_back(points[j][k] - points[start_point][k]);
                }
                value_t dot = dot_product(from_source.data(), d.data(), points[query_point].params.dims);
                std::vector<value_t> projection;
                projection.reserve(points[query_point].params.dims);
                for (int k = 0; k < points[query_point].params.dims; k++) {
                    projection.push_back(points[start_point][k] + dot * d[k] / d_dot_d);
                }
                value_t dist_target_to_proj = 0.0;
                for (int k = 0; k < points[query_point].params.dims; k++) {
                    value_t diff = points[query_point][k] - projection[k];
                    dist_target_to_proj += diff * diff;
                }
                value_t dist_proj_to_seen = 0.0;
                for (int k = 0; k < points[query_point].params.dims; k++) {
                    value_t diff = projection[k] - points[j][k];
                    dist_proj_to_seen += diff * diff;
                }
                seen[i] = std::make_tuple(distance_from_query, dist_target_to_proj, dist_proj_to_seen, false);
            }
        );

        for (auto it = seen.begin(); it != seen.end(); it++) {
            sightings[i].push_back(it->second);
        }
    });
    std::cout << "Done" << std::endl;

    // Compute edge lengths for valid edges
    std::cout << "Processing results..." << std::flush;
    auto all_sightings = parlay::flatten(sightings);
    auto edge_lengths = parlay::map(all_sightings, [] (auto &s) {
        return std::get<0>(s);
    });
    auto projection_lengths = parlay::map(all_sightings, [] (auto &s) {
        return std::get<1>(s);
    });
    auto perpendicular_lengths = parlay::map(all_sightings, [] (auto &s) {
        return std::get<2>(s);
    });
    auto used = parlay::map(all_sightings, [] (auto &s) {
        return std::get<3>(s);
    });
    std::cout << "Done" << std::endl;

    // Output edge lengths and centralities to file
    std::ofstream o_stream(o_file, std::ios::binary);
    uint32_t num_sightings = all_sightings.size();
    o_stream.write(reinterpret_cast<char*>(&num_sightings), sizeof(num_sightings));
    o_stream.write(reinterpret_cast<char*>(edge_lengths.begin()), num_sightings * sizeof(value_t));
    o_stream.write(reinterpret_cast<char*>(projection_lengths.begin()), num_sightings * sizeof(value_t));
    o_stream.write(reinterpret_cast<char*>(perpendicular_lengths.begin()), num_sightings * sizeof(value_t));
    o_stream.write(reinterpret_cast<char*>(used.begin()), num_sightings * sizeof(bool));
    std::cout << "Results written to " << o_file << std::endl;
 
    return 0;
}
