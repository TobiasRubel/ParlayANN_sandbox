#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
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

// Load a graph, then run multiple custom_beam_searches on it with random starting points,
// counting the number of times each edge is visited, which we'll call its centrality.
// Then output a file with both the length and centrality of each edge.
int main(int argc, char* argv[]) {
    char g_file[] = "/ssd1/anndata/ANNbench_/data/sift/graphs/sift_vamana_R32_L64_alpha1.2_k10_nthreads64.graph";
    char p_file[] = "/ssd1/anndata/ANNbench_/data/sift/sift_base.fbin";
    char o_file[] = "centrality.bin";
    const size_t num_queries = 10000;

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
        0.0, // cut
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
    auto centrality = parlay::tabulate<std::atomic<uint32_t>>(max_edges, [] (size_t i) {return 0;});
    parlay::parallel_for(0, num_queries, [&] (size_t i) {
        index_t start_point = queries[i].first, query_point = queries[i].second;
        parlay::sequence<index_t> starting_points(1, start_point);
        custom_beam_search(
            graph, points[query_point], points, points[query_point], points,
            starting_points, BP,
            [] (index_t i) {},
            [&] (index_t i, index_t j) {
                // Find the index of this edge in the adjacency list
                size_t slot = -1;
                for (size_t k = 0; k < graph.max_degree(); k++) {
                    if (graph[i][k] == j) {
                        slot = k;
                        break;
                    }
                }
                if (slot == -1) return;
                centrality[i * graph.max_degree() + slot]++;
            }
        );
    });
    std::cout << "Done" << std::endl;

    // Compute edge lengths for valid edges
    std::cout << "Processing results..." << std::flush;
    auto edge_lengths = parlay::tabulate<value_t>(max_edges,
        [&] (size_t i) -> value_t {
            if (i % graph.max_degree() >= graph[i / graph.max_degree()].size()) return -1;
            else return points[graph[i / graph.max_degree()][i % graph.max_degree()]].distance(points[i / graph.max_degree()]);
        }
    );

    // Filter out nonexistent edges
    auto zipped = parlay::delayed_tabulate<std::pair<value_t, uint32_t>>(max_edges,
        [&] (size_t i) -> std::pair<value_t, uint32_t> {
            return std::make_pair(edge_lengths[i], centrality[i].load());
        }
    );
    auto filtered = parlay::filter(zipped,
        [&] (std::pair<value_t, uint32_t> a) {
            return a.first != -1;
        }
    );
    std::cout << "Done" << std::endl;

    // Output edge lengths and centralities to file
    std::ofstream o_stream(o_file, std::ios::binary);
    uint32_t num_edges = filtered.size();
    o_stream.write(reinterpret_cast<char*>(&num_edges), sizeof(num_edges));
    auto chars = parlay::to_chars(filtered);
    parlay::chars_to_stream(chars, o_stream);
    std::cout << "Results written to " << o_file << std::endl;
 
    return 0;
}
