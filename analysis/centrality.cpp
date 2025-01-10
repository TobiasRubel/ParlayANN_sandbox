#include <iostream>
#include <vector>
#include <atomic>
#include <fstream>

#include "custom_beam_search.h"

using index_type = uint32_t;
using value_type = float;
using point_type = parlayANN::Euclidian_Point<value_type>;
using point_range_type = parlayANN::PointRange<point_type>;

// Load a graph, then run multiple custom_beam_searches on it with random starting points,
// counting the number of times each edge is visited, which we'll call its centrality.
// Then output a file with both the length and centrality of each edge.
int main(int argc, char* argv[]) {
    char g_file[] = "/ssd1/anndata/ANNbench_/data/sift/graphs/sift_vamana_R32_L64_alpha1.2_k10_nthreads64.graph";
    char p_file[] = "/ssd1/anndata/ANNbench_/data/sift/sift_base.fbin";

    parlayANN::Graph<index_type> graph(g_file);
    point_range_type points(p_file);
    if (graph.size() != points.size()) {
        std::cerr << "Graph and point range sizes do not match" << std::endl;
        return 1;
    }

    auto BP = parlayANN::QueryParams(
        10, // k
        64, // beam size
        0.0, // cut
        graph.size(), // limit
        graph.max_degree() // degree limit
    );

    srand(time(NULL));
    auto centrality = parlay::tabulate<std::atomic<uint32_t>>(graph.size() * graph.max_degree(), [] (size_t i) {return 0;});
    uint32_t start_point = rand() % points.size(), query_point = rand() % points.size();
    parlay::sequence<index_type> starting_points(1, start_point);
    custom_beam_search(
        graph, points[query_point], points, points[query_point], points,
        starting_points, BP,
        [] (index_type i) {},
        [&] (index_type i, index_type j) {
            // Find which of the max_degree slots in the edge array this edge is in
            uint32_t slot = -1;
            for (uint32_t k = 0; k < graph.max_degree(); k++) {
                if (graph[i][k] == j) {
                    slot = k;
                    break;
                }
            }
            if (slot != -1) centrality[i * graph.max_degree() + slot]++;
        }
    );

    // Compute edge lengths
    auto edge_lengths = parlay::tabulate<value_type>(graph.size() * graph.max_degree(),
        [&] (size_t i) -> value_type {
            if (i % graph.max_degree() >= graph[i / graph.max_degree()].size()) return -1;
            else return points[graph[i / graph.max_degree()][i % graph.max_degree()]].distance(points[i / graph.max_degree()]);
        }
    );

    // Output edge lengths and centralities
    char out_file[] = "centrality.txt";
    std::ofstream out(out_file);
    for (size_t i = 0; i < edge_lengths.size(); i++) {
        if (edge_lengths[i] == -1) continue;
        out << edge_lengths[i] << " " << centrality[i] << std::endl;
    }
    out.close();
 
    return 0;
}
