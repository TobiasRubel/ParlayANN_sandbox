#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <random>
#include <queue>
#include <utility>
#include <unordered_set>
#include <unordered_map>
#include <limits>

#include <parlay/parallel.h>
#include <parlay/random.h>

#include "utils/euclidian_point.h"
#include "utils/point_range.h"
#include "utils/graph.h"

using index_type = uint32_t;
using value_type = int8_t;

using Graph = parlayANN::Graph<index_type>;
using Point = parlayANN::Euclidian_Point<value_type>;
using PointRange = parlayANN::PointRange<Point>;

std::unordered_map<index_type, value_type> dijkstra(Graph &graph, PointRange &points, index_type source, std::unordered_set<index_type> &targets) {
    std::priority_queue<std::pair<value_type, index_type>, std::vector<std::pair<value_type, index_type>>, std::greater<>> pq;
    std::vector<value_type> dist(graph.size(), std::numeric_limits<value_type>::max());

    std::unordered_map<index_type, value_type> target_distances;

    dist[source] = 0;
    pq.emplace(0, source);

    while (!pq.empty() && target_distances.size() < targets.size()) {
        auto [current_dist, u] = pq.top();
        pq.pop();

        if (targets.count(u)) {
            target_distances[u] = current_dist;
            if (target_distances.size() == targets.size()) break;
        }

        if (current_dist > dist[u]) continue;

        for (const index_type &v : graph[u]) {
            value_type weight = points[u].distance(points[v]);
            if (dist[v] > dist[u] + weight) {
                dist[v] = dist[u] + weight;
                pq.emplace(dist[v], v);
            }
        }
    }

    return target_distances;
}

int main(int argc, char *argv[]) {
    char graph_file[] = "/ssd1/richard/anndata/sample_graphs/space1M_vamana.graph";
    char base_file[] = "/ssd1/trubel/big-ann-benchmarks/data/MSSPACEV1B/spacev1b_base.i8bin.crop_nb_1000000";

    Graph graph(graph_file);
    PointRange points(base_file);
    if (graph.size() != points.size()) {
        std::cerr << "Graph and point range sizes do not match." << std::endl;
        return 1;
    }

    std::cout << "Computing local shortest path samples" << std::endl;
    const size_t num_samples = 20;
    const size_t neighborhood_size = 100;
    parlay::random_generator gen;
    std::uniform_int_distribution<uint32_t> dis(0, graph.size() - 1);
    std::vector<std::vector<std::pair<float, float>>> lines_vs_paths(num_samples);
    parlay::parallel_for(0, num_samples, [&] (size_t i) {
        auto r = gen[i];
        index_type index = dis(r);

        auto distances = parlay::tabulate(points.size(), [&] (size_t j) {
            return points[index].distance(points[j]);
        });
        auto indices = parlay::tabulate<index_type>(points.size(), [&] (size_t j) { return static_cast<index_type>(j); });
        std::sort(indices.begin(), indices.end(), [&] (index_type a, index_type b) {
            return distances[a] < distances[b];
        });
        std::unordered_set<index_type> target_indices(indices.begin() + 1, indices.begin() + std::min<size_t>(neighborhood_size + 1, indices.size()));

        auto target_distances = dijkstra(graph, points, index, target_indices);
        std::vector<std::pair<float, float>> line_vs_path;
        for (const index_type &target : target_indices) {
            line_vs_path.emplace_back((float)distances[target], (float)target_distances[target]);
        }
        lines_vs_paths[i] = std::move(line_vs_path);
    }, 1);

    std::cout << "Writing output to file" << std::endl;
    auto all_datapoints = parlay::flatten(lines_vs_paths);
    char output_file[] = "local_shortest_paths.bin";
    std::ofstream writer(output_file, std::ios::binary);
    if (!writer.is_open()) {
        std::cerr << "Error opening file for writing: " << output_file << std::endl;
        return 1;
    }
    uint32_t num_points = all_datapoints.size();
    writer.write(reinterpret_cast<char*>(&num_points), sizeof(uint32_t));
    writer.write(reinterpret_cast<char*>(all_datapoints.data()), num_points * sizeof(std::pair<float, float>));
    writer.close();
    std::cout << "Local shortest paths written to " << output_file << std::endl;
}