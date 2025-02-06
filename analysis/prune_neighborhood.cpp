#include <iostream>
#include <cstdint>
#include <cstring>

#include <parlay/sequence.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include "utils/types.h"
#include "utils/graph.h"
#include "utils/beamSearch.h"
#include "utils/euclidian_point.h"
#include "utils/mips_point.h"
#include "utils/point_range.h"
#include "utils/stats.h"
#include "utils/parse_results.h"
#include "vamana/neighbors.h"

#define DATASET 1

int main(int argc, char *argv[]) {
    #if DATASET == 1
        std::string base_path = "/ssd1/anndata/ANNbench_/data/sift/sift_base.fbin";
        std::string query_path = "/ssd1/anndata/ANNbench_/data/sift/sift_query.fbin";
        std::string gt_path = "/ssd1/anndata/ANNbench_/data/sift/sift_query.gt";
    #elif DATASET == 2
        std::string base_path = "/ssd1/anndata/ANNbench_/data/text2image1M/base.fbin";
        std::string query_path = "/ssd1/anndata/ANNbench_/data/text2image1M/query10K.fbin";
        std::string gt_path = "/ssd1/anndata/ANNbench_/data/text2image1M/text2image1M.gt";
    #endif

    // Build an index out of the base points
    using index_t = uint32_t;
    using value_t = float;
    #if DATASET == 2
        using PointType = parlayANN::Mips_Point<value_t>;
    #else
        using PointType = parlayANN::Euclidian_Point<value_t>;
    #endif
    using PointRangeType = parlayANN::PointRange<PointType>;
    using GraphType = parlayANN::Graph<index_t>;
    size_t max_degree = 32;
    auto B = PointRangeType(base_path.data());
    auto Q = PointRangeType(query_path.data());
    auto GT = parlayANN::groundTruth<index_t>(gt_path.data());
    auto G = GraphType(max_degree, B.size());
    auto BP = parlayANN::BuildParams(max_degree, 64, 1.2, 2, false);
    auto I = parlayANN::knn_index<PointRangeType, PointRangeType, index_t>(BP);
    parlayANN::stats<index_t> sbuild(size_t(B.size()));
    I.build_index(G, B, B, sbuild);

    // Use the index to obtain the approximate neighborhood of each point
    std::cout << "Obtaining approximate neighborhoods..." << std::endl;
    auto QP = parlayANN::QueryParams(500, 1000, 1.35, B.size(), max_degree, 1);
    auto neighbors = parlayANN::qsearchAll<PointRangeType, PointRangeType, PointRangeType, index_t>(B, B, B, G, B, B, B, sbuild, 0, QP);

    std::cout << "Pruning neighborhoods..." << std::endl;
    auto new_G = GraphType(max_degree, B.size());
    parlay::parallel_for(0, B.size(), [&](size_t i) {
        auto [pruned, _] = I.robustPrune(i, neighbors[i], G, B, 1.2, false);
        new_G[i].clear_neighbors();
        for (size_t j = 0; j < pruned.size() && j < max_degree; j++) {
            new_G[i].append_neighbor(pruned[j]);
        }
    });

    std::cout << "Testing recall..." << std::endl;
    auto [avg_deg, max_deg] = parlayANN::graph_stats_(new_G);
    parlayANN::Graph_ G_("PrunedNeighborhood", "", G.size(), avg_deg, max_deg, 0);
    search_and_parse(G_, new_G, B, Q, GT, NULL, NULL, 10, true);
}