#include <iostream>
#include <cstdint>
#include <cstring>
#include <getopt.h>

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

void print_help() {
    std::cout << "Usage: ./prune_neighborhood [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help                Show this help message" << std::endl;
    std::cout << "  -d, --dataset <num>       Dataset number (1 or 2)" << std::endl;
    std::cout << "  -v, --use_vamana          Use vamana to approximate neighborhoods" << std::endl;
    std::cout << "  -e, --exactness <value>   Exactness value (default: 0.9)" << std::endl;
}

struct arguments {
    int dataset;
    bool use_vamana;
    double exactness;
};

void parse_arguments(int argc, char *argv[], arguments &args) {
    struct option long_options[] = {
        {"help", no_argument, NULL, 'h'},
        {"dataset", required_argument, NULL, 'd'},
        {"use_vamana", no_argument, NULL, 'v'},
        {"exactness", required_argument, NULL, 'e'},
        {NULL, 0, NULL, 0}
    };

    args.dataset = 1;
    args.use_vamana = false;
    args.exactness = 0.9;

    int opt;
    while ((opt = getopt_long(argc, argv, "hd:ve:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'h':
                print_help();
                exit(EXIT_SUCCESS);
            case 'd':
                args.dataset = std::stoi(optarg);
                break;
            case 'v':
                args.use_vamana = true;
                break;
            case 'e':
                args.exactness = std::stod(optarg);
                break;
            default:
                print_help();
                exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char *argv[]) {
    arguments args;
    parse_arguments(argc, argv, args);

    std::string base_path, query_path, gt_path, exact_path;
    int distance_type = 0;
    switch (args.dataset) {
        case 1:
            std::cout << "Dataset: SIFT" << std::endl;
            base_path = "/ssd1/anndata/ANNbench_/data/sift/sift_base.fbin";
            query_path = "/ssd1/anndata/ANNbench_/data/sift/sift_query.fbin";
            gt_path = "/ssd1/anndata/ANNbench_/data/sift/sift_query.gt";
            exact_path = "/ssd1/richard/anndata/sift_1000nn.gt";
            break;
        case 2:
            std::cout << "Dataset: Text2Image" << std::endl;
            base_path = "/ssd1/anndata/ANNbench_/data/text2image1M/base.fbin";
            query_path = "/ssd1/anndata/ANNbench_/data/text2image1M/query10K.fbin";
            gt_path = "/ssd1/anndata/ANNbench_/data/text2image1M/text2image1M.gt";
            exact_path = "/ssd1/richard/anndata/t2i_1000nn.gt";
            distance_type = 1;
            break;
        default:
            std::cerr << "Invalid dataset number." << std::endl;
            return 1;
    }

    using index_t = uint32_t;
    using value_t = float;
    using PointType = parlayANN::Euclidian_Point<value_t>;
    //using PointType = parlayANN::Mips_Point<value_t>;
    using PointRangeType = parlayANN::PointRange<PointType>;
    using GraphType = parlayANN::Graph<index_t>;
    size_t max_degree = 32;

    PointRangeType B, Q;
    parlayANN::groundTruth<index_t> GT, neighborhood;
    parlay::sequence<parlay::sequence<index_t>> pruned_neighbors;
    GraphType G;
    if (args.use_vamana) {
        B = PointRangeType(base_path.data());
        Q = PointRangeType(query_path.data());
        GT = parlayANN::groundTruth<index_t>(gt_path.data());
        G = GraphType(max_degree, B.size());
        auto BP = parlayANN::BuildParams(max_degree, 64, 1.2, 2, false);
        auto I = parlayANN::knn_index<PointRangeType, PointRangeType, index_t>(BP);
        parlayANN::stats<index_t> sbuild(size_t(B.size()));
        I.build_index(G, B, B, sbuild);

        std::cout << "Obtaining approximate neighborhoods..." << std::endl;
        auto QP = parlayANN::QueryParams(500, 1000, 1.35, B.size(), max_degree, 1);
        auto neighbors = parlayANN::qsearchAll<PointRangeType, PointRangeType, PointRangeType, index_t>(B, B, B, G, B, B, B, sbuild, 0, QP);

        std::cout << "Pruning neighborhoods..." << std::endl;
        pruned_neighbors = parlay::tabulate(B.size(), [&](size_t i) {
            return I.robustPrune(i, neighbors[i], G, B, 1.2, false).first;
        });
    }
    else {
        B = PointRangeType(base_path.data());
        Q = PointRangeType(query_path.data());
        GT = parlayANN::groundTruth<index_t>(exact_path.data());
        neighborhood = parlayANN::groundTruth<index_t>(exact_path.data());
        G = GraphType(max_degree, B.size());
        auto BP = parlayANN::BuildParams(max_degree, 64, 1.2, 2, false);
        auto I = parlayANN::knn_index<PointRangeType, PointRangeType, index_t>(BP);
        size_t target_neighborhood_size = 500;

        auto neighbors = parlay::tabulate(B.size(), [&](size_t i) {
            return parlay::tabulate(target_neighborhood_size, [&](size_t j) {
                return neighborhood.coordinates(i, j);
            });
        });

        std::cout << "Pruning neighborhoods..." << std::endl;
        pruned_neighbors = parlay::tabulate(B.size(), [&](size_t i) {
            return I.robustPrune(i, neighbors[i], G, B, 1.2, false).first;
        });
    }

    auto adjlist_sizes = parlay::delayed_tabulate(B.size(), [&](size_t i) {
        return pruned_neighbors[i].size();
    });
    std::cout << "Max degree: " << parlay::reduce(adjlist_sizes, parlay::maxm<size_t>()) << std::endl;
    std::cout << "Avg degree: " << (double)parlay::reduce(adjlist_sizes, parlay::addm<size_t>()) / B.size() << std::endl;
    parlay::parallel_for(0, B.size(), [&](size_t i) {
        G[i].clear_neighbors();
        for (size_t j = 0; j < pruned_neighbors[i].size() && j < max_degree; j++) {
            G[i].append_neighbor(pruned_neighbors[i][j]);
        }
    });

    std::cout << "Testing recall..." << std::endl;
    auto [avg_deg, max_deg] = parlayANN::graph_stats_(G);
    parlayANN::Graph_ G_("PrunedNeighborhood", "", G.size(), avg_deg, max_deg, 0);
    search_and_parse(G_, G, B, Q, GT, NULL, NULL, 10, true);
}