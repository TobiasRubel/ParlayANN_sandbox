#include <iostream>
#include <cstdint>
#include <cstring>
#include <random>
#include <mutex>
#include <getopt.h>

#include <parlay/sequence.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/random.h>

#include "utils/types.h"
#include "utils/euclidian_point.h"
#include "utils/mips_point.h"
#include "utils/point_range.h"

struct arguments {
    std::string base_path;
    size_t sample_size;
    double alpha = 1.2;
};

void parse_arguments(int argc, char *argv[], arguments &args) {
    struct option long_options[] = {
        {"help", no_argument, NULL, 'h'},
        {"base_path", required_argument, NULL, 'b'},
        {"sample_size", required_argument, NULL, 's'},
        {"alpha", required_argument, NULL, 'a'},
        {NULL, 0, NULL, 0}
    };

    args.base_path = "/ssd1/anndata/ANNbench_/data/sift/sift_base.fbin";
    args.sample_size = 1000;
    args.alpha = 1;

    int opt;
    while ((opt = getopt_long(argc, argv, "hb:s:a:", long_options, NULL)) != -1) {
        switch (opt) {
            case 'h':
                std::cout << "Usage: ./prune_neighborhood [options]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  -h, --help                     Show this help message" << std::endl;
                std::cout << "  -b, --base_path <path>         Path to the base dataset" << std::endl;
                std::cout << "  -s, --sample_size <num>        Number of samples to generate" << std::endl;
                std::cout << "  -a, --alpha <value>            Alpha value for the neighborhood pruning" << std::endl;
                exit(EXIT_SUCCESS);
            case 'b':
                args.base_path = std::string(optarg);
                break;
            case 's':
                args.sample_size = std::stoul(optarg);
                break;
            case 'a':
                args.alpha = std::stod(optarg);
                break;
            default:
                std::cerr << "Invalid option." << std::endl;
                exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char *argv[]) {
    arguments args;
    parse_arguments(argc, argv, args);

    using index_t = uint32_t;
    using value_t = float;
    //using PointType = parlayANN::Euclidian_Point<value_t>;
    using PointType = parlayANN::Mips_Point<value_t>;
    using PointRangeType = parlayANN::PointRange<PointType>;

    PointRangeType points;
    points = PointRangeType(args.base_path.data());

    parlay::random_generator gen;
    std::uniform_int_distribution<index_t> int_dis(0, points.size() - 1);
    size_t progress = 0, max_progress = args.sample_size;
    std::mutex progress_lock;
    auto neighbors = parlay::tabulate(args.sample_size, [&](size_t i) {
        auto tgen = gen[i];
        index_t v = int_dis(tgen);

        std::vector<index_t> candidates(points.size());
        std::iota(candidates.begin(), candidates.end(), 0);
        std::vector<value_t> distances;
        for (size_t j = 0; j < points.size(); j++) {
            if (j == v) candidates.push_back(0);
            else distances.push_back(points[v].distance(points[j]));
        }

        std::sort(candidates.begin(), candidates.end(), [&](index_t a, index_t b) {
            return distances[a] < distances[b];
        });

        std::vector<index_t> curr_neighbors;
        for (index_t u : candidates) {
            if (u == v) continue;

            bool add = true;
            for (index_t w : curr_neighbors) {
                value_t vw_dist = distances[w];
                value_t uw_dist = points[u].distance(points[w]);
                if (uw_dist * args.alpha < vw_dist) {
                    add = false;
                    break;
                }
            }

            if (add) curr_neighbors.push_back(candidates[u]);
        }

        std::lock_guard lock(progress_lock);
        std::cout << "\rProgress: " << ++progress << "/" << max_progress << std::flush;
        return curr_neighbors;
    });
    std::cout << std::endl;

    auto degrees = parlay::map(neighbors, [](auto &n) { return n.size(); });
    size_t min_degree = parlay::reduce(degrees, parlay::minm<size_t>());
    size_t max_degree = parlay::reduce(degrees, parlay::maxm<size_t>());
    double avg_degree = parlay::reduce(degrees, parlay::addm<size_t>()) / (double)args.sample_size;
    std::cout << "Min degree: " << min_degree << std::endl;
    std::cout << "Max degree: " << max_degree << std::endl;
    std::cout << "Avg degree: " << avg_degree << std::endl;
}