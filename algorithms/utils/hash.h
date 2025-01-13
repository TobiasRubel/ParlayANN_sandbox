#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include <immintrin.h>

#include <cmath>  // For sqrt
#include <iomanip> // For formatting output

#pragma once


template<typename Point, typename PointRange, typename indexType>
struct Hash_family{
    using distanceType = typename Point::distanceType;

    Hash_family(indexType b, indexType n, indexType d) {
        num_buckets = n;
        bucket_size = b;
        auto num_vectors = num_buckets*bucket_size;
        dimension = d;
        random_vectors = parlay::sequence<parlay::sequence<distanceType>>(num_vectors);
        for (size_t i = 0; i < num_vectors; i++) {
            parlay::random_generator gen(i);
            std::normal_distribution<float> dis(0.0, 1.0);
            auto vec = parlay::tabulate(d, [&](size_t i) {
                auto r = gen[i];
                return dis(r);
            });
            random_vectors[i] = vec;  
        }
    }

    parlay::sequence<size_t> hash(Point p) {
        parlay::sequence<size_t> hash_values = parlay::tabulate(num_buckets, [&](size_t i) {
            size_t hash = 0;
            for (size_t j = 0; j < bucket_size; j++) {
                float dot = 0;
                for (size_t k = 0; k < dimension; k++) {
                    dot += p[k] * random_vectors[(i*bucket_size)+j][k];
                }
                hash = hash << 1;
                if (dot > 0) {
                    hash = hash | 1;
                }
            }
            return hash;
        });
        return hash_values;
    }

    // Precompute hash values for all points in the PointRange
    void precompute_hashes(const PointRange& points) {
        size_t n = points.size();
        precomputed_hashes = parlay::sequence<parlay::sequence<size_t>>(n);

        parlay::parallel_for(0, n, [&](size_t i) {
            precomputed_hashes[i] = hash(points[i]);
        });
    }


    distanceType compute_hash_overlap(parlay::sequence<size_t> hash1, parlay::sequence<size_t> hash2) {
        distanceType overlap = 0;
        for (size_t i = 0; i < num_buckets; i++) {
            overlap += (hash1[i] == hash2[i]);
        }
        return overlap/num_buckets;
    }

    distanceType compute_hash_overlap(Point p1, Point p2) {
        return compute_hash_overlap_avx(hash(p1), hash(p2));
    }

    distanceType compute_hash_overlap(size_t i, size_t j) {
        return compute_hash_overlap_avx(precomputed_hashes[i], precomputed_hashes[j]);
    }

    // Function to compute and print statistics of hash buckets
    void compute_and_print_bucket_statistics() const {
        for (size_t bucket_index = 0; bucket_index < num_buckets; ++bucket_index) {
            // Compute summary statistics for the current bucket
            size_t num_unique_hashes = 0;
            size_t min_frequency = std::numeric_limits<size_t>::max();
            size_t max_frequency = 0;
            size_t total_frequency = 0;
            parlay::sequence<size_t> frequencies(num_buckets, 0);
            for (auto i: precomputed_hashes) {
                size_t hash_value = i[bucket_index];
                frequencies[hash_value]++;
            }
            std::cout << "Bucket " << bucket_index << " Summary Statistics:" << std::endl;
            for (size_t i = 0; i < num_buckets; ++i) {
                if (frequencies[i] > 0) {
                    num_unique_hashes++;
                    min_frequency = std::min(min_frequency, frequencies[i]);
                    max_frequency = std::max(max_frequency, frequencies[i]);
                    total_frequency += frequencies[i];
                }
            }
            double mean_frequency = static_cast<double>(total_frequency) / num_unique_hashes;
            double sum_of_squares = 0;
            for (size_t i = 0; i < num_buckets; ++i) {
                if (frequencies[i] > 0) {
                    sum_of_squares += (frequencies[i] - mean_frequency) * (frequencies[i] - mean_frequency);
                }
            }
            double std_dev = std::sqrt(sum_of_squares / num_unique_hashes);
            // Print formatted summary statistics
            std::cout << std::fixed << std::setprecision(2);
            std::cout << "Bucket " << bucket_index << " Summary Statistics:" << std::endl;
            std::cout << "  Number of unique hash values : " << num_unique_hashes << std::endl;
            std::cout << "  Minimum frequency            : " << min_frequency << std::endl;
            std::cout << "  Maximum frequency            : " << max_frequency << std::endl;
            std::cout << "  Mean frequency               : " << mean_frequency << std::endl;
            std::cout << "  Standard deviation           : " << std_dev << std::endl;
            std::cout << "----------------------------------------" << std::endl;
        }
    }



private:
    indexType num_buckets;
    indexType bucket_size;
    parlay::sequence<parlay::sequence<distanceType>> random_vectors;
    parlay::sequence<parlay::sequence<size_t>> precomputed_hashes;
    indexType dimension;

};