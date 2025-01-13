#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include <immintrin.h>

#include <cmath>  // For sqrt
#include <iomanip> // For formatting output

#pragma once


template<typename Point, typename PointRange, typename indexType>
struct simhash{
    using distanceType = typename Point::distanceType;
    indexType num_bits;
    parlay::sequence<float> random_vectors;
    indexType dimension;

    simhash(indexType b, indexType d) {
        if (b > 64) {
            std::cout << "Number of bits must be less than 64" << std::endl;
            exit(1);
        }
        num_bits = b;
        dimension = d;
        parlay::random_generator gen;
        std::normal_distribution<float> dis(0.0, 1.0);
        random_vectors = parlay::tabulate(d*b, [&](size_t i) {
            auto r = gen[i];
            return dis(r);
        });
    }

    indexType hash(Point p) {
        indexType h = 0;
        for (indexType i = 0; i < num_bits; i++) {
            distanceType dot = 0;
            for (indexType j = 0; j < dimension; j++) {
                dot += random_vectors[i*dimension + j] * p[j];
            }
            h = h << 1;
            if (dot > 0) {
                h = h | 1;
            }
        }
        return h;
    }

};