#include <cstdint>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cassert>

#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include "utils/types.h"
#include "utils/euclidian_point.h"
#include "utils/point_range.h"
#include "utils/graph.h"
#include "vamana/neighbors.h"

using index_type = uint32_t;
using value_type = float;
using point_type = parlayANN::Euclidian_Point<value_type>;
using point_range_type = parlayANN::PointRange<point_type>;