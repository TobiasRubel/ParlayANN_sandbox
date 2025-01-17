// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <math.h>

#include <algorithm>
#include <functional>
#include <queue>
#include <random>
#include <set>

#include "../utils/graph.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"

#include "../vamana/index.h"

namespace parlayANN {

template <typename PR, typename Seq>
auto run_vamana_on_indices(Seq &seq, PR &all_points, BuildParams &BP) {
  using indexType = uint32_t;
  Graph<indexType> G(BP.R, seq.size());

  PR points = PR(all_points, seq);

  using findex = knn_index<PR, PR, indexType>;
  findex I(BP);
  stats<unsigned int> BuildStats(G.size());
  I.build_index(G, points, points, BuildStats, /*sort_neighbors=*/true, /*print=*/false);

  using edge = std::pair<uint32_t, uint32_t>;
  parlay::sequence<edge> edges;
  for (size_t i=0; i < G.size(); ++i) {
    size_t our_ind = seq[i];
    for (size_t j=0; j < G[i].size(); ++j) {
      auto neighbor_ind = seq[G[i][j]];
      edges.push_back(std::make_pair(our_ind, neighbor_ind));
    }
  }

  return edges;
}


#include <atomic>

class SpinLock {
public:
    // boilerplate to make it 'copyable'. but we just clear the spinlock. there is never a use case to copy a locked spinlock
    SpinLock() { }
    SpinLock(const SpinLock&) { }
    SpinLock& operator=(const SpinLock&) { spinner.clear(std::memory_order_relaxed); return *this; }

    bool tryLock() {
        return !spinner.test_and_set(std::memory_order_acquire);
    }

    void lock() {
        while (spinner.test_and_set(std::memory_order_acquire)) {
            // spin
            // stack overflow says adding 'cpu_relax' instruction may improve performance
        }
    }

    void unlock() {
        spinner.clear(std::memory_order_release);
    }

private:
    std::atomic_flag spinner = ATOMIC_FLAG_INIT;
};



} // end namespace
