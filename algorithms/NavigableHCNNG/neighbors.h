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

#include <algorithm>
#include <cmath>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "../utils/NSGDist.h"  
#include "../utils/types.h"
#include "../utils/beamSearch.h"
#include "../utils/stats.h"
#include "../utils/parse_results.h"
#include "../utils/check_nn_recall.h"
#include "../utils/graph.h"
#include "hcnng_index.h"
#include "../bench/parse_command_line.h"

namespace parlayANN {

static std::string kAlgType = "NavHCNNG";

template<typename Point, typename PointRange, typename indexType>
void ANN(Graph<indexType> &G, long k, BuildParams &BP,
         PointRange &Query_Points,
         groundTruth<indexType> GT, char *res_file, char* exp_prefix,
         bool graph_built, PointRange &Points, commandLine& P) {

  parlay::internal::timer t("ANN"); 
  using findex = hcnng_index<Point, PointRange, indexType>;

  bool multi_pivot = P.getOption("-multi_pivot");
  std::string leaf_method = P.getOptionValue("-leaf_method", "DistMatQuadPrune");
  bool vamana_long_range = P.getOption("-vamana_long_range");
  double top_level_pct = P.getOptionDoubleValue("-top_level_pct", 0.005);
  bool prune = P.getOption("-prune");
  bool prune_all = P.getOption("-prune_all");
	long top_level_leaders = P.getOptionLongValue("-top_level_leaders", 950);
  double alpha = P.getOptionDoubleValue("-alpha", 1.1);
  long prune_degree = P.getOptionLongValue("-prune_degree", std::numeric_limits<long>::max());
  //int fanout = P.getOptionIntValue("-fanout", 1);
  int fanout_per_level = P.getOptionIntValue("-fanout_per_level", 0);
  double fraction_leaders = P.getOptionDoubleValue("-fraction_leaders", 0.0005);
  std::string fanout_scheme = P.getOptionValue("-fanout_scheme", "");
  double idx_time;
  if(!graph_built){
    findex I;
    I.build_index(G, Points, BP.num_clusters, BP.cluster_size, BP.MST_deg, multi_pivot, prune, prune_all, alpha, leaf_method, prune_degree, vamana_long_range, top_level_pct, top_level_leaders, BP.fanout, fanout_per_level,fraction_leaders, fanout_scheme);
    idx_time = t.next_time();
  } else{idx_time=0;}
  std::string name = "NavHCNNG";
  std::string params = "Trees = " + std::to_string(BP.num_clusters);
  auto [avg_deg, max_deg] = graph_stats_(G);
  Graph_ G_(name, params, G.size(), avg_deg, max_deg, idx_time);
  G_.print();

  if(Query_Points.size() != 0)
    search_and_parse(G_, G, Points, Query_Points, GT, res_file, exp_prefix, k, BP.verbose);
}

}; // end namespace
