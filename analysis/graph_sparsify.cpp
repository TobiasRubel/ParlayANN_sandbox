/*
 * This utility drops x% of the graphs given a specified dropping
 * rule, e.g., drop x% of the closest edges, longest edges, random
 * edges, etc.
 */

#include <algorithm>
#include <iostream>

#include "bench/parse_command_line.h"
#include "parlay/io.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "utils/beamSearch.h"
#include "utils/euclidian_point.h"
#include "utils/mips_point.h"
#include "utils/point_range.h"
#include "vamana/index.h"

using AdjGraph = parlay::sequence<parlay::sequence<unsigned int>>;

using namespace parlayANN;

template <class PR>
void Meld(commandLine& P, PR& Points) {
  char* target_path = P.getOptionValue("-target_path");
  char* source_path = P.getOptionValue("-source_path");
  using indexType = uint32_t;
  Graph<indexType> Target(target_path);
  Graph<indexType> Source(source_path);
  Graph<indexType> Output(Target.max_degree(), Target.size());

  // For each point, replace the k longest edges in the target with k
  // longest edges in the souce.
  long max_to_replace = P.getOptionLongValue("-k", 1);
  parlay::parallel_for(0, Target.size(), [&] (size_t i) {
    using edge = std::pair<float, uint32_t>;
    std::vector<edge> target_edges;
    auto point = Points[i];
    auto edges = Target[i];
    for (size_t j=0; j < edges.size(); ++j) {
      auto neighbor = edges[j];
      auto dist = point.distance(Points[neighbor]);
      target_edges.push_back(std::make_pair(dist, neighbor));
    }
    std::sort(target_edges.begin(), target_edges.end());

    std::vector<edge> source_edges;
    auto s_edges = Source[i];
    for (size_t j=0; j < s_edges.size(); ++j) {
      auto neighbor = s_edges[j];
      auto dist = point.distance(Points[neighbor]);
      source_edges.push_back(std::make_pair(dist, neighbor));
    }
    std::sort(source_edges.begin(), source_edges.end());

    long to_replace = std::min(static_cast<long>(std::min(target_edges.size(),
          source_edges.size())), max_to_replace);

    for (size_t j=0; j<to_replace; ++j) {
      target_edges[target_edges.size() - j - 1] =
        source_edges[source_edges.size() - j - 1]; 
    }

    for (auto [dist, ngh] : target_edges) {
      Output[i].append_neighbor(ngh);
    }
  });

  char* out_path = P.getOptionValue("-out_path");
  Output.save(out_path);
}

template <class PR>
void DropEdges(commandLine& P, PR& Points) {
  char* gpath = P.getOptionValue("-graph_path");
  using indexType = uint32_t;
  Graph<indexType> G(gpath);
  Graph<indexType> D(G.max_degree(), G.size());

  char* modestr = P.getOptionValue("-mode");
  std::string mode(modestr);
  using edge = std::tuple<float, uint32_t, uint32_t>;
  edge null_edge = {std::numeric_limits<float>::min(), std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max()};
  parlay::sequence<edge> edges(G.max_degree() * G.size(), null_edge);

  parlay::parallel_for(0, G.size(), [&] (size_t i) {
    for (size_t j=0; j < G[i].size(); ++j) {
      auto neighbor = G[i][j];
      auto dist = Points[i].distance(Points[j]);
      // Smaller => higher similarity
      if (mode == "lowest") {
        edges[i*G.max_degree() + j] = {dist, i, neighbor};
      } else {
        edges[i*G.max_degree() + j] = {-1*dist, i, neighbor};
      }
    }
  });
  parlay::sort_inplace(edges);

  auto edge_exists = parlay::delayed_tabulate<size_t>(edges.size(), [&] (size_t i) { return edges[i] != null_edge; });
  size_t total_edges = parlay::reduce(edge_exists);

  parlay::random_generator gen;
  std::uniform_real_distribution<float> dis(0, 1);

  double pct = P.getOptionDoubleValue("-pct", 0.1);

  if (mode == "rand") {
    auto keep_edge = [&] (edge e, size_t i) {
      auto r = gen[i];
      float val = dis(r);
      return !(val < pct);
    };
    size_t saved = 0;
    size_t total = 0;
    for (size_t i=0; i < edges.size(); ++i) {
      if (edges[i] != null_edge && keep_edge(edges[i], i)) {
        auto [d, u, v] = edges[i];
        D[u].append_neighbor(v);
        saved++;
      }
      if (edges[i] != null_edge) {
        total++;
      }
    }
  } else if (mode == "lowest" || mode == "highest") {
     size_t to_drop = total_edges*pct;
     size_t dropped = 0;
     size_t saved = 0;
     for (size_t i=0; i < edges.size(); ++i) {
      if (edges[i] != null_edge && dropped >= to_drop) {
        auto [d, u, v] = edges[i];
        D[u].append_neighbor(v);
        saved++;
      } else if (edges[i] != null_edge) {
        dropped++;
      }
    }
  }

  char* out_path = P.getOptionValue("-out_path");
  D.save(out_path);
}

template <class PR>
void Run(commandLine& P, PR& Points) {
  auto taskstr = std::string(P.getOptionValue("-task"));
  if (taskstr == "drop_edges") {
    DropEdges(P, Points);
  } else if (taskstr == "meld") {
    std::cout << "Running meld" << std::endl;
    Meld(P, Points);
  }
}

int main(int argc, char* argv[]) {
  commandLine P(
      argc, argv,
      "[-base_path <b>] [-query_path <q>] "
      "[-data_type <d>] [-k <k> ] [-dist_func <d>] [-gt_path <outfile>]");
  char* iFile = P.getOptionValue("-base_path");
  char* dfc = P.getOptionValue("-dist_func");
  char* vectype = P.getOptionValue("-data_type");
  std::string df = std::string(dfc);
  std::string tp = std::string(vectype);
  if (tp == "float") {
    if (df == "Euclidian") {
      using Point = Euclidian_Point<float>;
      using PR = PointRange<Point>;
      PR Points(iFile);
      Run(P, Points);
    } else if (df == "mips") {
      using Point = Mips_Point<float>;
      using PR = PointRange<Point>;
      PR Points(iFile);
      Run(P, Points);
    } else {
      std::cout << "Unknown df" << std::endl;
      exit(-1);
    }
  }
  if (tp == "uint8") {
    if (df == "Euclidian") {
      using Point = Euclidian_Point<uint8_t>;
      using PR = PointRange<Point>;
      PR Points(iFile);
      Run(P, Points);
    } else if (df == "mips") {
      using Point = Mips_Point<uint8_t>;
      using PR = PointRange<Point>;
      PR Points(iFile);
      Run(P, Points);
    } else {
      std::cout << "Unknown df" << std::endl;
      exit(-1);
    }
  }
}
