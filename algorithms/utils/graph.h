// This code is part of the Parlay Project
// Copyright (c) 2024 Guy Blelloch, Magdalen Dobson and the Parlay team
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

#include <algorithm>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/alloc.h"
#include "parlay/internal/file_map.h"

#include "types.h"

namespace parlayANN {
  
template<typename indexType>
struct edgeRange{

  size_t size() const {return edges[0];}

  indexType id() const {return id_;}

  edgeRange() : edges(parlay::make_slice<indexType*, indexType*>(nullptr, nullptr)) {}

  edgeRange(indexType* start, indexType* end, indexType id)
    : edges(parlay::make_slice<indexType*, indexType*>(start,end)), id_(id) {
    maxDeg = edges.size() - 1;
  }

  indexType operator [] (indexType j) const {
    if (j > edges[0]) {
      std::cout << "ERROR: index exceeds degree while accessing neighbors" << std::endl;
      abort();
    } else return edges[j+1];
  }

  void append_neighbor(indexType nbh){
    if (edges[0] == maxDeg) {
      std::cout << "ERROR in append_neighbor for: " << id_ << ". Cannot exceed max degree "
                << maxDeg << std::endl;
      abort();
    } else {
      edges[edges[0]+1] = nbh;
      edges[0] += 1;
    }
  }

  template<typename rangeType>
  void update_neighbors(const rangeType& r){
    if (r.size() > maxDeg) {
      std::cout << "ERROR in update_neighbors: cannot exceed max degree "
                << maxDeg << std::endl;
      abort();
    }
    edges[0] = r.size();
    for (int i = 0; i < r.size(); i++) {
      edges[i+1] = r[i];
    }
  }

  template<typename rangeType>
  void append_neighbors(const rangeType& r){
    if (r.size() + edges[0] > maxDeg) {
      std::cout << "ERROR in append_neighbors for point " << id_
                << ": cannot exceed max degree " << maxDeg << std::endl;
      std::cout << edges[0] << std::endl;
      std::cout << r.size() << std::endl;
      abort();
    }
    for (int i = 0; i < r.size(); i++) {
      edges[edges[0] + i + 1] = r[i];
    }
    edges[0] += r.size();
  }

  void clear_neighbors(){
    edges[0] = 0;
  }

  void prefetch() const {
    int l = ((edges[0] + 1) * sizeof(indexType))/64;
    for (int i = 0; i < l; i++)
      __builtin_prefetch((char*) edges.begin() + i *  64);
  }

  template<typename F>
  void sort(F&& less){
    std::sort(edges.begin() + 1, edges.begin() + 1 + edges[0], less);}

  indexType* begin() const {return edges.begin() + 1;}

  indexType* end() const {return edges.end() + 1 + edges[0];}

private:
  parlay::slice<indexType*, indexType*> edges;
  long maxDeg;
  indexType id_;
};

template<typename indexType_>
struct Graph{
  using indexType = indexType_;
  
  long max_degree() const {return maxDeg;}
  size_t size() const {return n;}

  Graph(){}

  void allocate_graph(long maxDeg, size_t n) {
    long cnt = n * (maxDeg + 1);
    long num_bytes = cnt * sizeof(indexType);
    indexType* ptr;
    if (num_bytes > 1 << 24) {
      constexpr size_t ALIGNMENT = 1L << 21;
      num_bytes = (num_bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
      ptr = (indexType*) aligned_alloc(ALIGNMENT, num_bytes);
      madvise(ptr, num_bytes, MADV_HUGEPAGE);
      graph = std::shared_ptr<indexType[]>(ptr, std::free);
    }
    else {
      ptr = (indexType*) parlay::p_malloc(num_bytes);
      graph = std::shared_ptr<indexType[]>(ptr, parlay::p_free);
    }
    parlay::parallel_for(0, cnt, [&] (long i) {ptr[i] = 0;});
  }

  Graph(long maxDeg, size_t n) : maxDeg(maxDeg), n(n) {
    allocate_graph(maxDeg, n);
  }

  Graph(char* gFile){
    std::ifstream reader(gFile);
    if (!reader.is_open()) {
      std::cout << "graph file " << gFile << " not found" << std::endl;
      abort();
    }

    //read num points and max degree
    indexType num_points;
    indexType max_deg;
    reader.read((char*)(&num_points), sizeof(indexType));
    n = num_points;
    reader.read((char*)(&max_deg), sizeof(indexType));
    maxDeg = max_deg;
    std::cout << "Graph: detected " << num_points
              << " points with max degree " << max_deg << std::endl;

    //read degrees and perform scan to find offsets
    indexType* degrees_start = new indexType[n];
    reader.read((char*) (degrees_start), sizeof(indexType) * n);
    indexType* degrees_end = degrees_start + n;
    parlay::slice<indexType*, indexType*> degrees0 =
      parlay::make_slice(degrees_start, degrees_end);
    auto degrees = parlay::tabulate(degrees0.size(), [&] (size_t i){
      return static_cast<size_t>(degrees0[i]);});
    auto [o, total] = parlay::scan(degrees);
    auto offsets = o;
    std::cout << "Total edges read from file: " << total << std::endl;
    offsets.push_back(total);

    allocate_graph(max_deg, n);

    //write 1000000 vertices at a time
    size_t BLOCK_SIZE = 1000000;
    size_t index = 0;
    size_t total_size_read = 0;
    while(index < n){
      size_t g_floor = index;
      size_t g_ceiling = g_floor + BLOCK_SIZE <= n ? g_floor + BLOCK_SIZE : n;
      size_t total_size_to_read = offsets[g_ceiling] - offsets[g_floor];
      indexType* edges_start = new indexType[total_size_to_read];
      reader.read((char*) (edges_start), sizeof(indexType) * total_size_to_read);
      indexType* edges_end = edges_start + total_size_to_read;
      parlay::slice<indexType*, indexType*> edges =
        parlay::make_slice(edges_start, edges_end);
      indexType* gr = graph.get();
      parlay::parallel_for(g_floor, g_ceiling, [&] (size_t i){
        gr[i * (maxDeg + 1)] = degrees[i];
        for(size_t j = 0; j < degrees[i]; j++){
          gr[i * (maxDeg + 1) + 1 + j] = edges[offsets[i] - total_size_read + j];
        }
      });
      total_size_read += total_size_to_read;
      index = g_ceiling;
      delete[] edges_start;
    }
    delete[] degrees_start;
  }

  void save(char* oFile) {
    std::cout << "Writing graph with " << n
              << " points and max degree " << maxDeg
              << std::endl;
    parlay::sequence<indexType> preamble =
      {static_cast<indexType>(n), static_cast<indexType>(maxDeg)};
    parlay::sequence<indexType> sizes = parlay::tabulate(n, [&] (size_t i){
      return static_cast<indexType>((*this)[i].size());});
    std::ofstream writer;
    writer.open(oFile, std::ios::binary | std::ios::out);
    writer.write((char*) preamble.begin(), 2 * sizeof(indexType));
    writer.write((char*) sizes.begin(), sizes.size() * sizeof(indexType));
    size_t BLOCK_SIZE = 1000000;
    size_t index = 0;
    while(index < n){
      size_t floor = index;
      size_t ceiling = index + BLOCK_SIZE <= n ? index + BLOCK_SIZE : n;
      auto edge_data = parlay::tabulate(ceiling - floor, [&] (size_t i){
        return parlay::tabulate(sizes[i + floor], [&] (size_t j){
          return (*this)[i + floor][j];});
      });
      parlay::sequence<indexType> data = parlay::flatten(edge_data);
      writer.write((char*)data.begin(), data.size() * sizeof(indexType));
      index = ceiling;
    }
    writer.close();
  }

  edgeRange<indexType> operator [] (indexType i) const {
    if (i > n) {
      std::cout << "ERROR: graph index out of range: " << i << std::endl;
      abort();
    }
    return edgeRange<indexType>(graph.get() + i * (maxDeg + 1),
                                graph.get() + (i + 1) * (maxDeg + 1),
                                i);
  }

  ~Graph(){}

private:
  size_t n;
  long maxDeg;
  std::shared_ptr<indexType[]> graph;
};

} // end namespace
