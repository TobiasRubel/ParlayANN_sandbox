#pragma once

#include <queue>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <algorithm>

struct TopN {
    TopN(size_t n) : n(n) { }
    size_t n;
    using value_type = std::pair<float, uint32_t>;
    std::priority_queue<value_type> pq;

    void Add(value_type e) {
        if (pq.size() < n) {
            pq.push(e);
        } else if (pq.top() > e) {
            pq.pop();
            pq.push(e);
        }
    }

    std::vector<value_type> Take() {
        std::vector<value_type> res(pq.size());
        // heap-sort because we can't access the container without hacks via inheritance
        while (!pq.empty()) {
            res[pq.size() - 1] = pq.top();
            pq.pop();
        }
        return res;
    }

    const value_type& Top() const { return pq.top(); }
};

template <class PR>
TopN ClosestLeaders(PR& points, PR& leader_points, uint32_t my_id, int k) {
    TopN top_k(k);
    for (uint32_t j = 0; j < leader_points.n; ++j) {
        auto dist = points[my_id].distance(leader_points[j]);
        top_k.Add(std::make_pair(dist, j));
    }
    return top_k;
}

template <class PR>
std::vector<std::pair<uint32_t, float>> closest_leaders(PR &points, PR &leader_points, uint32_t index, int k) {
    if (leader_points.size() <= k) {
        std::vector<std::pair<uint32_t, float>> res(leader_points.size());
        for (uint32_t i = 0; i < leader_points.size(); i++) {
            res[i] = std::make_pair(i, points[index].distance(leader_points[i]));
        }
        return res;
    }

    std::vector<std::pair<uint32_t, float>> top_k;
    top_k.reserve(k);
    for (uint32_t i = 0; i < k; i++) {
        top_k.push_back(std::make_pair(i, points[index].distance(leader_points[i])));
    }
    std::make_heap(top_k.begin(), top_k.end(), [](auto& a, auto& b) { return a.second < b.second; });

    for (uint32_t i = k; i < leader_points.size(); i++) {
        float dist = points[index].distance(leader_points[i]);
        if (dist < top_k.front().second) {
            std::pop_heap(top_k.begin(), top_k.end(), [](auto& a, auto& b) { return a.second < b.second; });
            top_k.back() = std::make_pair(i, dist);
            std::push_heap(top_k.begin(), top_k.end(), [](auto& a, auto& b) { return a.second < b.second; });
        }
    }

    std::sort_heap(top_k.begin(), top_k.end(), [](auto& a, auto& b) { return a.second < b.second; });
    return top_k;
}
template <class PR>
TopN PrunedLeaders(PR& Points, PR& leader_points, uint32_t p, int k, double alpha) {
    TopN top_k(k);
    //create pairs of leader id, my_ids distance to leader
    std::vector<std::pair<uint32_t, float>> candidates;
    for (uint32_t j = 0; j < leader_points.size(); ++j) {
        auto dist = Points[p].distance(leader_points[j]);
        candidates.push_back(std::make_pair(j, dist));
    }
    //sort by distance
    std::sort(candidates.begin(), candidates.end(), [](auto a, auto b) { return a.second < b.second; });
    size_t candidate_idx = 0;

    std::vector<uint32_t> new_nbhs;
    new_nbhs.reserve(k);
    while (candidate_idx < candidates.size() && top_k.pq.size() < k) {
      // Don't need to do modifications.
      auto p_prime = candidates[candidate_idx].first;
      // With this modification, p_star should not be -1.
      if (p_prime == p || p_prime == -1) {
        candidate_idx++;
        continue;
      }
      // Check if p_prime is already pruned out based on what we have
      // added so far.
      bool add = true;
      for (auto p_star : new_nbhs) {
        auto dist_starprime = Points[p_star].distance(Points[p_prime]);
        auto dist_pprime = candidates[candidate_idx].second;
        if (alpha * dist_starprime <= dist_pprime) {
          add = false;
          break;
        }
      }
      if (add)  new_nbhs.push_back(p_prime);
      candidate_idx++;
    }
    for (auto p_prime : new_nbhs) {
        top_k.Add(std::make_pair(Points[p].distance(leader_points[p_prime]), p_prime));
    }
    // if (top_k.pq.size() < k) {
    //     std::cout << "Warning: not enough leaders to prune" << std::endl;
    // }
    return top_k;
}