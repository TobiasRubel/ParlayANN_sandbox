#include <queue>
#include <cstdint>

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
