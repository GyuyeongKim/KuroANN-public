// A navigable small-world graph over the partition centroids. Built once at load, read-only during search.
#ifdef __AVX512F__
#include <immintrin.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "internal.h"

namespace kuroann {

float squared_l2(const float *a, const float *b, int d) {
#ifdef __AVX512F__
  __m512 acc = _mm512_setzero_ps();
  int i = 0;
  for (; i + 16 <= d; i += 16) {
    __m512 x = _mm512_loadu_ps(a + i), y = _mm512_loadu_ps(b + i), t = _mm512_sub_ps(x, y);
    acc = _mm512_fmadd_ps(t, t, acc);
  }
  float s = _mm512_reduce_add_ps(acc);
  for (; i < d; i++) {
    float t = a[i] - b[i];
    s += t * t;
  }
  return s;
#else
  float s = 0;
  for (int i = 0; i < d; i++) {
    float t = a[i] - b[i];
    s += t * t;
  }
  return s;
#endif
}

float CentroidGraph::dist(const float *q, int i) const { return squared_l2(q, point(i), dim_); }

std::vector<int> &CentroidGraph::links(int node, int level) { return level == 0 ? level0_[node] : upper_[node][level - 1]; }
const std::vector<int> &CentroidGraph::links(int node, int level) const {
  return level == 0 ? level0_[node] : upper_[node][level - 1];
}

int CentroidGraph::greedy(const float *q, int entry, int level) const {
  int cur = entry;
  float cd = dist(q, cur);
  bool improved = true;
  while (improved) {
    improved = false;
    for (int nn : links(cur, level)) {
      float dd = dist(q, nn);
      if (dd < cd) {
        cd = dd;
        cur = nn;
        improved = true;
      }
    }
  }
  return cur;
}

void CentroidGraph::search_level(const float *q, int entry, float entry_dist, int ef, int level,
                                 std::vector<Item> &res) const {
  static thread_local std::vector<uint32_t> visited;
  static thread_local uint32_t stamp = 0;
  if ((int) visited.size() < n_) {
    visited.assign(n_, 0u);
    stamp = 0;
  }
  if (++stamp == 0) {
    std::fill(visited.begin(), visited.end(), 0u);
    stamp = 1;
  }
  auto farther = [](const Item &a, const Item &b) { return a.first < b.first; };
  auto nearer = [](const Item &a, const Item &b) { return a.first > b.first; };
  std::vector<Item> cand;
  cand.reserve(ef * 2);
  res.clear();
  res.push_back({entry_dist, entry});
  std::push_heap(res.begin(), res.end(), farther);
  cand.push_back({entry_dist, entry});
  std::push_heap(cand.begin(), cand.end(), nearer);
  visited[entry] = stamp;
  while (!cand.empty()) {
    auto c = cand.front();
    if (c.first > res.front().first && (int) res.size() >= ef) break;
    std::pop_heap(cand.begin(), cand.end(), nearer);
    cand.pop_back();
    for (int nn : links(c.second, level)) {
      if (visited[nn] == stamp) continue;
      visited[nn] = stamp;
      float dd = dist(q, nn);
      if ((int) res.size() < ef || dd < res.front().first) {
        cand.push_back({dd, nn});
        std::push_heap(cand.begin(), cand.end(), nearer);
        res.push_back({dd, nn});
        std::push_heap(res.begin(), res.end(), farther);
        if ((int) res.size() > ef) {
          std::pop_heap(res.begin(), res.end(), farther);
          res.pop_back();
        }
      }
    }
  }
}

void CentroidGraph::select_links(std::vector<Item> &candidates, int keep, std::vector<int> &out) const {
  std::vector<Item> w(candidates.begin(), candidates.end());
  std::sort(w.begin(), w.end(), [](const Item &a, const Item &b) { return a.first < b.first; });
  out.clear();
  for (auto &c : w) {
    if ((int) out.size() >= keep) break;
    bool ok = true;
    for (int r : out) {
      if (squared_l2(point(c.second), point(r), dim_) < c.first) {
        ok = false;
        break;
      }
    }
    if (ok) out.push_back(c.second);
  }
}

void CentroidGraph::connect(int node, int level, std::vector<int> &neighbors) {
  links(node, level) = neighbors;
  const int limit = level == 0 ? links_level0_ : links_per_level_;
  for (int nn : neighbors) {
    auto &lst = links(nn, level);
    lst.push_back(node);
    if ((int) lst.size() > limit) {
      std::vector<Item> cand;
      cand.reserve(lst.size());
      for (int x : lst) cand.push_back({squared_l2(point(nn), point(x), dim_), x});
      std::make_heap(cand.begin(), cand.end(), [](const Item &a, const Item &b) { return a.first < b.first; });
      std::vector<int> kept;
      select_links(cand, limit, kept);
      lst = kept;
    }
  }
}

void CentroidGraph::build(const float *points, int n, int dim) {
  points_ = points;
  n_ = n;
  dim_ = dim;
  const double ml = 1.0 / std::log((double) std::max(2, links_per_level_));
  level0_.assign(n, {});
  upper_.assign(n, {});
  entry_ = 0;
  top_level_ = 0;
  auto uniform = [this]() {
    rng_ ^= rng_ << 13;
    rng_ ^= rng_ >> 17;
    rng_ ^= rng_ << 5;
    return (rng_ & 0xFFFFFFu) / (float) 0x1000000;
  };
  for (int i = 0; i < n; i++) {
    const int l = (int) (-std::log(std::max(1e-7f, uniform())) * ml);
    upper_[i].assign(l, {});
    if (i == 0) {
      top_level_ = std::max(top_level_, l);
      continue;
    }
    int cur = entry_;
    for (int lev = top_level_; lev > l; lev--) cur = greedy(point(i), cur, lev);
    for (int lev = std::min(l, top_level_); lev >= 0; lev--) {
      std::vector<Item> res;
      search_level(point(i), cur, dist(point(i), cur), build_ef_, lev, res);
      std::vector<int> neighbors;
      select_links(res, lev == 0 ? links_level0_ : links_per_level_, neighbors);
      connect(i, lev, neighbors);
      cur = neighbors.empty() ? cur : neighbors[0];
    }
    if (l > top_level_) {
      top_level_ = l;
      entry_ = i;
    }
  }
  // Pruning a full link list can drop every in-link of a node, and search never returns such a centroid.
  std::vector<char> seen(n, 0);
  std::vector<int> stack{entry_};
  seen[entry_] = 1;
  auto walk = [&]() {
    while (!stack.empty()) {
      int u = stack.back();
      stack.pop_back();
      for (int v : level0_[u]) {
        if (!seen[v]) {
          seen[v] = 1;
          stack.push_back(v);
        }
      }
    }
  };
  walk();
  for (int i = 0; i < n; i++) {
    if (seen[i]) continue;
    int cur = entry_;
    for (int lev = top_level_; lev > 0; lev--) cur = greedy(point(i), cur, lev);
    std::vector<Item> res;
    search_level(point(i), cur, dist(point(i), cur), build_ef_, 0, res);
    int best = -1;
    float bd = 3.4e38f;
    for (auto &r : res) {
      if (seen[r.second] && r.second != i && r.first < bd) {
        bd = r.first;
        best = r.second;
      }
    }
    if (best < 0) {
      for (int j = 0; j < n; j++) {
        if (seen[j]) {
          float d = dist(point(i), j);
          if (d < bd) {
            bd = d;
            best = j;
          }
        }
      }
    }
    level0_[best].push_back(i);
    seen[i] = 1;
    stack.push_back(i);
    walk();
  }
}

void CentroidGraph::nearest(const float *q, int count, std::vector<int> &out) const {
  const int ef = std::min(n_, std::max((int) (count * 2.5) + 32, 128));
  int cur = entry_;
  for (int lev = top_level_; lev > 0; lev--) cur = greedy(q, cur, lev);
  std::vector<Item> res;
  search_level(q, cur, dist(q, cur), std::max(ef, count), 0, res);
  std::sort(res.begin(), res.end(), [](const Item &a, const Item &b) { return a.first < b.first; });
  out.clear();
  const int t = std::min((int) res.size(), count);
  for (int k = 0; k < t; k++) out.push_back(res[k].second);
}

}  // namespace kuroann
