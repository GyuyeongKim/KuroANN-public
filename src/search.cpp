#ifdef __AVX512F__
#include <immintrin.h>
#endif
#include <liburing.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include "internal.h"
#include "predicate.h"

namespace kuroann {

namespace {

constexpr int kPrefetchAhead = 8;
constexpr uint32_t kMaxStep = 256;
constexpr uint32_t kMaxRing = 4096;
constexpr int kBlock = 4096;

struct Candidate {
  float d;
  int32_t id;
};

inline bool worse_first(const Candidate &a, const Candidate &b) { return a.d < b.d || (a.d == b.d && a.id < b.id); }

inline float estimate(const uint8_t *code, const float *lut, int m, int ksub) {
#ifdef __AVX512F__
  if (ksub == 256 && (m & 15) == 0) {
    __m512 acc = _mm512_setzero_ps();
    const __m512i lanes = _mm512_setr_epi32(0, 256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840);
    for (int p = 0; p < m; p += 16) {
      __m512i c = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *) (code + p)));
      __m512i idx = _mm512_add_epi32(_mm512_add_epi32(_mm512_set1_epi32(p * 256), lanes), c);
      acc = _mm512_add_ps(acc, _mm512_i32gather_ps(idx, lut, 4));
    }
    return _mm512_reduce_add_ps(acc);
  }
#endif
  float a = 0;
  for (int p = 0; p < m; p++) a += lut[p * ksub + code[p]];
  return a;
}

void wilson(double hits, double n, double z, double &lo, double &hi) {
  const double ph = hits / n, z2 = z * z, den = 1 + z2 / n, c = (ph + z2 / (2 * n)) / den;
  const double h = z * std::sqrt(ph * (1 - ph) / n + z2 / (4 * n * n)) / den;
  lo = std::max(0.0, c - h);
  hi = std::min(1.0, c + h);
}

bool delta_member(const Tree &t, int v, const int32_t *labels, const int32_t *labels_end) {
  const Node &x = t.nodes[v];
  if (x.type == kAnd) {
    for (int k : x.kids) {
      if (!delta_member(t, k, labels, labels_end)) return false;
    }
    return true;
  }
  if (x.type == kOr) {
    for (int k : x.kids) {
      if (delta_member(t, k, labels, labels_end)) return true;
    }
    return false;
  }
  const bool in = x.type == kAll || std::binary_search(labels, labels_end, x.label);
  return in != x.negated;
}

double delta_driver(const Updates &u, const Tree &t, int v) {
  const Node &x = t.nodes[v];
  double d = (double) u.delta_size;
  if (x.type == kAnd) {
    for (int k : x.kids) d = std::min(d, delta_driver(u, t, k));
  } else if (x.type == kOr) {
    double s = 0;
    for (int k : x.kids) s += delta_driver(u, t, k);
    d = std::min(d, s);
  } else if (x.type == kLabel && !x.negated) {
    d = (double) u.label_delta[x.label].size();
  }
  return d;
}

// Delta entries that can match a node, as the driver lists do for the packed arrays. May hold repeats.
void delta_candidates(const Updates &u, const Tree &t, int v, std::vector<DeltaRef> &out) {
  const Node &x = t.nodes[v];
  if (x.type == kAnd) {
    int b = x.kids[0];
    for (int k : x.kids) {
      if (delta_driver(u, t, k) < delta_driver(u, t, b)) b = k;
    }
    delta_candidates(u, t, b, out);
  } else if (x.type == kOr && delta_driver(u, t, v) < (double) u.delta_size) {
    for (int k : x.kids) delta_candidates(u, t, k, out);
  } else if (x.type == kLabel && !x.negated) {
    out.insert(out.end(), u.label_delta[x.label].begin(), u.label_delta[x.label].end());
  } else {
    for (int p = 0; p < (int) u.delta.size(); p++) {
      for (int s = 0; s < (int) u.delta[p].ids.size(); s++) out.push_back({p, s});
    }
  }
}

}  // namespace

struct Predicate::Impl {
  Tree positive, negative;
};

Predicate::Predicate(const Index &index, const std::string &expression) : impl_(new Impl) {
  impl_->positive = parse_expression(*index.impl_, expression, false);
  impl_->negative = parse_expression(*index.impl_, expression, true);
}
Predicate::~Predicate() = default;
Predicate::Predicate(Predicate &&) noexcept = default;
Predicate &Predicate::operator=(Predicate &&) noexcept = default;

struct Searcher::Impl {
  const Index::Impl &ix;
  uint32_t k, max_pool;
  std::vector<float> lut;
  struct io_uring ring;
  uint8_t *buffers = nullptr;
  uint32_t batch = 0;
  EvalContext pos, neg;
  std::vector<int> cells, local_cells;
  std::vector<int32_t> local_labels;
  std::vector<std::pair<const uint32_t *, const uint32_t *>> local_runs;
  std::vector<DeltaRef> delta_refs;
  const uint8_t *removed = nullptr;
  std::vector<Candidate> heap;
  std::vector<float> floor_after;
  std::vector<uint8_t> done;
  std::vector<float> top_d;
  std::vector<int32_t> top_id;
  uint32_t top_n = 0;
  uint64_t io_errors = 0;
  std::map<std::pair<std::string, float>, StopTable> fixed_tables;

  Impl(const Index::Impl &index, uint32_t k_, uint32_t pool) : ix(index), k(k_), max_pool(pool) {
    lut.resize((size_t) ix.m * ix.ksub);
    struct io_uring_params params;
    std::memset(&params, 0, sizeof params);
    const uint32_t entries = std::min<uint32_t>(std::max<uint32_t>(64, max_pool + 8), kMaxRing);
    const int rc = io_uring_queue_init_params(entries, &ring, &params);
    if (rc < 0) throw std::runtime_error("cannot create an io_uring ring");
    batch = entries - 8;
    if (posix_memalign((void **) &buffers, kBlock, (size_t) batch * kBlock) != 0) {
      io_uring_queue_exit(&ring);
      throw std::runtime_error("cannot allocate read buffers");
    }
    heap.reserve(max_pool + 1);
    top_d.resize(k);
    top_id.resize(k);
  }
  ~Impl() {
    free(buffers);
    io_uring_queue_exit(&ring);
  }

  const PlanRow *row_for(double v) const {
    for (const PlanRow &r : ix.plan) {
      if (v <= r.vmax) return &r;
    }
    return &ix.plan.back();
  }

  double plan_count(const Tree &negative, bool &ready) {
    const int r = pos.tree->root;
    const double lower = pos.lower[r], upper = pos.upper[r];
    if (row_for(lower) == row_for(upper)) return upper;
    // Counting walks the packed arrays. Updates add the matches in the delta segment, and removals scale both, as random removals do.
    const Updates &u = ix.updates;
    const double delta = u.pending ? (double) count_delta_matches(*pos.tree) : 0;
    const double live = u.pending ? (double) u.live_total / ((double) ix.n + (double) u.delta_size) : 1;
    const auto current = [&](double packed) { return std::min(std::max((packed + delta) * live, lower), upper); };
    const CountOptions &opt = ix.count;
    if (opt.samples > 0 && pos.driver[r] >= (double) opt.min_work) {
      seek(pos, 0u);
      const int s = (int) opt.samples;
      long hits = 0;
      for (int j = 0; j < s; j++) hits += test(pos, r, (uint32_t) (((double) j + 0.5) * (double) ix.n / s));
      double a, b;
      wilson((double) hits, (double) s, opt.z, a, b);
      if (row_for(current(a * ix.n)) == row_for(current(b * ix.n))) return current((double) hits / s * ix.n);
    }
    prepare(neg, ix, negative);
    return current(exact_count(pos, neg, ready));
  }

  // Distinct matches of a label, or of an OR of labels, and all vectors, in the partitions nearest to the query.
  // False for any other filter.
  bool local_share(const float *q, const Tree &tree, double &matches, double &total) {
    const Node &top = tree.nodes[tree.root];
    local_labels.clear();
    if (top.type == kLabel && !top.negated) {
      local_labels.push_back(top.label);
    } else if (top.type == kOr) {
      for (int k : top.kids) {
        const Node &c = tree.nodes[k];
        if (c.type != kLabel || c.negated) return false;
        local_labels.push_back(c.label);
      }
    } else {
      return false;
    }
    ix.graph.nearest(q, (int) std::min<uint32_t>(ix.count.local_partitions, (uint32_t) ix.nlist), local_cells);
    matches = 0;
    total = 0;
    for (int l : local_cells) {
      const uint32_t b = (uint32_t) ix.lptr[l], e = (uint32_t) ix.lptr[l + 1];
      total += (double) (e - b);
      local_runs.clear();
      for (int32_t label : local_labels) {
        const uint32_t *s = ix.sidx.data() + ix.siptr[label];
        const uint32_t *end = ix.sidx.data() + ix.siptr[label + 1];
        const uint32_t *lo = std::lower_bound(s, end, b);
        const uint32_t *hi = std::lower_bound(lo, end, e);
        if (lo < hi) local_runs.push_back({lo, hi});
      }
      if (local_runs.size() == 1) {
        matches += (double) (local_runs[0].second - local_runs[0].first);
        continue;
      }
      while (!local_runs.empty()) {
        uint32_t m = UINT32_MAX;
        for (const auto &r : local_runs) m = std::min(m, *r.first);
        matches += 1;
        for (size_t i = 0; i < local_runs.size();) {
          if (*local_runs[i].first == m && ++local_runs[i].first == local_runs[i].second) {
            local_runs[i] = local_runs.back();
            local_runs.pop_back();
          } else {
            i++;
          }
        }
      }
    }
    return true;
  }

  void make_lut(const float *q) {
    const int m = ix.m, ksub = ix.ksub, dsub = ix.dsub;
    for (int p = 0; p < m; p++) {
      const float *qs = q + p * dsub;
      const float *bk = ix.book.data() + (size_t) p * ksub * dsub;
      float *lm = lut.data() + p * ksub;
      for (int c = 0; c < ksub; c++) {
        const float *code = bk + (size_t) c * dsub;
        float s = 0;
        for (int d = 0; d < dsub; d++) {
          float t = qs[d] - code[d];
          s += t * t;
        }
        lm[c] = s;
      }
    }
  }

  inline void rank(const uint8_t *code, int32_t id, uint32_t pool) {
    if (removed && ((removed[(size_t) id >> 3] >> (id & 7)) & 1)) return;
    const float e = estimate(code, lut.data(), ix.m, ix.ksub);
    if ((uint32_t) heap.size() < pool) {
      heap.push_back({e, id});
      std::push_heap(heap.begin(), heap.end(), worse_first);
    } else if (e < heap.front().d || (e == heap.front().d && id < heap.front().id)) {
      std::pop_heap(heap.begin(), heap.end(), worse_first);
      heap.back() = {e, id};
      std::push_heap(heap.begin(), heap.end(), worse_first);
    }
  }

  inline void consider(uint32_t position, uint32_t pool) {
    rank(ix.codes.data() + (size_t) position * ix.m, ix.lids[position], pool);
  }

  void scan(const uint32_t *p, size_t n, uint32_t pool) {
    for (size_t i = 0; i < n; i++) {
      if (i + kPrefetchAhead < n) {
        __builtin_prefetch(ix.codes.data() + (size_t) p[i + kPrefetchAhead] * ix.m, 0, 3);
        __builtin_prefetch(&ix.lids[p[i + kPrefetchAhead]], 0, 3);
      }
      consider(p[i], pool);
    }
  }

  uint64_t rank_delta_entry(const DeltaPartition &dp, int slot, const Tree &tree, uint32_t pool) {
    if (!delta_member(tree, tree.root, dp.labels.data() + dp.label_start[slot], dp.labels.data() + dp.label_start[slot + 1])) return 0;
    rank(dp.codes.data() + (size_t) slot * ix.m, dp.ids[slot], pool);
    return 1;
  }

  uint64_t scan_delta_partition(int partition, const Tree &tree, uint32_t pool) {
    const DeltaPartition &dp = ix.updates.delta[partition];
    uint64_t matched = 0;
    for (int s = 0; s < (int) dp.ids.size(); s++) matched += rank_delta_entry(dp, s, tree, pool);
    return matched;
  }

  void collect_delta_candidates(const Tree &tree) {
    delta_refs.clear();
    delta_candidates(ix.updates, tree, tree.root, delta_refs);
    std::sort(delta_refs.begin(), delta_refs.end());
    delta_refs.erase(std::unique(delta_refs.begin(), delta_refs.end()), delta_refs.end());
  }

  uint64_t scan_delta_matches(const Tree &tree, uint32_t pool) {
    collect_delta_candidates(tree);
    uint64_t matched = 0;
    for (const DeltaRef &r : delta_refs) matched += rank_delta_entry(ix.updates.delta[r.partition], r.slot, tree, pool);
    return matched;
  }

  uint64_t count_delta_matches(const Tree &tree) {
    collect_delta_candidates(tree);
    uint64_t matched = 0;
    for (const DeltaRef &r : delta_refs) {
      const DeltaPartition &dp = ix.updates.delta[r.partition];
      matched += delta_member(tree, tree.root, dp.labels.data() + dp.label_start[r.slot], dp.labels.data() + dp.label_start[r.slot + 1]);
    }
    return matched;
  }

  void refine(const float *q, int32_t id, const uint8_t *block) {
    const uint8_t *p = block + (size_t) (id % ix.per_block) * ix.vector_bytes;
    const int dim = ix.dim;
    float d2 = 0;
    if (ix.element_bytes == 1) {
      for (int d = 0; d < dim; d++) {
        float t = q[d] - (float) p[d];
        d2 += t * t;
      }
    } else {
      const float *x = (const float *) p;
      for (int d = 0; d < dim; d++) {
        float t = q[d] - x[d];
        d2 += t * t;
      }
    }
    if (top_n < k) {
      top_d[top_n] = d2;
      top_id[top_n] = id;
      top_n++;
      return;
    }
    uint32_t w = 0;
    for (uint32_t t = 1; t < k; t++) {
      if (top_d[t] > top_d[w] || (top_d[t] == top_d[w] && top_id[t] > top_id[w])) w = t;
    }
    if (d2 < top_d[w] || (d2 == top_d[w] && id < top_id[w])) {
      top_d[w] = d2;
      top_id[w] = id;
    }
  }

  void queue_read(int j, int slot) {
    const long offset = (long) (heap[j].id / ix.per_block) * kBlock;
    struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
    io_uring_prep_read(sqe, ix.vectors_fd, buffers + (size_t) slot * kBlock, kBlock, offset);
    sqe->user_data = ((unsigned long long) j << 20) | (unsigned) slot;
  }

  bool read_ok(int j, int res) const {
    const long offset = (long) (heap[j].id / ix.per_block) * kBlock;
    return res == (int) std::min<long>(kBlock, ix.vectors_size - offset);
  }

  // Submits the queued reads and returns once c completions are ready. A signal can end the wait early, so it waits again.
  void submit_and_wait(unsigned c) {
    int r;
    do {
      r = io_uring_submit_and_wait(&ring, c);
    } while (r == -EINTR || (r >= 0 && io_uring_cq_ready(&ring) < c));
    if (r < 0) throw std::runtime_error(std::string("io_uring: ") + std::strerror(-r));
  }

  uint32_t read_all(const float *q) {
    const int nc = (int) heap.size();
    std::sort(heap.begin(), heap.end(), worse_first);
    done.resize(batch);
    for (int first = 0; first < nc; first += (int) batch) {
      const int c = std::min<int>((int) batch, nc - first);
      for (int s = 0; s < c; s++) queue_read(first + s, s);
      submit_and_wait((unsigned) c);
      std::fill(done.begin(), done.begin() + c, (uint8_t) 0);
      struct io_uring_cqe *cqe;
      unsigned head, seen = 0;
      io_uring_for_each_cqe(&ring, head, cqe) {
        const int j = (int) (cqe->user_data >> 20), slot = (int) (cqe->user_data & 0xfffff);
        if (read_ok(j, cqe->res)) done[slot] = 1;
        else io_errors++;
        seen++;
      }
      io_uring_cq_advance(&ring, seen);
      if ((int) seen != c) throw std::runtime_error("lost a read completion");
      for (int s = 0; s < c; s++) {
        if (done[s]) refine(q, heap[first + s].id, buffers + (size_t) s * kBlock);
      }
    }
    return (uint32_t) nc;
  }

  uint32_t read_until_safe(const float *q, const StopTable &table, uint32_t step) {
    std::sort(heap.begin(), heap.end(), worse_first);
    const int nc = (int) heap.size();
    floor_after.resize(nc + 1);
    floor_after[nc] = 3.4e38f;
    for (int j = nc - 1; j >= 0; j--) {
      floor_after[j] = std::min(floor_after[j + 1], heap[j].d + table.errors[table.bucket(j)]);
    }
    const uint32_t chunk = std::min(std::min(step, kMaxStep), batch);
    int issued = 0;
    while (issued < nc) {
      if (top_n == k) {
        float worst = top_d[0];
        for (uint32_t t = 1; t < k; t++) worst = std::max(worst, top_d[t]);
        if (worst <= floor_after[issued]) break;
      }
      const int c = std::min<int>((int) chunk, nc - issued);
      for (int s = 0; s < c; s++) queue_read(issued + s, s);
      submit_and_wait((unsigned) c);
      struct io_uring_cqe *cqe;
      unsigned head, seen = 0;
      io_uring_for_each_cqe(&ring, head, cqe) {
        const int j = (int) (cqe->user_data >> 20), slot = (int) (cqe->user_data & 0xfffff);
        if (read_ok(j, cqe->res)) refine(q, heap[j].id, buffers + (size_t) slot * kBlock);
        else io_errors++;
        seen++;
      }
      io_uring_cq_advance(&ring, seen);
      if ((int) seen != c) throw std::runtime_error("lost a read completion");
      issued += c;
    }
    return (uint32_t) issued;
  }

  uint32_t run(const float *q, const Predicate::Impl &pred, const Setting *fixed, Hit *out, SearchStats *stats) {
    prepare(pos, ix, pred.positive);
    const int root = pred.positive.root;
    const Node &top = pred.positive.nodes[root];
    double count = pos.upper[root];
    bool ready = false;

    bool exhaustive;
    uint32_t partitions, pool, step;
    float alpha;
    const StopTable *table = nullptr;
    int row_index = -1;
    bool local = false;
    if (!fixed) {
      if (ix.plan.empty()) throw std::logic_error("no plan is loaded");
      count = plan_count(pred.negative, ready);
      double m, v;
      if (ix.count.local_partitions > 0 && !ix.updates.pending && local_share(q, pred.positive, m, v) && v > 0) {
        double a, b;
        wilson(m, v, ix.count.z, a, b);
        if (a * (double) ix.n > ix.count.local_ratio * count) {
          // As for a sampled count: the estimate only when both ends of the interval pick the same row.
          count = row_for(a * (double) ix.n) == row_for(b * (double) ix.n) ? m / v * (double) ix.n : a * (double) ix.n;
          local = true;
        }
      }
      const PlanRow *row = row_for(count);
      row_index = (int) (row - ix.plan.data());
      exhaustive = row->partitions == 0;
      partitions = row->partitions;
      pool = row->pool;
      alpha = row->alpha;
      step = 32;
      if (alpha > 0) table = &ix.tables[row->table];
    } else {
      partitions = fixed->partitions;
      pool = fixed->pool;
      alpha = fixed->alpha;
      step = fixed->read_step;
      exhaustive = fixed->route == Route::kExhaustive ||
                   (fixed->route == Route::kAuto && count < (double) partitions * ((double) ix.n / ix.nlist));
      if (alpha > 0) {
        auto key = std::make_pair(fixed->table, alpha);
        auto it = fixed_tables.find(key);
        if (it == fixed_tables.end()) it = fixed_tables.emplace(key, load_stop_table(fixed->table, alpha)).first;
        table = &it->second;
      }
    }
    if (pool > max_pool) throw std::invalid_argument("the pool exceeds the searcher's limit");

    make_lut(q);
    heap.clear();
    top_n = 0;
    removed = ix.updates.any_removed ? ix.updates.removed.data() : nullptr;
    const bool has_delta = ix.updates.delta_size > 0;
    uint64_t scanned = 0;
    uint32_t probed = 0;
    const bool one_label = top.type == kLabel && !top.negated;
    if (exhaustive) {
      if (one_label) {
        const uint32_t *s = ix.sidx.data() + ix.siptr[top.label];
        const size_t n = (size_t) (ix.siptr[top.label + 1] - ix.siptr[top.label]);
        scan(s, n, pool);
        scanned = n;
      } else {
        if (!ready) evaluate(pos, 0u, (uint32_t) ix.n, pos.all);
        scan(pos.all.data(), pos.all.size(), pool);
        scanned = pos.all.size();
      }
      if (has_delta) scanned += scan_delta_matches(pred.positive, pool);
    } else {
      ix.graph.nearest(q, std::min<int>((int) partitions, ix.nlist), cells);
      probed = (uint32_t) cells.size();
      for (int l : cells) {
        if (one_label) {
          const uint32_t *s = ix.sidx.data() + ix.siptr[top.label];
          const uint32_t *e = ix.sidx.data() + ix.siptr[top.label + 1];
          const uint32_t *lo = std::lower_bound(s, e, (uint32_t) ix.lptr[l]);
          const uint32_t *hi = std::lower_bound(lo, e, (uint32_t) ix.lptr[l + 1]);
          scan(lo, (size_t) (hi - lo), pool);
          scanned += (uint64_t) (hi - lo);
        } else {
          evaluate(pos, (uint32_t) ix.lptr[l], (uint32_t) ix.lptr[l + 1], pos.out);
          scan(pos.out.data(), pos.out.size(), pool);
          scanned += pos.out.size();
        }
        if (has_delta) scanned += scan_delta_partition(l, pred.positive, pool);
      }
    }

    uint32_t reads = 0;
    if (!heap.empty()) reads = alpha > 0 ? read_until_safe(q, *table, step) : read_all(q);

    std::vector<Hit> hits(top_n);
    for (uint32_t t = 0; t < top_n; t++) hits[t] = {(uint32_t) top_id[t], top_d[t]};
    std::sort(hits.begin(), hits.end(), [](const Hit &a, const Hit &b) {
      return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
    });
    std::copy(hits.begin(), hits.end(), out);
    if (stats) {
      stats->count = count;
      stats->plan_row = row_index;
      stats->local = local;
      stats->exhaustive = exhaustive;
      stats->partitions = probed;
      stats->reads = reads;
      stats->scanned = (uint32_t) scanned;
    }
    return top_n;
  }
};

Searcher::Searcher(const Index &index, uint32_t k, uint32_t max_pool) : impl_(new Impl(*index.impl_, k, max_pool)) {
  if (k == 0 || max_pool < k) throw std::invalid_argument("need 0 < k <= max_pool");
}
Searcher::~Searcher() = default;

uint32_t Searcher::search(const float *query, const Predicate &predicate, Hit *out, SearchStats *stats) {
  return impl_->run(query, *predicate.impl_, nullptr, out, stats);
}
uint32_t Searcher::search(const float *query, const Predicate &predicate, const Setting &setting, Hit *out,
                          SearchStats *stats) {
  return impl_->run(query, *predicate.impl_, &setting, out, stats);
}
uint64_t Searcher::io_errors() const { return impl_->io_errors; }

}  // namespace kuroann
