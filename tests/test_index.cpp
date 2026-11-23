// End-to-end checks on synthetic clustered data against a brute-force oracle over the vector ids.
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "kuroann/index.h"

using namespace kuroann;

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
  std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
  if (!ok) ++failures;
}

struct Data {
  uint32_t n, dim;
  std::vector<float> x;
  std::vector<std::vector<int32_t>> labels;
  std::vector<uint32_t> price;
  std::vector<std::string> city;
};

template <class T>
void put(std::ofstream &f, const std::vector<T> &v) {
  f.write((const char *) v.data(), (std::streamsize) (v.size() * sizeof(T)));
}

Data make_data(uint32_t n, uint32_t dim, uint32_t nlabels) {
  std::mt19937_64 rng(7);
  std::normal_distribution<float> gauss;
  const uint32_t clusters = 64;
  std::vector<float> centers(clusters * dim);
  for (auto &c : centers) c = gauss(rng) * 4.0f;
  Data d{n, dim, std::vector<float>((size_t) n * dim), {}, {}, {}};
  d.labels.resize(n);
  d.price.resize(n);
  d.city.resize(n);
  const std::vector<std::string> names = {"lima", "oslo", "kyiv", "rome", "bern", "doha"};
  for (uint32_t i = 0; i < n; i++) {
    const uint32_t c = (uint32_t) (rng() % clusters);
    for (uint32_t k = 0; k < dim; k++) d.x[(size_t) i * dim + k] = centers[(size_t) c * dim + k] + gauss(rng);
    for (uint32_t l = 0; l < nlabels; l++) {
      if (rng() % (1ull << (l + 1)) == 0) d.labels[i].push_back((int32_t) l);
    }
    if (rng() % 4 == 0) d.labels[i].push_back((int32_t) (nlabels + c % 8));
    std::sort(d.labels[i].begin(), d.labels[i].end());
    d.price[i] = (uint32_t) (rng() % 100000);
    d.city[i] = names[rng() % names.size()];
  }
  return d;
}

void write_inputs(const Data &d, uint32_t columns, const std::string &dir) {
  std::ofstream b(dir + "/base.fbin", std::ios::binary);
  const uint32_t h[2] = {d.n, d.dim};
  b.write((const char *) h, 8);
  b.write((const char *) d.x.data(), (std::streamsize) ((size_t) d.n * d.dim * sizeof(float)));
  b.close();
  std::vector<int64_t> ptr(d.n + 1, 0);
  std::vector<int32_t> idx;
  for (uint32_t i = 0; i < d.n; i++) {
    idx.insert(idx.end(), d.labels[i].begin(), d.labels[i].end());
    ptr[i + 1] = (int64_t) idx.size();
  }
  std::vector<float> val(idx.size(), 1.0f);
  std::ofstream s(dir + "/labels.spmat", std::ios::binary);
  const int64_t sh[3] = {(int64_t) d.n, (int64_t) columns, (int64_t) idx.size()};
  s.write((const char *) sh, 24);
  put(s, ptr);
  put(s, idx);
  put(s, val);
  s.close();
  std::ofstream p(dir + "/price.u32", std::ios::binary);
  put(p, d.price);
  p.close();
  std::ofstream c(dir + "/city.txt");
  for (const auto &v : d.city) c << v << "\n";
}

struct Case {
  std::string expr;
  std::function<bool(uint32_t)> member;
};

std::vector<std::pair<float, uint32_t>> exact_top(const Data &d, const float *q, const std::function<bool(uint32_t)> &member, uint32_t k) {
  std::vector<std::pair<float, uint32_t>> all;
  for (uint32_t i = 0; i < d.n; i++) {
    if (!member(i)) continue;
    float s = 0;
    const float *x = d.x.data() + (size_t) i * d.dim;
    for (uint32_t t = 0; t < d.dim; t++) {
      const float e = q[t] - x[t];
      s += e * e;
    }
    all.push_back({s, i});
  }
  const size_t c = std::min<size_t>(k, all.size());
  std::partial_sort(all.begin(), all.begin() + c, all.end());
  all.resize(c);
  return all;
}

bool same(const std::vector<std::pair<float, uint32_t>> &truth, const Hit *h, uint32_t n) {
  if (truth.size() != n) return false;
  for (uint32_t i = 0; i < n; i++) {
    if (truth[i].second != h[i].id || std::fabs(truth[i].first - h[i].distance) > 1e-3f * (1 + truth[i].first)) return false;
  }
  return true;
}

void write_stop_table(const std::string &path, float error) {
  std::ofstream f(path, std::ios::binary);
  const uint32_t h[4] = {0x44334331u, 2, 1, 0};
  const int32_t bounds[3] = {0, 16, 100000};
  const float alpha = 0.01f;
  const float q[2] = {error, error};
  f.write((const char *) h, 16);
  f.write((const char *) bounds, 12);
  f.write((const char *) &alpha, 4);
  f.write((const char *) q, 8);
}

template <class T>
std::vector<T> slurp(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  std::vector<T> v((size_t) f.tellg() / sizeof(T));
  f.seekg(0);
  f.read((char *) v.data(), (std::streamsize) (v.size() * sizeof(T)));
  return v;
}

template <class T>
void spit(const std::string &path, const std::vector<T> &v) {
  std::ofstream f(path, std::ios::binary);
  put(f, v);
}

bool same_file(const std::string &a, const std::string &b) { return slurp<char>(a) == slurp<char>(b); }

// The index of the first `keep` vectors, with the centroids and codebooks of the full index.
void write_prefix_index(const std::string &full, const std::string &out, uint32_t keep) {
  std::vector<int32_t> meta = slurp<int32_t>(full + ".meta");
  const size_t m = (size_t) meta[3];
  const int per_block = 4096 / (meta[1] * meta[6]);
  const std::vector<int64_t> lptr = slurp<int64_t>(full + ".lptr");
  const std::vector<int32_t> lids = slurp<int32_t>(full + ".lids");
  const std::vector<uint8_t> codes = slurp<uint8_t>(full + ".codes");
  const std::vector<int64_t> siptr = slurp<int64_t>(full + ".siptr");
  const std::vector<uint32_t> sidx = slurp<uint32_t>(full + ".sidx");
  std::vector<int64_t> new_lptr(lptr.size(), 0), new_siptr(siptr.size(), 0);
  std::vector<int32_t> new_lids;
  std::vector<uint8_t> new_codes;
  std::vector<uint32_t> new_sidx, map(lids.size(), UINT32_MAX);
  for (size_t c = 0; c + 1 < lptr.size(); c++) {
    for (int64_t p = lptr[c]; p < lptr[c + 1]; p++) {
      if ((uint32_t) lids[p] >= keep) continue;
      map[p] = (uint32_t) new_lids.size();
      new_lids.push_back(lids[p]);
      new_codes.insert(new_codes.end(), codes.begin() + p * (int64_t) m, codes.begin() + (p + 1) * (int64_t) m);
    }
    new_lptr[c + 1] = (int64_t) new_lids.size();
  }
  for (size_t l = 0; l + 1 < siptr.size(); l++) {
    for (int64_t e = siptr[l]; e < siptr[l + 1]; e++) {
      if (map[sidx[e]] != UINT32_MAX) new_sidx.push_back(map[sidx[e]]);
    }
    new_siptr[l + 1] = (int64_t) new_sidx.size();
  }
  std::vector<uint8_t> blocks = slurp<uint8_t>(full + ".vectors");
  const size_t vector_bytes = (size_t) meta[1] * meta[6];
  const size_t kept_blocks = ((size_t) keep + per_block - 1) / per_block;
  blocks.resize(kept_blocks * 4096);
  for (size_t id = keep; id < kept_blocks * per_block; id++) {
    std::fill_n(blocks.begin() + (id / per_block) * 4096 + (id % per_block) * vector_bytes, vector_bytes, (uint8_t) 0);
  }
  meta[0] = meta[7] = (int32_t) keep;
  spit(out + ".meta", meta);
  spit(out + ".coarse", slurp<float>(full + ".coarse"));
  spit(out + ".book", slurp<float>(full + ".book"));
  spit(out + ".lptr", new_lptr);
  spit(out + ".lids", new_lids);
  spit(out + ".codes", new_codes);
  spit(out + ".siptr", new_siptr);
  spit(out + ".sidx", new_sidx);
  spit(out + ".vectors", blocks);
}

void test_updates(const std::string &dir, const std::string &attributes_prefix) {
  const uint32_t n = 20000, added = 600, dim = 64, nlabels = 6, columns = nlabels + 8, k = 10;
  Data d = make_data(n + added, dim, nlabels);
  const uint32_t keep = 18001;
  d.n = n;
  write_inputs(d, columns, dir);
  BuildOptions o;
  o.partitions = 32;
  o.pq_pieces = 16;
  o.iterations = 8;
  const std::string full = dir + "/u_full", part = dir + "/u_part";
  build_index(dir + "/base.fbin", dir + "/labels.spmat", {}, full, o);
  write_prefix_index(full, part, keep);

  std::mt19937_64 rng(13);
  std::vector<std::vector<float>> queries(12, std::vector<float>(dim));
  for (auto &q : queries) {
    const uint32_t a = (uint32_t) (rng() % (n + added));
    std::normal_distribution<float> g;
    for (uint32_t t = 0; t < dim; t++) q[t] = d.x[(size_t) a * dim + t] + g(rng) * 0.5f;
  }
  // A new vector under both kids of an OR must come back once.
  for (uint32_t i = keep; i < n; i++) {
    if (std::binary_search(d.labels[i].begin(), d.labels[i].end(), 3) && std::binary_search(d.labels[i].begin(), d.labels[i].end(), 4)) {
      queries.emplace_back(d.x.begin() + (size_t) i * dim, d.x.begin() + (size_t) (i + 1) * dim);
    }
  }
  check(queries.size() > 12, "the held-out vectors include one with both labels");

  std::vector<uint8_t> alive(n + added, 0);
  std::fill(alive.begin(), alive.begin() + keep, (uint8_t) 1);
  auto has = [&](uint32_t i, int32_t l) { return std::binary_search(d.labels[i].begin(), d.labels[i].end(), l); };
  const std::vector<Case> cases = {
      {"ALL", [&](uint32_t) { return true; }},
      {"L 0", [&](uint32_t i) { return has(i, 0); }},
      {"L 5", [&](uint32_t i) { return has(i, 5); }},
      {"NOT L 0", [&](uint32_t i) { return !has(i, 0); }},
      {"AND 2 L 0 L 1", [&](uint32_t i) { return has(i, 0) && has(i, 1); }},
      {"OR 2 L 3 L 4", [&](uint32_t i) { return has(i, 3) || has(i, 4); }},
      {"AND 2 L 12 NOT L 0", [&](uint32_t i) { return has(i, 12) && !has(i, 0); }},
      {"OR 2 AND 2 L 1 L 2 NOT OR 2 L 0 L 8", [&](uint32_t i) { return (has(i, 1) && has(i, 2)) || !(has(i, 0) || has(i, 8)); }},
      {"AND 2 OR 2 L 1 L 2 OR 2 L 2 L 3", [&](uint32_t i) { return (has(i, 1) || has(i, 2)) && (has(i, 2) || has(i, 3)); }},
  };

  auto exact_everywhere = [&](Index &index, uint32_t nlist) {
    d.n = n + added;
    Searcher searcher(index, k, n + added);
    std::vector<Hit> out(k);
    SearchStats st;
    Setting codes;
    codes.route = Route::kExhaustive;
    codes.pool = n + added;
    Setting cells;
    cells.route = Route::kPartition;
    cells.partitions = nlist;
    cells.pool = n + added;
    bool ok = true;
    for (const Case &c : cases) {
      Predicate p(index, c.expr);
      auto member = [&](uint32_t i) { return alive[i] && c.member(i); };
      for (const auto &q : queries) {
        const auto truth = exact_top(d, q.data(), member, k);
        uint32_t got = searcher.search(q.data(), p, codes, out.data(), &st);
        if (!same(truth, out.data(), got)) {
          ok = false;
          std::printf("  exhaustive route differs: %s\n", c.expr.c_str());
        }
        got = searcher.search(q.data(), p, cells, out.data(), &st);
        if (!same(truth, out.data(), got)) {
          ok = false;
          std::printf("  partition route differs: %s\n", c.expr.c_str());
        }
      }
    }
    return ok;
  };

  {
    Index index(part);
    check(index.size() == keep, "the shorter index has the kept vectors");
    const std::vector<std::vector<int32_t>> labels(d.labels.begin() + keep, d.labels.begin() + n);
    const std::vector<uint32_t> ids = index.insert(d.x.data() + (size_t) keep * dim, n - keep, labels);
    check(ids.size() == n - keep && ids.front() == keep && ids.back() == n - 1, "new vectors take the next ids");
    check(index.size() == n, "inserts count at once");
    std::fill(alive.begin(), alive.begin() + n, (uint8_t) 1);
    check(exact_everywhere(index, 32), "with a delta segment, both routes are exact");
    index.merge();
    index.save();
    check(index.size() == n, "a merge keeps the count");
    check(exact_everywhere(index, 32), "after a merge, both routes are exact");
  }
  bool identical = true;
  for (const char *name : {".meta", ".lptr", ".lids", ".codes", ".siptr", ".sidx", ".vectors"}) {
    if (!same_file(full + name, part + name)) {
      identical = false;
      std::printf("  %s differs from a fresh build\n", name);
    }
  }
  check(identical, "inserting held-out vectors and merging gives the files of a fresh build");

  const std::string plan = dir + "/u_plan.txt";
  {
    std::ofstream f(plan);
    f << "vmax=1000 np=0 L=512 alpha=0\nvmax=1000000 np=32 L=2048 alpha=0\n";
  }
  Index index(part);
  index.load_plan(plan);
  std::vector<uint32_t> removed;
  for (uint32_t id = 0; id < n; id += 5) removed.push_back(id);
  bool removed_ok = true;
  for (uint32_t id : removed) removed_ok = removed_ok && index.remove(id);
  check(removed_ok, "removing present ids succeeds");
  check(!index.remove(0) && !index.remove(n + added), "removing a removed or unknown id returns false");
  for (uint32_t id : removed) alive[id] = 0;
  check(index.size() == n - removed.size(), "removals count at once");
  check(exact_everywhere(index, 32), "with tombstones, both routes are exact");

  const std::vector<std::vector<int32_t>> more(d.labels.begin() + n, d.labels.end());
  const std::vector<uint32_t> more_ids = index.insert(d.x.data() + (size_t) n * dim, added, more);
  check(more_ids.front() == n && more_ids.back() == n + added - 1, "ids continue after a reload");
  std::fill(alive.begin() + n, alive.end(), (uint8_t) 1);
  for (uint32_t id = n; id < n + added; id += 3) {
    check(index.remove(id), "removing a delta entry succeeds");
    alive[id] = 0;
  }
  check(exact_everywhere(index, 32), "with a delta segment and tombstones, both routes are exact");

  auto live_count = [&](const std::function<bool(uint32_t)> &member) {
    double c = 0;
    for (uint32_t i = 0; i < n + added; i++) c += alive[i] && member(i);
    return c;
  };
  {
    double packed = 0, in_delta = 0;
    for (uint32_t i = 0; i < n + added; i++) {
      if (has(i, 0) && has(i, 1)) (i < n ? packed : in_delta) += 1;
    }
    const double expected = (packed + in_delta) * (double) index.size() / (double) (n + added);
    const double c0 = live_count([&](uint32_t i) { return has(i, 0); }), c1 = live_count([&](uint32_t i) { return has(i, 1); });
    const uint64_t between = (uint64_t) std::ceil(expected) + 1;
    check((double) between < std::min(c0, c1), "the AND of the count check is narrower than each of its labels");
    const std::string narrow = dir + "/u_plan_narrow.txt";
    {
      std::ofstream f(narrow);
      f << "vmax=" << between << " np=0 L=512 alpha=0\nvmax=1000000 np=32 L=2048 alpha=0\n";
    }
    index.load_plan(narrow);
    Searcher planned(index, k, index.plan_max_pool());
    std::vector<Hit> out(k);
    SearchStats st;
    Predicate l5(index, "L 5");
    planned.search(queries[0].data(), l5, out.data(), &st);
    check(st.count == live_count([&](uint32_t i) { return has(i, 5); }), "the plan reads a label's current count");
    Predicate both(index, "AND 2 L 0 L 1");
    planned.search(queries[0].data(), both, out.data(), &st);
    check(std::fabs(st.count - expected) < 1e-6 && st.exhaustive, "with updates pending, an AND is counted over the packed arrays and the delta segment, so it takes the row of its matches");
    index.load_plan(plan);
  }

  index.merge();
  check(exact_everywhere(index, 32), "after a merge that drops vectors, both routes are exact");
  check(!index.remove(removed[1]) && !index.remove(n), "a merged-out id stays absent");
  {
    Searcher planned(index, k, index.plan_max_pool());
    std::vector<Hit> out(k);
    SearchStats st;
    Predicate both(index, "AND 2 L 0 L 1");
    planned.search(queries[0].data(), both, out.data(), &st);
    check(st.count == live_count([&](uint32_t i) { return has(i, 0) && has(i, 1); }), "after a merge, an AND has its exact count");
  }
  index.save();
  {
    Index again(part);
    check(again.size() == index.size(), "a saved index keeps its vectors");
    check(exact_everywhere(again, 32), "a saved index is exact on both routes");
    check(!again.remove(removed[1]), "a saved index keeps merged-out ids absent");
    const std::vector<uint32_t> next = again.insert(d.x.data(), 1, {d.labels[0]});
    check(next[0] == n + added, "a saved index keeps the next id");
  }

  {
    Index attributes(attributes_prefix);
    bool refused = false;
    try {
      attributes.remove(0);
    } catch (const std::logic_error &) {
      refused = true;
    }
    check(refused, "an index with attribute fields refuses updates");
    Index labeled(part);
    bool bad = false;
    try {
      labeled.insert(d.x.data(), 1, {{(int32_t) columns}});
    } catch (const std::invalid_argument &) {
      bad = true;
    }
    check(bad && labeled.size() == n + added - (uint32_t) removed.size() - added / 3, "a label out of range is refused");
    bool pending = false;
    labeled.insert(d.x.data(), 1, {{0}});
    try {
      labeled.save();
    } catch (const std::logic_error &) {
      pending = true;
    }
    check(pending, "saving with unmerged updates is refused");
  }

  {
    const uint32_t n2 = 801, dim2 = 512;
    Data e = make_data(n2 + 3, dim2, 2);
    e.n = n2;
    write_inputs(e, 10, dir);
    BuildOptions o2;
    o2.partitions = 8;
    o2.pq_pieces = 16;
    o2.iterations = 4;
    build_index(dir + "/base.fbin", dir + "/labels.spmat", {}, dir + "/u_two", o2);
    Index two(dir + "/u_two");
    const std::vector<std::vector<int32_t>> labels(e.labels.begin() + n2, e.labels.end());
    two.insert(e.x.data() + (size_t) n2 * dim2, 3, labels);
    Searcher sr(two, k, n2 + 3);
    Predicate everything(two, "ALL");
    Setting full_scan;
    full_scan.route = Route::kExhaustive;
    full_scan.pool = n2 + 3;
    std::vector<Hit> out(k);
    SearchStats st;
    bool self = true;
    for (uint32_t id : {799u, 800u, 801u, 802u, 803u}) {
      const uint32_t got = sr.search(e.x.data() + (size_t) id * dim2, everything, full_scan, out.data(), &st);
      self = self && got == k && out[0].id == id && out[0].distance == 0.0f;
    }
    check(self, "inserts fill a half-used block and start new ones");
  }

  {
    const uint32_t n3 = 4000, dim3 = 16;
    std::mt19937_64 byte_rng(5);
    std::vector<uint8_t> bytes((size_t) (n3 + 2) * dim3);
    for (auto &b : bytes) b = (uint8_t) (byte_rng() % 256);
    std::ofstream f(dir + "/bytes.bin", std::ios::binary);
    const uint32_t h[2] = {n3, dim3};
    f.write((const char *) h, 8);
    f.write((const char *) bytes.data(), (std::streamsize) ((size_t) n3 * dim3));
    f.close();
    BuildOptions o3;
    o3.partitions = 8;
    o3.pq_pieces = 4;
    o3.iterations = 4;
    build_index(dir + "/bytes.bin", "", {}, dir + "/u_bytes", o3);
    Index small(dir + "/u_bytes");
    std::vector<float> v(dim3 * 2);
    for (uint32_t t = 0; t < dim3 * 2; t++) v[t] = (float) bytes[(size_t) n3 * dim3 + t];
    small.insert(v.data(), 2);
    Searcher sr(small, k, n3 + 2);
    Predicate everything(small, "ALL");
    Setting full_scan;
    full_scan.route = Route::kExhaustive;
    full_scan.pool = n3 + 2;
    std::vector<Hit> out(k);
    SearchStats st;
    const uint32_t got = sr.search(v.data() + dim3, everything, full_scan, out.data(), &st);
    check(got == k && out[0].id == n3 + 1 && out[0].distance == 0.0f, "a uint8 index takes new vectors");
    bool refused = false;
    v[0] = 0.5f;
    try {
      small.insert(v.data(), 1);
    } catch (const std::invalid_argument &) {
      refused = true;
    }
    check(refused, "a uint8 index refuses other values");
  }
}

}  // namespace

int main() {
  char tmpl[] = "kuroann_test_XXXXXX";
  const std::string dir = mkdtemp(tmpl);
  const uint32_t n = 40000, dim = 64, nlabels = 10, columns = nlabels + 8, k = 10;
  const Data d = make_data(n, dim, nlabels);
  write_inputs(d, columns, dir);

  BuildOptions o;
  o.partitions = 64;
  o.pq_pieces = 16;
  o.iterations = 8;
  const std::string prefix = dir + "/idx";
  build_index(dir + "/base.fbin", dir + "/labels.spmat",
              {{AttributeSource::Kind::kRange, "price", dir + "/price.u32"}, {AttributeSource::Kind::kString, "city", dir + "/city.txt"}},
              prefix, o);
  Index index(prefix);
  check(index.size() == n && index.dimension() == dim && index.partitions() == 64 && index.pq_pieces() == 16, "index shape");
  check(index.labels() == columns, "label count");

  auto has = [&](uint32_t i, int32_t l) { return std::binary_search(d.labels[i].begin(), d.labels[i].end(), l); };
  std::vector<Case> cases = {
      {"ALL", [](uint32_t) { return true; }},
      {"L 0", [&](uint32_t i) { return has(i, 0); }},
      {"L 6", [&](uint32_t i) { return has(i, 6); }},
      {"NOT L 0", [&](uint32_t i) { return !has(i, 0); }},
      {"AND 2 L 0 L 1", [&](uint32_t i) { return has(i, 0) && has(i, 1); }},
      {"OR 2 L 3 L 4", [&](uint32_t i) { return has(i, 3) || has(i, 4); }},
      {"R price 100 5000", [&](uint32_t i) { return d.price[i] >= 100 && d.price[i] < 5000; }},
      {"AND 2 L 1 R price 0 20000", [&](uint32_t i) { return has(i, 1) && d.price[i] < 20000; }},
      {"AND 2 R price 0 60000 L 0", [&](uint32_t i) { return d.price[i] < 60000 && has(i, 0); }},
      {"AND 3 L 1 R price 0 70000 NOT S city eq doha", [&](uint32_t i) { return has(i, 1) && d.price[i] < 70000 && d.city[i] != "doha"; }},
      {"AND 3 NOT L 0 L 2 NOT R price 1000 30000", [&](uint32_t i) { return !has(i, 0) && has(i, 2) && !(d.price[i] >= 1000 && d.price[i] < 30000); }},
      {"NOT R price 0 90000", [&](uint32_t i) { return d.price[i] >= 90000; }},
      {"S city eq rome", [&](uint32_t i) { return d.city[i] == "rome"; }},
      {"S city in lima,oslo", [&](uint32_t i) { return d.city[i] == "lima" || d.city[i] == "oslo"; }},
      {"S city like %o%", [&](uint32_t i) { return d.city[i].find('o') != std::string::npos; }},
      {"AND 2 OR 2 L 0 L 2 NOT S city eq doha", [&](uint32_t i) { return (has(i, 0) || has(i, 2)) && d.city[i] != "doha"; }},
      {"OR 2 AND 2 L 12 R price 0 50000 NOT OR 2 L 0 L 1", [&](uint32_t i) { return (has(i, 12) && d.price[i] < 50000) || !(has(i, 0) || has(i, 1)); }},
  };
  std::mt19937_64 rng(11);
  std::vector<std::vector<float>> queries(20, std::vector<float>(dim));
  for (auto &q : queries) {
    const uint32_t a = (uint32_t) (rng() % n);
    std::normal_distribution<float> g;
    for (uint32_t t = 0; t < dim; t++) q[t] = d.x[(size_t) a * dim + t] + g(rng) * 0.5f;
  }

  Searcher searcher(index, k, n);
  std::vector<Hit> out(k);
  SearchStats st;

  Setting all_codes;
  all_codes.route = Route::kExhaustive;
  all_codes.pool = n;
  Setting all_cells;
  all_cells.route = Route::kPartition;
  all_cells.partitions = 64;
  all_cells.pool = n;
  for (const Case &c : cases) {
    Predicate p(index, c.expr);
    bool ex = true, pa = true, count_ok = true;
    uint64_t matches = 0;
    for (uint32_t i = 0; i < n; i++) matches += c.member(i);
    for (const auto &q : queries) {
      const auto truth = exact_top(d, q.data(), c.member, k);
      uint32_t got = searcher.search(q.data(), p, all_codes, out.data(), &st);
      ex = ex && same(truth, out.data(), got);
      count_ok = count_ok && st.scanned == matches && st.reads == std::min<uint64_t>(matches, n);
      got = searcher.search(q.data(), p, all_cells, out.data(), &st);
      pa = pa && same(truth, out.data(), got) && st.scanned == matches;
    }
    check(ex, "exhaustive route is exact: " + c.expr);
    check(pa, "partition route over all partitions is exact: " + c.expr);
    check(count_ok, "exhaustive route scans and reads every match: " + c.expr);
  }

  Setting small = all_codes;
  small.pool = 32;
  Predicate wide(index, "L 0");
  searcher.search(queries[0].data(), wide, small, out.data(), &st);
  check(st.reads == 32, "the pool limits the reads");

  const std::string t_never = dir + "/never.d3cal", t_zero = dir + "/zero.d3cal";
  write_stop_table(t_never, -1e9f);
  write_stop_table(t_zero, 0.0f);
  Setting stop = all_codes;
  stop.pool = 512;
  stop.alpha = 0.01f;
  stop.table = t_never;
  Setting whole = stop;
  whole.alpha = 0;
  bool identical = true, read_all = true, fewer = false;
  uint64_t reads_zero = 0, reads_whole = 0;
  double kept = 0;
  std::vector<Hit> a(k), b(k);
  for (const auto &q : queries) {
    uint32_t na = searcher.search(q.data(), wide, stop, a.data(), &st);
    read_all = read_all && st.reads == 512;
    uint32_t nb = searcher.search(q.data(), wide, whole, b.data(), &st);
    reads_whole += st.reads;
    identical = identical && na == nb;
    for (uint32_t i = 0; i < std::min(na, nb); i++) identical = identical && a[i].id == b[i].id;
    Setting z = stop;
    z.table = t_zero;
    uint32_t nz = searcher.search(q.data(), wide, z, a.data(), &st);
    reads_zero += st.reads;
    for (uint32_t i = 0; i < nz; i++) {
      for (uint32_t j = 0; j < nb; j++) kept += a[i].id == b[j].id;
    }
  }
  fewer = reads_zero < reads_whole;
  check(identical, "a stop table that never fires gives the same answer as reading the pool");
  check(read_all, "a stop table that never fires reads the whole pool");
  check(fewer, "a zero-error stop table reads fewer vectors");
  check(kept / (20.0 * k) > 0.5, "a zero-error stop table keeps most of the answer");

  const std::string plan = dir + "/plan.txt";
  {
    std::ofstream f(plan);
    f << "# count ranges\n";
    f << "vmax=1000 np=0 C=0 L=64 alpha=0\n";
    f << "vmax=8000 np=0 C=0 L=128 alpha=0.01 table=never.d3cal\n";
    f << "vmax=100000 np=32 C=0 L=256 alpha=0\n";
  }
  index.load_plan(plan);
  check(index.plan_max_pool() == 256, "plan pool limit");
  Searcher planned(index, k, index.plan_max_pool());
  struct RowCase {
    std::string expr;
    int row;
    bool exhaustive;
  };
  const std::vector<RowCase> rows = {{"L 6", 0, true}, {"L 3", 1, true}, {"L 0", 2, false}, {"ALL", 2, false}, {"AND 2 L 0 L 1", 1, true}};
  bool row_ok = true;
  for (const RowCase &r : rows) {
    Predicate p(index, r.expr);
    planned.search(queries[0].data(), p, out.data(), &st);
    if (st.plan_row != r.row || st.exhaustive != r.exhaustive) {
      row_ok = false;
      std::printf("  %s: row %d exhaustive %d count %.0f\n", r.expr.c_str(), st.plan_row, (int) st.exhaustive, st.count);
    }
  }
  check(row_ok, "the plan row follows the match count");

  auto rejected = [&](const std::string &e) {
    try {
      Predicate p(index, e);
    } catch (const std::invalid_argument &) {
      return true;
    }
    return false;
  };
  check(rejected("") && rejected("L 99") && rejected("R nope 0 1") && rejected("AND 2 L 0") && rejected("L 0 L 1") && rejected("S city has x"),
        "bad expressions are rejected");

  {
    const uint32_t n2 = 5000, dim2 = 512;
    Data e = make_data(n2, dim2, 2);
    write_inputs(e, 10, dir);
    BuildOptions o2;
    o2.partitions = 16;
    o2.pq_pieces = 16;
    o2.iterations = 4;
    build_index(dir + "/base.fbin", dir + "/labels.spmat", {}, dir + "/two", o2);
    Index two(dir + "/two");
    Searcher sr(two, k, n2);
    Predicate everything(two, "ALL");
    Setting full;
    full.route = Route::kExhaustive;
    full.pool = n2;
    bool self = true;
    for (uint32_t id : {0u, 1u, 2u, 3u, 2499u, 4998u, 4999u}) {
      const uint32_t got = sr.search(e.x.data() + (size_t) id * dim2, everything, full, out.data(), &st);
      self = self && got == k && out[0].id == id && out[0].distance == 0.0f;
    }
    check(self, "two vectors per block: each vector finds itself at distance 0");
  }

  {
    // Label 0 listed twice in some rows, and a second build into the same prefix without the price attribute.
    const uint32_t n3 = 3000, dim3 = 32;
    Data e = make_data(n3, dim3, 4);
    for (uint32_t i = 0; i < n3; i += 3) e.labels[i].push_back(0);
    write_inputs(e, 12, dir);
    BuildOptions o3;
    o3.partitions = 8;
    o3.pq_pieces = 8;
    o3.iterations = 4;
    const std::string twice = dir + "/twice";
    build_index(dir + "/base.fbin", dir + "/labels.spmat", {{AttributeSource::Kind::kRange, "price", dir + "/price.u32"}}, twice, o3);
    build_index(dir + "/base.fbin", dir + "/labels.spmat", {}, twice, o3);
    Index again(twice);
    Searcher sa(again, k, n3);
    Predicate zero(again, "L 0");
    Setting full3;
    full3.route = Route::kExhaustive;
    full3.pool = n3;
    const uint32_t got = sa.search(e.x.data(), zero, full3, out.data(), &st);
    std::set<uint32_t> distinct;
    for (uint32_t j = 0; j < got; j++) distinct.insert(out[j].id);
    const auto members = std::count_if(e.labels.begin(), e.labels.end(), [](const std::vector<int32_t> &r) { return std::find(r.begin(), r.end(), 0) != r.end(); });
    check(got == k && distinct.size() == got && st.count == (double) members, "a label listed twice in a row counts once");
    bool dropped = false;
    try {
      Predicate old(again, "R price 0 100");
    } catch (const std::invalid_argument &) {
      dropped = true;
    }
    check(dropped, "a rebuild without attributes drops the attribute fields of the previous build");
  }

  {
    // Labels nlabels..nlabels+7 each belong to one group of clusters, so near a query of that group they are far denser
    // than their share of the index. Label 1 is spread evenly.
    uint32_t a = 0;
    while (a < n && (d.labels[a].empty() || d.labels[a].back() < (int32_t) nlabels)) a++;
    const int32_t g = d.labels[a].back(), g2 = (int32_t) nlabels + (g - (int32_t) nlabels + 1) % 8;
    const std::string lplan = dir + "/local_plan.txt";
    {
      std::ofstream f(lplan);
      f << "vmax=3000 np=0 L=256 alpha=0\nvmax=1000000 np=8 L=256 alpha=0\n";
    }
    Index global(prefix);
    global.load_plan(lplan);
    CountOptions lc;
    lc.local_partitions = 1;
    Index local(prefix, lc);
    local.load_plan(lplan);
    Searcher sg(global, k, 256), sl(local, k, 256);
    const float *q = d.x.data() + (size_t) a * dim;
    const std::string one = "L " + std::to_string(g), two = "OR 2 L " + std::to_string(g) + " L " + std::to_string(g2);
    bool off = true, on = true;
    for (const std::string &e : {one, two}) {
      Predicate pg(global, e), pl(local, e);
      sg.search(q, pg, out.data(), &st);
      off = off && !st.local && st.plan_row == 0;
      const double count = st.count;
      const uint32_t got = sl.search(q, pl, out.data(), &st);
      on = on && st.local && st.plan_row == 1 && st.count > 2 * count && got == k;
    }
    check(off, "without the local count, a clustered label takes the row of its count");
    check(on, "the local count moves a clustered label and an OR of such labels to the row of their local density");
    bool alone = true;
    for (const std::string &e : {std::string("L 1"), std::string("OR 2 L 1 L 1"), std::string("OR 3 L 1 L 1 L 1")}) {
      Predicate spread(local, e);
      sl.search(q, spread, out.data(), &st);
      alone = alone && !st.local;
    }
    check(alone, "the local count leaves an evenly spread label alone, also when an OR lists it more than once");
    Predicate both(local, "AND 2 L " + std::to_string(g) + " L 1");
    sl.search(q, both, out.data(), &st);
    check(!st.local, "the local count leaves an AND alone");
    // A row boundary between the largest label and the sum of the labels sends an OR to the exact count, the union.
    const std::string uplan = dir + "/local_union_plan.txt";
    {
      std::ofstream f(uplan);
      f << "vmax=15000 np=0 L=256 alpha=0\nvmax=1000000 np=8 L=256 alpha=0\n";
    }
    Index local_union(prefix, lc);
    local_union.load_plan(uplan);
    Searcher su(local_union, k, 256);
    Predicate thrice(local_union, "OR 3 L 1 L 1 L 1");
    su.search(q, thrice, out.data(), &st);
    check(!st.local && st.plan_row == 0, "the local count of an OR counts each match once");
    bool rejected_ratio = false;
    try {
      CountOptions bad;
      bad.local_ratio = 0;
      Index x(prefix, bad);
    } catch (const std::invalid_argument &) {
      rejected_ratio = true;
    }
    check(rejected_ratio, "a non-positive local ratio is rejected");
  }

  test_updates(dir, prefix);

  std::string cmd = "rm -rf -- '" + dir + "'";
  if (std::system(cmd.c_str()) != 0) return 2;
  std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
  return failures ? 1 : 0;
}
