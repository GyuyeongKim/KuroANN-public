#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>

#include "internal.h"

namespace kuroann {

namespace {

constexpr int kBlock = 4096;
constexpr uint32_t kAbsent = UINT32_MAX;

void require_label_filters(const Index::Impl &x) {
  if (!x.ranges.empty() || !x.strings.empty()) throw std::logic_error("updates serve label filters, and this index has attribute fields");
}

void start_updates(Index::Impl &x) {
  Updates &u = x.updates;
  if (u.pending) return;
  u.delta.assign(x.nlist, DeltaPartition());
  u.label_delta.assign(x.nlabels, {});
  u.live.resize(x.nlabels);
  for (int l = 0; l < x.nlabels; l++) u.live[l] = x.siptr[l + 1] - x.siptr[l];
  u.live_total = x.n;
  u.first_delta_id = x.next_id;
  u.removed.assign(((size_t) x.next_id + 7) / 8, 0);
  u.pending = true;
}

void index_positions(Index::Impl &x) {
  Updates &u = x.updates;
  u.position_of.assign(x.next_id, kAbsent);
  for (int32_t p = 0; p < x.n; p++) u.position_of[x.lids[p]] = (uint32_t) p;
  const int64_t entries = x.siptr[x.nlabels];
  u.position_label_start.assign((size_t) x.n + 1, 0);
  for (int64_t e = 0; e < entries; e++) u.position_label_start[x.sidx[e] + 1]++;
  for (int32_t p = 0; p < x.n; p++) u.position_label_start[p + 1] += u.position_label_start[p];
  u.position_labels.resize((size_t) entries);
  std::vector<int64_t> fill(u.position_label_start.begin(), u.position_label_start.end() - 1);
  for (int l = 0; l < x.nlabels; l++) {
    for (int64_t e = x.siptr[l]; e < x.siptr[l + 1]; e++) u.position_labels[fill[x.sidx[e]]++] = l;
  }
}

void write_vectors(Index::Impl &x, const float *vectors, size_t count) {
  const std::string path = x.prefix + ".vectors";
  const int fd = open(path.c_str(), O_WRONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path + " for writing");
  std::vector<uint8_t> block(kBlock);
  long end = x.vectors_size;
  bool ok = true;
  for (size_t i = 0; i < count && ok; i++) {
    const long id = (long) x.next_id + (long) i;
    const long first = (id / x.per_block) * kBlock;
    const int slot = (int) (id % x.per_block);
    const float *v = vectors + i * x.dim;
    std::fill(block.begin(), block.end(), (uint8_t) 0);
    if (x.element_bytes == 4) {
      std::memcpy(block.data(), v, x.vector_bytes);
    } else {
      for (int d = 0; d < x.dim; d++) block[d] = (uint8_t) v[d];
    }
    // The first vector of a block writes all of it, so that the file keeps whole blocks.
    const size_t bytes = slot == 0 ? kBlock : x.vector_bytes;
    ok = pwrite(fd, block.data(), bytes, first + (long) slot * x.vector_bytes) == (ssize_t) bytes;
    end = std::max(end, first + kBlock);
  }
  ok = ok && fdatasync(fd) == 0;
  close(fd);
  if (!ok) throw std::runtime_error("cannot write " + path);
  x.vectors_size = end;
}

}  // namespace

std::vector<uint32_t> Index::insert(const float *vectors, size_t count, const std::vector<std::vector<int32_t>> &labels) {
  Impl &x = *impl_;
  require_label_filters(x);
  if (!labels.empty() && labels.size() != count) throw std::invalid_argument("expected one label list per vector");
  if ((uint64_t) x.next_id + count > (uint64_t) std::numeric_limits<int32_t>::max()) throw std::invalid_argument("too many vectors");
  std::vector<std::vector<int32_t>> sorted(count);
  for (size_t i = 0; i < labels.size(); i++) {
    sorted[i] = labels[i];
    std::sort(sorted[i].begin(), sorted[i].end());
    sorted[i].erase(std::unique(sorted[i].begin(), sorted[i].end()), sorted[i].end());
    for (int32_t l : sorted[i]) {
      if (l < 0 || l >= x.nlabels) throw std::invalid_argument("label " + std::to_string(l) + " out of range");
    }
  }
  if (x.element_bytes == 1) {
    for (size_t i = 0; i < count * x.dim; i++) {
      if (vectors[i] < 0.0f || vectors[i] > 255.0f || vectors[i] != std::floor(vectors[i])) throw std::invalid_argument("this index holds uint8 values");
    }
  }
  if (count == 0) return {};

  std::vector<int32_t> cell(count);
  std::vector<uint8_t> code(count * x.m);
#pragma omp parallel for schedule(dynamic, 16)
  for (size_t i = 0; i < count; i++) {
    cell[i] = encode_vector(vectors + i * x.dim, x.coarse.data(), x.nlist, x.dim, x.book.data(), x.m, x.dsub, x.ksub, &code[i * x.m]);
  }
  write_vectors(x, vectors, count);

  start_updates(x);
  Updates &u = x.updates;
  std::vector<uint32_t> ids(count);
  for (size_t i = 0; i < count; i++) {
    const int32_t id = x.next_id + (int32_t) i;
    ids[i] = (uint32_t) id;
    DeltaPartition &dp = u.delta[cell[i]];
    const int32_t slot = (int32_t) dp.ids.size();
    dp.ids.push_back(id);
    dp.codes.insert(dp.codes.end(), code.begin() + (long) (i * x.m), code.begin() + (long) ((i + 1) * x.m));
    dp.labels.insert(dp.labels.end(), sorted[i].begin(), sorted[i].end());
    dp.label_start.push_back((int64_t) dp.labels.size());
    u.delta_of_id.push_back({cell[i], slot});
    for (int32_t l : sorted[i]) {
      u.label_delta[l].push_back({cell[i], slot});
      u.live[l]++;
    }
  }
  x.next_id += (int32_t) count;
  u.delta_size += (int64_t) count;
  u.live_total += (int64_t) count;
  u.removed.resize(((size_t) x.next_id + 7) / 8, 0);
  return ids;
}

bool Index::remove(uint32_t id) {
  Impl &x = *impl_;
  require_label_filters(x);
  if ((uint64_t) id >= (uint64_t) x.next_id) return false;
  Updates &u = x.updates;
  if (u.pending && ((u.removed[id >> 3] >> (id & 7)) & 1)) return false;
  const int32_t *first, *last;
  if (u.pending && (int32_t) id >= u.first_delta_id) {
    const DeltaRef r = u.delta_of_id[id - (uint32_t) u.first_delta_id];
    const DeltaPartition &dp = u.delta[r.partition];
    first = dp.labels.data() + dp.label_start[r.slot];
    last = dp.labels.data() + dp.label_start[r.slot + 1];
  } else {
    if (u.position_of.empty()) index_positions(x);
    const uint32_t p = u.position_of[id];
    if (p == kAbsent) return false;
    first = u.position_labels.data() + u.position_label_start[p];
    last = u.position_labels.data() + u.position_label_start[p + 1];
  }
  start_updates(x);
  for (const int32_t *l = first; l != last; l++) u.live[*l]--;
  u.removed[id >> 3] |= (uint8_t) (1u << (id & 7));
  u.any_removed = true;
  u.live_total--;
  return true;
}

void Index::merge() {
  Impl &x = *impl_;
  Updates &u = x.updates;
  if (!u.pending) return;
  auto is_removed = [&](int32_t id) { return ((u.removed[(size_t) id >> 3] >> (id & 7)) & 1) != 0; };
  const int m = x.m;

  std::vector<int64_t> lptr((size_t) x.nlist + 1, 0);
#pragma omp parallel for schedule(dynamic, 16)
  for (int c = 0; c < x.nlist; c++) {
    int64_t kept = 0;
    for (int64_t p = x.lptr[c]; p < x.lptr[c + 1]; p++) kept += !is_removed(x.lids[p]);
    for (int32_t id : u.delta[c].ids) kept += !is_removed(id);
    lptr[c + 1] = kept;
  }
  for (int c = 0; c < x.nlist; c++) lptr[c + 1] += lptr[c];
  const int64_t total = lptr[x.nlist];

  // New ids are the largest, so that each partition stays in ascending id order.
  HugeArray<uint8_t> codes;
  uint8_t *code_out = codes.allocate((size_t) total * m);
  std::vector<int32_t> lids((size_t) total);
  std::vector<uint32_t> packed_map((size_t) x.n, kAbsent);
  std::vector<std::vector<uint32_t>> delta_map(x.nlist);
#pragma omp parallel for schedule(dynamic, 16)
  for (int c = 0; c < x.nlist; c++) {
    int64_t w = lptr[c];
    for (int64_t p = x.lptr[c]; p < x.lptr[c + 1]; p++) {
      const int32_t id = x.lids[p];
      if (is_removed(id)) continue;
      std::memcpy(code_out + (size_t) w * m, x.codes.data() + (size_t) p * m, m);
      lids[w] = id;
      packed_map[p] = (uint32_t) w++;
    }
    const DeltaPartition &dp = u.delta[c];
    delta_map[c].assign(dp.ids.size(), kAbsent);
    for (size_t s = 0; s < dp.ids.size(); s++) {
      if (is_removed(dp.ids[s])) continue;
      std::memcpy(code_out + (size_t) w * m, dp.codes.data() + s * m, m);
      lids[w] = dp.ids[s];
      delta_map[c][s] = (uint32_t) w++;
    }
  }

  std::vector<int64_t> siptr((size_t) x.nlabels + 1, 0);
  for (int l = 0; l < x.nlabels; l++) siptr[l + 1] = siptr[l] + u.live[l];
  HugeArray<uint32_t> sidx;
  uint32_t *posting_out = sidx.allocate((size_t) siptr[x.nlabels]);
  std::atomic<bool> counts_ok{true};
#pragma omp parallel for schedule(dynamic, 16)
  for (int l = 0; l < x.nlabels; l++) {
    std::vector<uint32_t> added;
    for (const DeltaRef &r : u.label_delta[l]) {
      const uint32_t np = delta_map[r.partition][r.slot];
      if (np != kAbsent) added.push_back(np);
    }
    std::sort(added.begin(), added.end());
    int64_t w = siptr[l];
    size_t j = 0;
    for (int64_t e = x.siptr[l]; e < x.siptr[l + 1]; e++) {
      const uint32_t np = packed_map[x.sidx[e]];
      if (np == kAbsent) continue;
      while (j < added.size() && added[j] < np && w < siptr[l + 1]) posting_out[w++] = added[j++];
      if (w < siptr[l + 1]) posting_out[w] = np;
      w++;
    }
    while (j < added.size() && w < siptr[l + 1]) posting_out[w++] = added[j++];
    if (w != siptr[l + 1] || j != added.size()) counts_ok = false;
  }
  if (!counts_ok) throw std::logic_error("merge: a label count does not match its members");

  x.codes = std::move(codes);
  x.lids.swap(lids);
  x.lptr.swap(lptr);
  x.siptr.swap(siptr);
  x.sidx = std::move(sidx);
  x.n = (int32_t) total;
  x.updates = Updates();
}

void Index::save() {
  Impl &x = *impl_;
  if (x.updates.pending) throw std::logic_error("merge the updates before saving");
  const int32_t meta[8] = {x.n, x.dim, x.nlist, x.m, x.dsub, x.ksub, x.element_bytes, x.next_id};
  const std::string p = x.prefix;
  write_file(p + ".codes.new", x.codes.data(), (size_t) x.n * x.m);
  write_file(p + ".lptr.new", x.lptr.data(), x.lptr.size());
  write_file(p + ".lids.new", x.lids.data(), x.lids.size());
  write_file(p + ".siptr.new", x.siptr.data(), x.siptr.size());
  write_file(p + ".sidx.new", x.sidx.data(), x.sidx.size());
  write_file(p + ".meta.new", meta, 8);
  for (const char *name : {".codes", ".lptr", ".lids", ".siptr", ".sidx", ".meta"}) {
    if (std::rename((p + name + ".new").c_str(), (p + name).c_str()) != 0) throw std::runtime_error("cannot replace " + p + name);
  }
}

}  // namespace kuroann
