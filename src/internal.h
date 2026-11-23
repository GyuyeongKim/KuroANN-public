#pragma once

#include <sys/mman.h>

#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "kuroann/index.h"

namespace kuroann {

long file_size(const std::string &path);

// Anonymous mapping advised for huge pages before the first touch, then filled from a file.
template <class T>
class HugeArray {
 public:
  HugeArray() = default;
  ~HugeArray() { reset(); }
  HugeArray(const HugeArray &) = delete;
  HugeArray &operator=(const HugeArray &) = delete;
  HugeArray(HugeArray &&o) noexcept : data_(o.data_), n_(o.n_) { o.data_ = nullptr; o.n_ = 0; }
  HugeArray &operator=(HugeArray &&o) noexcept {
    if (this != &o) {
      reset();
      data_ = o.data_; n_ = o.n_;
      o.data_ = nullptr; o.n_ = 0;
    }
    return *this;
  }
  T *allocate(size_t n);
  void load(const std::string &path, size_t n);
  const T *data() const { return data_; }
  const T &operator[](size_t i) const { return data_[i]; }
  size_t size() const { return n_; }

 private:
  void reset() {
    if (data_) munmap(data_, n_ * sizeof(T));
    data_ = nullptr; n_ = 0;
  }
  T *data_ = nullptr;
  size_t n_ = 0;
};

template <class T>
std::vector<T> read_file(const std::string &path, size_t n);

template <class T>
void write_file(const std::string &path, const T *data, size_t n) {
  FILE *f = fopen(path.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot create " + path);
  const bool ok = fwrite(data, sizeof(T), n, f) == n;
  if (fclose(f) != 0 || !ok) throw std::runtime_error("cannot write " + path);
}

struct RangeField {
  std::string name;
  HugeArray<uint32_t> col, sorted, perm;
};

struct StringField {
  std::string name;
  std::vector<std::string> dict;
  std::vector<int64_t> ptr;
  HugeArray<uint32_t> col, pos;
};

struct StopTable {
  std::vector<int64_t> bounds;
  std::vector<float> errors;
  int bucket(int rank) const {
    int b = 0;
    const int nb = (int) errors.size();
    while (b + 1 < nb && (int64_t) rank >= bounds[b + 1]) b++;
    return b;
  }
};

StopTable load_stop_table(const std::string &path, float alpha);

struct PlanRow {
  double vmax = 0;
  uint32_t partitions = 0;     // 0 = exhaustive
  uint32_t pool = 0;
  float alpha = 0;
  int table = -1;
};

class CentroidGraph {
 public:
  void build(const float *points, int n, int dim);
  void nearest(const float *query, int count, std::vector<int> &out) const;

 private:
  using Item = std::pair<float, int>;
  const float *point(int i) const { return points_ + (size_t) i * dim_; }
  float dist(const float *q, int i) const;
  std::vector<int> &links(int node, int level);
  const std::vector<int> &links(int node, int level) const;
  int greedy(const float *q, int entry, int level) const;
  void search_level(const float *q, int entry, float entry_dist, int ef, int level, std::vector<Item> &result) const;
  void select_links(std::vector<Item> &candidates, int keep, std::vector<int> &out) const;
  void connect(int node, int level, std::vector<int> &neighbors);

  int n_ = 0, dim_ = 0;
  int links_per_level_ = 16, links_level0_ = 32, build_ef_ = 128;
  const float *points_ = nullptr;
  std::vector<std::vector<int>> level0_;
  std::vector<std::vector<std::vector<int>>> upper_;
  int entry_ = -1, top_level_ = 0;
  mutable uint32_t rng_ = 2463534242u;
};

float squared_l2(const float *a, const float *b, int dim);

// The nearest centroid of a vector, and its PQ code.
int encode_vector(const float *v, const float *coarse, int nlist, int dim, const float *book, int m, int dsub, int ksub, uint8_t *code);

// Vectors added since the last merge, per partition.
struct DeltaPartition {
  std::vector<uint8_t> codes;
  std::vector<int32_t> ids;
  std::vector<int64_t> label_start{0};
  std::vector<int32_t> labels;             // ascending per vector
};

struct DeltaRef {
  int32_t partition, slot;
  bool operator<(const DeltaRef &o) const { return partition != o.partition ? partition < o.partition : slot < o.slot; }
  bool operator==(const DeltaRef &o) const { return partition == o.partition && slot == o.slot; }
};

// The packed arrays stay as they are between merges. pending is true from the first update to the next merge.
struct Updates {
  bool pending = false;
  std::vector<DeltaPartition> delta;
  std::vector<std::vector<DeltaRef>> label_delta;
  std::vector<DeltaRef> delta_of_id;       // from first_delta_id on
  int64_t delta_size = 0;
  int32_t first_delta_id = 0;
  std::vector<int64_t> live;               // current members of each label
  int64_t live_total = 0;
  std::vector<uint8_t> removed;            // one bit per id
  bool any_removed = false;
  std::vector<uint32_t> position_of;       // packed vectors, built by the first removal
  std::vector<int64_t> position_label_start;
  std::vector<int32_t> position_labels;
};

struct Index::Impl {
  std::string prefix;
  CountOptions count;
  int32_t n = 0, dim = 0, nlist = 0, m = 0, dsub = 0, ksub = 0;
  int32_t next_id = 0;
  int vector_bytes = 0, element_bytes = 0, per_block = 1;
  std::vector<float> coarse, book;
  HugeArray<uint8_t> codes;
  std::vector<int64_t> lptr;
  std::vector<int32_t> lids;
  int nlabels = 0;
  std::vector<int64_t> siptr;
  HugeArray<uint32_t> sidx;
  std::vector<RangeField> ranges;
  std::vector<StringField> strings;
  CentroidGraph graph;
  int vectors_fd = -1;
  long vectors_size = 0;
  std::vector<PlanRow> plan;
  std::vector<StopTable> tables;
  Updates updates;
  ~Impl();
};

}  // namespace kuroann
