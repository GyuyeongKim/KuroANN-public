#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace kuroann {

// Files of an index with prefix P:
//   P.meta     int32: vectors, dimension, partitions, PQ pieces, piece length, codes per piece, bytes per element, next id
//   P.coarse   partition centroids, float32
//   P.book     PQ codebooks, float32
//   P.codes    one PQ code per vector, in partition order
//   P.lptr     int64: first position of each partition
//   P.lids     int32: vector id at each position
//   P.siptr    int64: first entry of each label's posting
//   P.sidx     uint32: positions per label, ascending
//   P.vectors  raw vectors by id, whole vectors per 4 KiB block, no header
//   P.attr.*   attribute fields, one value per position
//
// The distance is squared L2. Normalize the vectors first for cosine similarity.

struct BuildOptions {
  uint32_t partitions = 0;     // 0: the power of two nearest to one partition per 2,500 vectors
  uint32_t pq_pieces = 64;     // PQ bytes per vector; must divide the dimension
  uint32_t train_size = 400000;
  uint32_t iterations = 25;
  uint64_t seed = 1;
  uint32_t threads = 0;        // 0: all hardware threads
};

struct AttributeSource {
  enum class Kind { kRange, kString };
  Kind kind = Kind::kRange;
  std::string name;
  std::string path;            // kRange: uint32 per vector; kString: one value per line
};

// base: u32 count, u32 dimension, then float32 or uint8 values.
// labels: optional sparse matrix (i64 rows, columns, entries; i64 row starts; i32 labels; f32 values).
void build_index(const std::string &base, const std::string &labels, const std::vector<AttributeSource> &attributes,
                 const std::string &prefix, const BuildOptions &options);

enum class Route {
  kAuto,        // exhaustive when the matches are fewer than the vectors of the probed partitions
  kExhaustive,  // rank every match
  kPartition,   // rank the matches in the nearest partitions
};

// A search setting. The plan picks one per query; a caller can also fix one.
struct Setting {
  Route route = Route::kAuto;
  uint32_t partitions = 64;    // partitions to probe
  uint32_t pool = 128;         // candidates kept by PQ estimate, read from the SSD at most
  float alpha = 0;             // miss rate of the stop table; 0 reads the whole pool
  uint32_t read_step = 32;     // reads issued between two stop checks
  std::string table;           // stop table; required when alpha > 0
};

struct SearchStats {
  double count = 0;            // match count used to pick the row
  int plan_row = -1;
  bool exhaustive = false;
  uint32_t partitions = 0;     // partitions probed
  uint32_t reads = 0;          // 4 KiB reads
  uint32_t scanned = 0;        // PQ codes ranked
  bool local = false;          // the count came from the nearest partitions
};

struct Hit {
  uint32_t id;
  float distance;
};

class Predicate;
class Index;

// One thread's search state: its io_uring ring, read buffers, and scratch lists.
class Searcher {
 public:
  Searcher(const Index &index, uint32_t k, uint32_t max_pool);
  ~Searcher();
  Searcher(const Searcher &) = delete;
  Searcher &operator=(const Searcher &) = delete;

  // Follows the index's plan. Returns the hits written to out, at most k, by (distance, id).
  uint32_t search(const float *query, const Predicate &predicate, Hit *out, SearchStats *stats = nullptr);
  // Follows one setting instead of the plan.
  uint32_t search(const float *query, const Predicate &predicate, const Setting &setting, Hit *out,
                  SearchStats *stats = nullptr);

  uint64_t io_errors() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// A filter as a prefix expression:
//   ALL | L <label> | R <field> <lo> <hi> (lo <= value < hi) | S <field> eq|in|like <value>
//   AND <n> e1..en | OR <n> e1..en | NOT e
class Predicate {
 public:
  Predicate(const Index &index, const std::string &expression);
  ~Predicate();
  Predicate(Predicate &&) noexcept;
  Predicate &operator=(Predicate &&) noexcept;

 private:
  friend class Searcher;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct CountOptions {
  uint32_t samples = 16384;       // evenly spaced positions tested when an exact count is costly; 0 always counts exactly
  uint64_t min_work = 262144;     // sample only when an exact count would touch at least this many entries
  double z = 3.0;                 // width of the confidence interval on the sampled share
  // A label or an OR of labels can also be counted in the partitions nearest to the query. When the lower end of the
  // confidence interval on its share there, scaled to the index, exceeds local_ratio times its count, the plan row follows
  // that share, or the lower end when the two ends of the interval fall in different rows. 0 partitions: off.
  uint32_t local_partitions = 0;
  double local_ratio = 2.0;
};

class Index {
 public:
  explicit Index(const std::string &prefix, const CountOptions &count = CountOptions());
  ~Index();
  Index(Index &&) noexcept;
  Index &operator=(Index &&) noexcept;

  // Plan file: one row per line, "vmax=<count> np=<partitions, 0 = exhaustive> L=<pool> alpha=<a> [table=<path>]",
  // increasing vmax. A relative table path is taken from the plan's directory.
  void load_plan(const std::string &path);
  uint32_t plan_max_pool() const;

  // Updates must not overlap searches on the same index. They serve label filters, so an index with attribute fields
  // refuses them. New vectors and removals take effect at once and need no new plan. The raw values of new vectors go to
  // the vector file at once; their codes, labels, and the removals stay in memory until a merge and a save.

  // Each vector takes the next id. labels: one list per vector, or empty when the vectors have no labels.
  std::vector<uint32_t> insert(const float *vectors, size_t count, const std::vector<std::vector<int32_t>> &labels = {});
  // False when the id is absent.
  bool remove(uint32_t id);
  // Moves the new vectors into the packed arrays in the order of a fresh build, and drops the removed ones.
  void merge();
  // Writes the packed arrays to the index files. Merge first.
  void save();

  uint32_t size() const;       // vectors that a search can return
  uint32_t dimension() const;
  uint32_t partitions() const;
  uint32_t pq_pieces() const;
  uint32_t labels() const;

  struct Impl;

 private:
  friend class Searcher;
  friend class Predicate;
  std::unique_ptr<Impl> impl_;
};

}  // namespace kuroann
