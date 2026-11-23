#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "internal.h"

namespace kuroann {

long file_size(const std::string &path) {
  struct stat st;
  if (stat(path.c_str(), &st) != 0) throw std::runtime_error("cannot stat " + path);
  return (long) st.st_size;
}

template <class T>
T *HugeArray<T>::allocate(size_t n) {
  reset();
  const size_t bytes = n * sizeof(T);
  if (bytes == 0) return nullptr;
  void *m = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (m == MAP_FAILED) throw std::runtime_error("cannot map memory");
  madvise(m, bytes, MADV_HUGEPAGE);
  data_ = (T *) m;
  n_ = n;
  return data_;
}

template <class T>
void HugeArray<T>::load(const std::string &path, size_t n) {
  reset();
  if (file_size(path) != (long) (n * sizeof(T))) throw std::runtime_error(path + ": unexpected size");
  if (n == 0) return;
  T *m = allocate(n);
  FILE *f = fopen(path.c_str(), "rb");
  if (!f || fread(m, sizeof(T), n, f) != n) {
    if (f) fclose(f);
    reset();
    throw std::runtime_error("cannot read " + path);
  }
  fclose(f);
}

template class HugeArray<uint8_t>;
template class HugeArray<uint32_t>;

template <class T>
std::vector<T> read_file(const std::string &path, size_t n) {
  if (file_size(path) != (long) (n * sizeof(T))) throw std::runtime_error(path + ": unexpected size");
  std::vector<T> v(n);
  FILE *f = fopen(path.c_str(), "rb");
  if (!f || fread(v.data(), sizeof(T), n, f) != n) {
    if (f) fclose(f);
    throw std::runtime_error("cannot read " + path);
  }
  fclose(f);
  return v;
}

template std::vector<float> read_file<float>(const std::string &, size_t);
template std::vector<int32_t> read_file<int32_t>(const std::string &, size_t);
template std::vector<int64_t> read_file<int64_t>(const std::string &, size_t);

StopTable load_stop_table(const std::string &path, float alpha) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) throw std::runtime_error("cannot open stop table " + path);
  uint32_t h[4];
  bool ok = fread(h, 4, 4, f) == 4 && h[0] == 0x44334331u;
  if (!ok) {
    fclose(f);
    throw std::runtime_error(path + ": not a stop table");
  }
  const int nb = (int) h[1], na = (int) h[2];
  std::vector<int32_t> bounds(nb + 1);
  std::vector<float> alphas(na), q((size_t) na * nb);
  ok = fread(bounds.data(), 4, nb + 1, f) == (size_t) nb + 1 && fread(alphas.data(), 4, na, f) == (size_t) na &&
       fread(q.data(), 4, q.size(), f) == q.size();
  fclose(f);
  if (!ok) throw std::runtime_error(path + ": truncated");
  int ai = -1;
  for (int a = 0; a < na; a++) {
    if (std::fabs(alphas[a] - alpha) < 1e-7f) ai = a;
  }
  if (ai < 0) throw std::runtime_error(path + ": no column for alpha " + std::to_string(alpha));
  StopTable t;
  t.bounds.assign(bounds.begin(), bounds.end());
  t.errors.assign(q.begin() + (size_t) ai * nb, q.begin() + (size_t) (ai + 1) * nb);
  return t;
}

Index::Impl::~Impl() {
  if (vectors_fd >= 0) close(vectors_fd);
}

Index::Index(const std::string &prefix, const CountOptions &count) : impl_(new Impl) {
  Impl &x = *impl_;
  x.prefix = prefix;
  x.count = count;
  if (!(count.local_ratio > 0)) throw std::invalid_argument("local_ratio must be positive");
  if (!(count.z > 0)) throw std::invalid_argument("z must be positive");
  const long meta_bytes = file_size(prefix + ".meta");
  std::vector<int32_t> meta = read_file<int32_t>(prefix + ".meta", (size_t) meta_bytes / 4);
  if (meta.size() < 6) throw std::runtime_error(prefix + ".meta: too short");
  x.n = meta[0];
  x.dim = meta[1];
  x.nlist = meta[2];
  x.m = meta[3];
  x.dsub = meta[4];
  x.ksub = meta[5];
  x.element_bytes = meta.size() > 6 ? meta[6] : 4;
  x.next_id = meta.size() > 7 ? meta[7] : x.n;
  x.vector_bytes = x.dim * x.element_bytes;
  if (x.vector_bytes > 4096) throw std::runtime_error("a vector does not fit in a 4 KiB block");
  x.per_block = 4096 / x.vector_bytes;
  x.coarse = read_file<float>(prefix + ".coarse", (size_t) x.nlist * x.dim);
  x.book = read_file<float>(prefix + ".book", (size_t) x.m * x.ksub * x.dsub);
  x.codes.load(prefix + ".codes", (size_t) x.n * x.m);
  x.lptr = read_file<int64_t>(prefix + ".lptr", (size_t) x.nlist + 1);
  x.lids = read_file<int32_t>(prefix + ".lids", (size_t) x.n);
  x.nlabels = (int) (file_size(prefix + ".siptr") / 8 - 1);
  x.siptr = read_file<int64_t>(prefix + ".siptr", (size_t) x.nlabels + 1);
  x.sidx.load(prefix + ".sidx", (size_t) x.siptr[x.nlabels]);

  const std::string fields = prefix + ".attr.fields";
  std::ifstream in(fields);
  if (in) {
    std::string type, name;
    while (in >> type >> name) {
      const std::string b = prefix + ".attr." + name;
      if (type == "range") {
        x.ranges.emplace_back();
        RangeField &f = x.ranges.back();
        f.name = name;
        f.col.load(b + ".col", x.n);
        f.sorted.load(b + ".sval", x.n);
        f.perm.load(b + ".perm", x.n);
      } else if (type == "string") {
        x.strings.emplace_back();
        StringField &f = x.strings.back();
        f.name = name;
        std::ifstream d(b + ".dict");
        if (!d) throw std::runtime_error("cannot open " + b + ".dict");
        for (std::string line; std::getline(d, line);) f.dict.push_back(line);
        f.ptr = read_file<int64_t>(b + ".ptr", f.dict.size() + 1);
        f.col.load(b + ".col", x.n);
        f.pos.load(b + ".pos", x.n);
      } else {
        throw std::runtime_error(fields + ": unknown field type " + type);
      }
    }
  }

  const std::string vectors = prefix + ".vectors";
  x.vectors_fd = open(vectors.c_str(), O_RDONLY | O_DIRECT);
  if (x.vectors_fd < 0) throw std::runtime_error("cannot open " + vectors);
  x.vectors_size = file_size(vectors);
  if (x.vectors_size < (((long) x.next_id + x.per_block - 1) / x.per_block) * 4096L) {
    throw std::runtime_error(vectors + ": too short");
  }
  x.graph.build(x.coarse.data(), x.nlist, x.dim);
}

Index::~Index() = default;
Index::Index(Index &&) noexcept = default;
Index &Index::operator=(Index &&) noexcept = default;

void Index::load_plan(const std::string &path) {
  Impl &x = *impl_;
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open plan " + path);
  const size_t slash = path.rfind('/');
  const std::string dir = slash == std::string::npos ? "" : path.substr(0, slash + 1);
  std::vector<PlanRow> rows;
  std::vector<StopTable> tables;
  std::vector<std::pair<std::string, float>> keys;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    PlanRow r;
    r.vmax = -1;
    r.alpha = -1;
    std::string table, tok;
    while (ls >> tok) {
      const size_t e = tok.find('=');
      if (e == std::string::npos) throw std::runtime_error(path + ": bad token " + tok);
      const std::string k = tok.substr(0, e), v = tok.substr(e + 1);
      if (k == "vmax") r.vmax = std::stod(v);
      else if (k == "np") r.partitions = (uint32_t) std::stoul(v);
      else if (k == "L") r.pool = (uint32_t) std::stoul(v);
      else if (k == "alpha") r.alpha = std::stof(v);
      else if (k == "table") table = v;
      else if (k == "C") {
        if (std::stol(v) != 0) throw std::runtime_error(path + ": C > 0 is not supported");
      } else {
        throw std::runtime_error(path + ": unknown key " + k);
      }
    }
    if (r.vmax < 0 || r.pool == 0 || r.alpha < 0 || (r.alpha > 0 && table.empty())) {
      throw std::runtime_error(path + ": bad row");
    }
    if (!rows.empty() && r.vmax <= rows.back().vmax) throw std::runtime_error(path + ": vmax must increase");
    if (r.alpha > 0) {
      const std::string full = (!table.empty() && table[0] == '/') ? table : dir + table;
      for (size_t k = 0; k < keys.size(); k++) {
        if (keys[k].first == full && keys[k].second == r.alpha) r.table = (int) k;
      }
      if (r.table < 0) {
        tables.push_back(load_stop_table(full, r.alpha));
        keys.push_back({full, r.alpha});
        r.table = (int) keys.size() - 1;
      }
    }
    rows.push_back(r);
  }
  if (rows.empty()) throw std::runtime_error(path + ": empty plan");
  x.plan.swap(rows);
  x.tables.swap(tables);
}

uint32_t Index::plan_max_pool() const {
  uint32_t m = 0;
  for (const PlanRow &r : impl_->plan) m = std::max(m, r.pool);
  return m;
}

uint32_t Index::size() const { return (uint32_t) (impl_->updates.pending ? impl_->updates.live_total : impl_->n); }
uint32_t Index::dimension() const { return (uint32_t) impl_->dim; }
uint32_t Index::partitions() const { return (uint32_t) impl_->nlist; }
uint32_t Index::pq_pieces() const { return (uint32_t) impl_->m; }
uint32_t Index::labels() const { return (uint32_t) impl_->nlabels; }

}  // namespace kuroann
