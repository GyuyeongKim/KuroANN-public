#include <fcntl.h>
#include <omp.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "internal.h"

namespace kuroann {

namespace {

constexpr int kCodesPerPiece = 256;
constexpr size_t kPqTrainPerCode = 256;
constexpr int kBlock = 4096;

class MappedFile {
 public:
  explicit MappedFile(const std::string &path) {
    size_ = (size_t) file_size(path);
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    data_ = (const uint8_t *) mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (data_ == MAP_FAILED) throw std::runtime_error("cannot map " + path);
  }
  ~MappedFile() { munmap((void *) data_, size_); }
  const uint8_t *data() const { return data_; }
  size_t size() const { return size_; }

 private:
  const uint8_t *data_ = nullptr;
  size_t size_ = 0;
};

// Lloyd's algorithm. An empty centroid restarts near the largest cluster.
void kmeans(const float *x, size_t n, int dim, int k, int iterations, uint64_t seed, float *centroids) {
  if (n < (size_t) k) throw std::invalid_argument("fewer training vectors than centroids");
  std::mt19937_64 rng(seed);
  std::vector<size_t> ids(n);
  std::iota(ids.begin(), ids.end(), (size_t) 0);
  std::shuffle(ids.begin(), ids.end(), rng);
  for (int c = 0; c < k; c++) std::memcpy(centroids + (size_t) c * dim, x + ids[c] * dim, sizeof(float) * dim);
  std::vector<int> assign(n);
  std::vector<double> sums((size_t) k * dim);
  std::vector<size_t> counts(k);
  for (int it = 0; it < iterations; it++) {
#pragma omp parallel for schedule(dynamic, 1024)
    for (size_t i = 0; i < n; i++) {
      const float *v = x + i * dim;
      int best = 0;
      float bd = 3.4e38f;
      for (int c = 0; c < k; c++) {
        const float d = squared_l2(v, centroids + (size_t) c * dim, dim);
        if (d < bd) {
          bd = d;
          best = c;
        }
      }
      assign[i] = best;
    }
    std::fill(sums.begin(), sums.end(), 0.0);
    std::fill(counts.begin(), counts.end(), (size_t) 0);
    for (size_t i = 0; i < n; i++) {
      const int c = assign[i];
      counts[c]++;
      const float *v = x + i * dim;
      double *s = sums.data() + (size_t) c * dim;
      for (int d = 0; d < dim; d++) s[d] += v[d];
    }
    for (int c = 0; c < k; c++) {
      if (counts[c] == 0) continue;
      float *o = centroids + (size_t) c * dim;
      const double *s = sums.data() + (size_t) c * dim;
      for (int d = 0; d < dim; d++) o[d] = (float) (s[d] / counts[c]);
    }
    for (int c = 0; c < k; c++) {
      if (counts[c] != 0) continue;
      const int big = (int) (std::max_element(counts.begin(), counts.end()) - counts.begin());
      float *o = centroids + (size_t) c * dim;
      float *b = centroids + (size_t) big * dim;
      for (int d = 0; d < dim; d++) {
        const float e = (d % 2 == 0 ? 1.0f : -1.0f) * 1e-6f * (std::fabs(b[d]) + 1e-6f);
        o[d] = b[d] + e;
        b[d] = b[d] - e;
      }
      counts[c] = counts[big] / 2;
      counts[big] -= counts[c];
    }
  }
}

void load_vector(const uint8_t *raw, int dim, int element_bytes, float *out) {
  if (element_bytes == 4) {
    std::memcpy(out, raw, sizeof(float) * dim);
  } else {
    for (int d = 0; d < dim; d++) out[d] = (float) raw[d];
  }
}

void build_range_field(const std::string &name, const std::string &path, const std::vector<int32_t> &lids,
                       const std::string &out) {
  const size_t n = lids.size();
  MappedFile in(path);
  if (in.size() != n * 4) throw std::runtime_error(path + ": expected one uint32 per vector");
  const uint32_t *v = (const uint32_t *) in.data();
  std::vector<uint32_t> col(n);
  std::vector<uint64_t> keys(n);
#pragma omp parallel for
  for (size_t p = 0; p < n; p++) {
    col[p] = v[lids[p]];
    keys[p] = ((uint64_t) col[p] << 32) | (uint64_t) p;
  }
  std::sort(keys.begin(), keys.end());
  std::vector<uint32_t> sorted(n), perm(n);
  for (size_t i = 0; i < n; i++) {
    sorted[i] = (uint32_t) (keys[i] >> 32);
    perm[i] = (uint32_t) keys[i];
  }
  write_file(out + "." + name + ".col", col.data(), n);
  write_file(out + "." + name + ".sval", sorted.data(), n);
  write_file(out + "." + name + ".perm", perm.data(), n);
}

void build_string_field(const std::string &name, const std::string &path, const std::vector<int32_t> &lids,
                        const std::string &out) {
  const size_t n = lids.size();
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::map<std::string, uint32_t> ids;
  std::vector<uint32_t> by_id;
  by_id.reserve(n);
  for (std::string line; std::getline(in, line);) {
    if (line.find_first_of(" \t") != std::string::npos) throw std::runtime_error(path + ": values cannot hold spaces");
    ids.emplace(line, 0);
    by_id.push_back(0);
  }
  if (by_id.size() != n) throw std::runtime_error(path + ": expected one line per vector");
  std::vector<std::string> dict;
  dict.reserve(ids.size());
  for (auto &kv : ids) {
    kv.second = (uint32_t) dict.size();
    dict.push_back(kv.first);
  }
  in.clear();
  in.seekg(0);
  size_t row = 0;
  for (std::string line; std::getline(in, line);) by_id[row++] = ids[line];
  std::vector<uint32_t> col(n);
  std::vector<int64_t> ptr(dict.size() + 1, 0);
  for (size_t p = 0; p < n; p++) {
    col[p] = by_id[lids[p]];
    ptr[col[p] + 1]++;
  }
  for (size_t d = 0; d < dict.size(); d++) ptr[d + 1] += ptr[d];
  std::vector<int64_t> next(ptr.begin(), ptr.end() - 1);
  std::vector<uint32_t> pos(n);
  for (size_t p = 0; p < n; p++) pos[next[col[p]]++] = (uint32_t) p;
  write_file(out + "." + name + ".col", col.data(), n);
  write_file(out + "." + name + ".ptr", ptr.data(), ptr.size());
  write_file(out + "." + name + ".pos", pos.data(), n);
  std::ofstream d(out + "." + name + ".dict");
  for (const std::string &s : dict) d << s << "\n";
  d.close();
  if (!d) throw std::runtime_error("cannot write " + out + "." + name + ".dict");
}

}  // namespace

int encode_vector(const float *v, const float *coarse, int nlist, int dim, const float *book, int m, int dsub, int ksub, uint8_t *code) {
  int best = 0;
  float bd = 3.4e38f;
  for (int c = 0; c < nlist; c++) {
    const float d = squared_l2(v, coarse + (size_t) c * dim, dim);
    if (d < bd) {
      bd = d;
      best = c;
    }
  }
  for (int p = 0; p < m; p++) {
    const float *xs = v + p * dsub;
    const float *bk = book + (size_t) p * ksub * dsub;
    int bc = 0;
    float bs = 3.4e38f;
    for (int c = 0; c < ksub; c++) {
      const float *word = bk + (size_t) c * dsub;
      float s = 0;
      for (int d = 0; d < dsub; d++) {
        const float t = xs[d] - word[d];
        s += t * t;
      }
      if (s < bs) {
        bs = s;
        bc = c;
      }
    }
    code[p] = (uint8_t) bc;
  }
  return best;
}

void build_index(const std::string &base_path, const std::string &labels_path,
                 const std::vector<AttributeSource> &attributes, const std::string &prefix, const BuildOptions &opt) {
  if (opt.threads) omp_set_num_threads((int) opt.threads);
  // A rebuild into the same prefix must not keep the attribute fields of the previous index.
  std::remove((prefix + ".attr.fields").c_str());
  MappedFile base(base_path);
  if (base.size() < 8) throw std::runtime_error(base_path + ": too short");
  const uint32_t *header = (const uint32_t *) base.data();
  const size_t n = header[0];
  const int dim = (int) header[1];
  const size_t payload = base.size() - 8;
  int element_bytes;
  if (payload == n * dim * 4) element_bytes = 4;
  else if (payload == n * dim) element_bytes = 1;
  else throw std::runtime_error(base_path + ": size does not match its header");
  const int vector_bytes = dim * element_bytes;
  if (vector_bytes > kBlock) throw std::invalid_argument("a vector does not fit in a 4 KiB block");
  const uint8_t *raw = base.data() + 8;

  int nlist = (int) opt.partitions;
  if (nlist == 0) {
    const double ideal = std::max(1.0, (double) n / 2500.0);
    nlist = std::min(32768, 1 << (int) std::lround(std::log2(ideal)));
  }
  const int m = (int) opt.pq_pieces;
  if (m <= 0 || dim % m != 0) throw std::invalid_argument("the PQ pieces must divide the dimension");
  const int dsub = dim / m;

  const size_t train = std::min<size_t>(n, opt.train_size);
  std::mt19937_64 rng(opt.seed);
  std::vector<uint32_t> sample(n);
  std::iota(sample.begin(), sample.end(), 0u);
  for (size_t i = 0; i < train; i++) {
    const size_t j = i + (size_t) (rng() % (n - i));
    std::swap(sample[i], sample[j]);
  }
  sample.resize(train);
  std::sort(sample.begin(), sample.end());
  std::vector<float> tx(train * dim);
#pragma omp parallel for
  for (size_t i = 0; i < train; i++) load_vector(raw + (size_t) sample[i] * vector_bytes, dim, element_bytes, tx.data() + i * dim);

  std::vector<float> coarse((size_t) nlist * dim);
  kmeans(tx.data(), train, dim, nlist, (int) opt.iterations, opt.seed + 1, coarse.data());

  std::vector<float> book((size_t) m * kCodesPerPiece * dsub);
  const size_t pq_train = std::min(train, kPqTrainPerCode * kCodesPerPiece);
#pragma omp parallel for schedule(dynamic, 1)
  for (int p = 0; p < m; p++) {
    std::vector<float> sub(pq_train * dsub);
    const size_t stride = std::max<size_t>(1, train / pq_train);
    for (size_t i = 0; i < pq_train; i++) std::memcpy(&sub[i * dsub], &tx[i * stride * dim + (size_t) p * dsub], sizeof(float) * dsub);
    kmeans(sub.data(), pq_train, dsub, kCodesPerPiece, (int) opt.iterations, opt.seed + 2 + p, &book[(size_t) p * kCodesPerPiece * dsub]);
  }
  std::vector<float>().swap(tx);

  std::vector<int32_t> cell(n);
  std::vector<uint8_t> by_id(n * m);
#pragma omp parallel
  {
    std::vector<float> v(dim);
#pragma omp for schedule(dynamic, 4096)
    for (size_t i = 0; i < n; i++) {
      load_vector(raw + i * vector_bytes, dim, element_bytes, v.data());
      cell[i] = encode_vector(v.data(), coarse.data(), nlist, dim, book.data(), m, dsub, kCodesPerPiece, &by_id[i * m]);
    }
  }

  std::vector<int64_t> lptr((size_t) nlist + 1, 0);
  for (size_t i = 0; i < n; i++) lptr[cell[i] + 1]++;
  for (int c = 0; c < nlist; c++) lptr[c + 1] += lptr[c];
  std::vector<int64_t> next(lptr.begin(), lptr.end() - 1);
  std::vector<int32_t> lids(n);
  std::vector<uint8_t> codes(n * m);
  for (size_t i = 0; i < n; i++) {
    const int64_t p = next[cell[i]]++;
    lids[p] = (int32_t) i;
    std::memcpy(&codes[(size_t) p * m], &by_id[i * m], m);
  }
  std::vector<uint8_t>().swap(by_id);
  std::vector<int32_t>().swap(cell);

  const int32_t meta[8] = {(int32_t) n, dim, nlist, m, dsub, kCodesPerPiece, element_bytes, (int32_t) n};
  write_file(prefix + ".meta", meta, 8);
  write_file(prefix + ".coarse", coarse.data(), coarse.size());
  write_file(prefix + ".book", book.data(), book.size());
  write_file(prefix + ".codes", codes.data(), codes.size());
  write_file(prefix + ".lptr", lptr.data(), lptr.size());
  write_file(prefix + ".lids", lids.data(), lids.size());
  std::vector<uint8_t>().swap(codes);

  std::vector<int64_t> siptr(1, 0);
  std::vector<uint32_t> sidx;
  if (!labels_path.empty()) {
    MappedFile sp(labels_path);
    if (sp.size() < 24) throw std::runtime_error(labels_path + ": too short");
    const int64_t *h = (const int64_t *) sp.data();
    const int64_t rows = h[0], columns = h[1], entries = h[2];
    if ((size_t) rows != n) throw std::runtime_error(labels_path + ": expected one row per vector");
    if (columns < 0 || columns > INT32_MAX || entries < 0) throw std::runtime_error(labels_path + ": bad header");
    if (sp.size() < 24 + ((size_t) rows + 1) * 8 + (size_t) entries * 4) throw std::runtime_error(labels_path + ": shorter than its header says");
    const int64_t *row_start = h + 3;
    const int32_t *label = (const int32_t *) (row_start + rows + 1);
    if (row_start[0] != 0 || row_start[rows] != entries) throw std::runtime_error(labels_path + ": row starts do not cover the entries");
    for (int64_t v = 0; v < rows; v++) {
      if (row_start[v + 1] < row_start[v]) throw std::runtime_error(labels_path + ": row starts decrease");
    }
    // A label listed twice in one row counts once, so that a vector appears once in the label's posting.
    std::vector<int64_t> seen((size_t) columns, -1);
    siptr.assign((size_t) columns + 1, 0);
    for (int64_t v = 0; v < rows; v++) {
      for (int64_t e = row_start[v]; e < row_start[v + 1]; e++) {
        const int32_t l = label[e];
        if (l < 0 || l >= columns) throw std::runtime_error(labels_path + ": label out of range");
        if (seen[l] == v) continue;
        seen[l] = v;
        siptr[l + 1]++;
      }
    }
    for (int64_t l = 0; l < columns; l++) siptr[l + 1] += siptr[l];
    sidx.resize((size_t) siptr[columns]);
    std::vector<int64_t> fill(siptr.begin(), siptr.end() - 1);
    std::fill(seen.begin(), seen.end(), -1);
    for (size_t p = 0; p < n; p++) {
      const int32_t v = lids[p];
      for (int64_t e = row_start[v]; e < row_start[v + 1]; e++) {
        const int32_t l = label[e];
        if (seen[l] == v) continue;
        seen[l] = v;
        sidx[fill[l]++] = (uint32_t) p;
      }
    }
  }
  write_file(prefix + ".siptr", siptr.data(), siptr.size());
  write_file(prefix + ".sidx", sidx.data(), sidx.size());

  if (!attributes.empty()) {
    std::ofstream fields(prefix + ".attr.fields");
    for (const AttributeSource &a : attributes) {
      if (a.kind == AttributeSource::Kind::kRange) {
        build_range_field(a.name, a.path, lids, prefix + ".attr");
        fields << "range " << a.name << "\n";
      } else {
        build_string_field(a.name, a.path, lids, prefix + ".attr");
        fields << "string " << a.name << "\n";
      }
    }
    fields.close();
    if (!fields) throw std::runtime_error("cannot write " + prefix + ".attr.fields");
  }

  const int per = kBlock / vector_bytes;
  FILE *f = fopen((prefix + ".vectors").c_str(), "wb");
  if (!f) throw std::runtime_error("cannot create " + prefix + ".vectors");
  std::vector<uint8_t> buffer((size_t) kBlock * 4096);
  const size_t blocks = (n + per - 1) / per;
  bool ok = true;
  for (size_t b0 = 0; b0 < blocks && ok; b0 += 4096) {
    const size_t c = std::min<size_t>(4096, blocks - b0);
    std::fill(buffer.begin(), buffer.begin() + c * kBlock, (uint8_t) 0);
    for (size_t b = 0; b < c; b++) {
      for (int j = 0; j < per; j++) {
        const size_t id = (b0 + b) * per + j;
        if (id < n) std::memcpy(&buffer[b * kBlock + (size_t) j * vector_bytes], raw + id * vector_bytes, vector_bytes);
      }
    }
    ok = fwrite(buffer.data(), kBlock, c, f) == c;
  }
  if (fclose(f) != 0 || !ok) throw std::runtime_error("cannot write " + prefix + ".vectors");
}

}  // namespace kuroann
