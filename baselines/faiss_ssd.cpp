// Faiss-SSD: Faiss's IndexIVFPQ over the partitions and PQ codes of a KuroANN index, reranking from its file of raw vectors.
//
// The coarse quantizer is a Faiss HNSW over the centroids. The inverted lists hold KuroANN's codes and ids in list order, without
// residual encoding. A query evaluates its filter into a bitmap over vector ids, inside the timed window, and takes one of the two
// routes of Faiss's filtered-search recipe:
//   nprobe > 0   IndexIVFPQ::search with SearchParametersIVF {nprobe, IDSelectorBitmap}
//   nprobe = 0   a scan of the codes of all matching vectors with Faiss's PQ distance table
// The route returns its top L candidates by code. All of them are read from the SSD, one 4 KiB read each and submitted at once,
// and the 10 nearest by exact distance are the answer.
//
// The process loads the index once and then reads commands from stdin:
//   run QUERIES.fbin FILTERS.expr DUMP_PREFIX NPROBE:L[,NPROBE:L...]
// It prints one RES line per setting, then END, and writes DUMP_PREFIX_np<NPROBE>_L<L>.bin (i32 queries, i32 10, u32 ids).
//
// A filter is a prefix expression: L label | R field lo hi | AND k e1..ek | OR k e1..ek | NOT e.
//
// usage: faiss_ssd --index P --labels L.spmat [--range NAME=FILE.u32] [--threads T]
#include <fcntl.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/invlists/InvertedLists.h>
#include <liburing.h>
#include <omp.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using faiss::idx_t;

namespace {

// Faiss's HNSW does not repair connectivity, and with fewer links or a narrower construction search some centroids are unreachable.
const int QUANTIZER_LINKS = 64;
const int QUANTIZER_CONSTRUCTION_EF = 256;
const size_t BLOCK = 4096;
const int K = 10;
const size_t MAX_DEPTH = 8;

[[noreturn]] void die(const std::string &message) {
  std::fprintf(stderr, "faiss_ssd: %s\n", message.c_str());
  std::exit(2);
}

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

template <class T>
std::vector<T> read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) die("cannot open " + path);
  const size_t bytes = (size_t) f.tellg();
  if (bytes % sizeof(T)) die(path + ": size is not a multiple of the element");
  std::vector<T> v(bytes / sizeof(T));
  f.seekg(0);
  if (!f.read((char *) v.data(), (std::streamsize) bytes)) die("cannot read " + path);
  return v;
}

struct Node {
  char kind = 0;
  int32_t label = 0;
  uint32_t lo = 0, hi = 0;
  std::vector<std::unique_ptr<Node>> kids;
};

std::unique_ptr<Node> parse_tokens(const std::vector<std::string> &t, size_t &i, const std::string &range_name) {
  if (i >= t.size()) die("a filter ends early");
  auto n = std::make_unique<Node>();
  const std::string s = t[i++];
  if (s == "NOT") {
    n->kind = 'N';
    n->kids.push_back(parse_tokens(t, i, range_name));
  } else if (s == "AND" || s == "OR") {
    n->kind = s == "AND" ? 'A' : 'O';
    const int k = std::stoi(t.at(i++));
    if (k < 1) die("AND or OR without operands");
    for (int j = 0; j < k; j++) n->kids.push_back(parse_tokens(t, i, range_name));
  } else if (s == "L") {
    n->kind = 'L';
    n->label = std::stoi(t.at(i++));
  } else if (s == "R") {
    n->kind = 'R';
    if (t.at(i++) != range_name) die("unknown range field " + t[i - 1]);
    n->lo = (uint32_t) std::stoul(t.at(i++));
    n->hi = (uint32_t) std::stoul(t.at(i++));
  } else {
    die("unsupported token " + s);
  }
  return n;
}

// The filter data, by vector id: the ascending ids of each label, and all ids sorted by the range field.
struct FilterData {
  int64_t n = 0;
  std::vector<int64_t> label_start;
  std::vector<uint32_t> label_ids;
  std::string range_name;
  std::vector<uint32_t> range_ids, range_values;

  void load_labels(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) die("cannot open " + path);
    int64_t head[3];
    f.read((char *) head, sizeof(head));
    if (head[0] != n) die(path + ": rows differ from the index");
    if (head[1] < 0 || head[2] < 0) die(path + ": bad header");
    std::vector<int64_t> row(n + 1);
    std::vector<int32_t> labels(head[2]);
    f.read((char *) row.data(), (std::streamsize) (row.size() * 8));
    f.read((char *) labels.data(), (std::streamsize) (labels.size() * 4));
    if (!f) die("cannot read " + path);
    if (row[0] != 0 || row[n] != head[2]) die(path + ": row starts do not cover the entries");
    for (int64_t v = 0; v < n; v++) {
      if (row[v + 1] < row[v]) die(path + ": row starts decrease");
    }
    label_start.assign(head[1] + 1, 0);
    for (int32_t l : labels) {
      if (l < 0 || l >= head[1]) die(path + ": label out of range");
      label_start[l + 1]++;
    }
    for (int64_t l = 0; l < head[1]; l++) label_start[l + 1] += label_start[l];
    label_ids.resize(labels.size());
    std::vector<int64_t> fill(label_start.begin(), label_start.end() - 1);
    for (int64_t v = 0; v < n; v++) {
      for (int64_t j = row[v]; j < row[v + 1]; j++) label_ids[fill[labels[j]]++] = (uint32_t) v;
    }
  }

  void load_range(const std::string &name, const std::string &path) {
    range_name = name;
    const auto values = read_file<uint32_t>(path);
    if ((int64_t) values.size() != n) die(path + ": rows differ from the index");
    range_ids.resize(n);
    for (int64_t i = 0; i < n; i++) range_ids[i] = (uint32_t) i;
    std::stable_sort(range_ids.begin(), range_ids.end(), [&](uint32_t a, uint32_t b) { return values[a] < values[b]; });
    range_values.resize(n);
    for (int64_t i = 0; i < n; i++) range_values[i] = values[range_ids[i]];
  }
};

// One per search thread. It works on 64-bit words of a bitmap over vector ids.
struct Evaluator {
  const FilterData *data;
  size_t words;
  std::vector<std::vector<uint64_t>> scratch;

  explicit Evaluator(const FilterData *d) : data(d), words(((size_t) d->n + 63) / 64), scratch(MAX_DEPTH, std::vector<uint64_t>(words)) {}

  void eval(const Node *t, uint64_t *out, size_t depth = 0) {
    const auto set = [out](uint32_t v) { out[v >> 6] |= 1ull << (v & 63); };
    switch (t->kind) {
      case 'L':
        std::memset(out, 0, words * 8);
        if (t->label < 0 || t->label + 1 >= (int64_t) data->label_start.size()) die("label out of range");
        for (int64_t j = data->label_start[t->label]; j < data->label_start[t->label + 1]; j++) set(data->label_ids[j]);
        break;
      case 'R': {
        std::memset(out, 0, words * 8);
        const auto begin = data->range_values.begin();
        const auto first = std::lower_bound(begin, data->range_values.end(), t->lo) - begin;
        const auto last = std::lower_bound(begin, data->range_values.end(), t->hi) - begin;
        for (auto j = first; j < last; j++) set(data->range_ids[j]);
        break;
      }
      case 'N':
        eval(t->kids[0].get(), out, depth + 1);
        for (size_t w = 0; w < words; w++) out[w] = ~out[w];
        if (data->n % 64) out[words - 1] &= (1ull << (data->n % 64)) - 1;
        break;
      default: {
        if (depth >= MAX_DEPTH) die("a filter is nested too deeply");
        eval(t->kids[0].get(), out, depth + 1);
        std::vector<uint64_t> &other = scratch[depth];
        for (size_t j = 1; j < t->kids.size(); j++) {
          eval(t->kids[j].get(), other.data(), depth + 1);
          if (t->kind == 'A') {
            for (size_t w = 0; w < words; w++) out[w] &= other[w];
          } else {
            for (size_t w = 0; w < words; w++) out[w] |= other[w];
          }
        }
      }
    }
  }
};

struct Server {
  std::unique_ptr<faiss::IndexIVFPQ> index;
  std::vector<const uint8_t *> list_codes;
  FilterData filter;
  int fd = -1, per_block = 1, threads = 1, nlist = 0, pieces = 0, codes_per_piece = 0;
  idx_t n = 0;
  size_t dim = 0, vector_bytes = 0;

  void load(const std::string &prefix) {
    const auto meta = read_file<int32_t>(prefix + ".meta");
    if (meta.size() < 6) die("bad meta file");
    const int element_bytes = meta.size() > 6 ? meta[6] : 4;
    if (element_bytes != 4) die("this baseline reads float32 vectors only");
    if (meta.size() > 7 && meta[7] != meta[0]) die("the index has removed vectors");
    n = meta[0], dim = (size_t) meta[1], nlist = meta[2], pieces = meta[3], codes_per_piece = meta[5];
    vector_bytes = dim * 4;
    per_block = (int) (BLOCK / vector_bytes);

    auto coarse = read_file<float>(prefix + ".coarse");
    auto book = read_file<float>(prefix + ".book");
    const auto list_start = read_file<int64_t>(prefix + ".lptr");
    const auto ids = read_file<int32_t>(prefix + ".lids");
    if ((int) list_start.size() != nlist + 1 || list_start[nlist] != n || (idx_t) ids.size() != n) die("bad list files");

    // One thread builds the quantizer graph, because a parallel build gives a different graph in every run.
    omp_set_num_threads(1);
    auto *quantizer = new faiss::IndexHNSWFlat((int) dim, QUANTIZER_LINKS);
    quantizer->hnsw.efConstruction = QUANTIZER_CONSTRUCTION_EF;
    quantizer->add(nlist, coarse.data());
    index = std::make_unique<faiss::IndexIVFPQ>(quantizer, dim, nlist, pieces, 8);
    index->own_fields = true;
    index->by_residual = false;
    if (index->pq.centroids.size() != book.size()) die("codebook size differs from Faiss's");
    std::memcpy(index->pq.centroids.data(), book.data(), book.size() * sizeof(float));
    index->pq.sync_transposed_centroids();
    index->is_trained = true;

    std::ifstream codes(prefix + ".codes", std::ios::binary);
    if (!codes) die("cannot open the codes");
    std::vector<uint8_t> buffer;
    std::vector<idx_t> list_ids;
    for (int l = 0; l < nlist; l++) {
      const idx_t m = list_start[l + 1] - list_start[l];
      if (m == 0) continue;
      buffer.resize((size_t) m * pieces);
      list_ids.assign(ids.begin() + list_start[l], ids.begin() + list_start[l + 1]);
      if (!codes.read((char *) buffer.data(), (std::streamsize) buffer.size())) die("cannot read the codes");
      index->invlists->add_entries(l, m, list_ids.data(), buffer.data());
    }
    index->ntotal = n;
    index->make_direct_map(true);
    list_codes.resize(nlist);
    for (int l = 0; l < nlist; l++) list_codes[l] = index->invlists->get_codes(l);

    fd = open((prefix + ".vectors").c_str(), O_RDONLY | O_DIRECT);
    if (fd < 0) die("cannot open the vectors");
  }

  void run(const std::string &query_path, const std::string &filter_path, const std::string &dump, const std::string &settings_text) {
    const auto header = read_file<uint32_t>(query_path);
    if (header.size() < 2 || header[1] != dim) die(query_path + ": dimension differs from the index");
    const size_t nq = header[0];
    const float *queries = (const float *) (header.data() + 2);

    std::vector<std::unique_ptr<Node>> filters;
    std::ifstream lines(filter_path);
    if (!lines) die("cannot open " + filter_path);
    for (std::string line; std::getline(lines, line);) {
      if (line.empty()) continue;
      std::stringstream ss(line);
      std::vector<std::string> tokens;
      for (std::string w; ss >> w;) tokens.push_back(w);
      size_t i = 0;
      filters.push_back(parse_tokens(tokens, i, filter.range_name));
      if (i != tokens.size()) die("trailing tokens in a filter");
    }
    if (filters.size() != nq) die("one filter per query expected");

    std::vector<std::pair<int, int>> settings;
    std::stringstream ss(settings_text);
    bool any_scan = false;
    for (std::string token; std::getline(ss, token, ',');) {
      const auto colon = token.find(':');
      settings.emplace_back(std::stoi(token.substr(0, colon)), std::stoi(token.substr(colon + 1)));
      if (settings.back().first < 0 || settings.back().first > nlist || settings.back().second < K) die("bad setting " + token);
      any_scan |= settings.back().first == 0;
    }

    // The largest matching set sizes the scan's list in each thread's setup, outside the timed window.
    size_t scan_capacity = 0;
    if (any_scan) {
      Evaluator e(&filter);
      std::vector<uint64_t> bits(e.words);
      for (size_t q = 0; q < nq; q++) {
        e.eval(filters[q].get(), bits.data());
        size_t count = 0;
        for (uint64_t x : bits) count += (size_t) __builtin_popcountll(x);
        scan_capacity = std::max(scan_capacity, count);
      }
    }
    for (const auto &setting : settings) measure(queries, nq, filters, dump, setting.first, setting.second, scan_capacity);
    std::printf("END\n");
    std::fflush(stdout);
  }

  void measure(const float *queries, size_t nq, const std::vector<std::unique_ptr<Node>> &filters, const std::string &dump, int nprobe, int pool,
               size_t scan_capacity) {
    std::vector<double> latency(nq);
    std::vector<uint32_t> results(nq * K, 0xFFFFFFFFu);
    std::vector<uint32_t> reads(nq, 0);
    std::vector<double> finish(threads, 0);
    std::atomic<long> io_errors{0};
    std::atomic<size_t> next{0};
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; t++) {
      workers.emplace_back([&, t]() {
        omp_set_num_threads(1);
        Evaluator evaluator(&filter);
        std::vector<uint64_t> bits(evaluator.words, 0);
        faiss::IDSelectorBitmap selector((size_t) n, reinterpret_cast<const uint8_t *>(bits.data()));
        faiss::SearchParametersHNSW quantizer_params;
        quantizer_params.efSearch = std::min(nlist, std::max((int) (nprobe * 2.5) + 32, 128));
        faiss::SearchParametersIVF params;
        params.nprobe = nprobe;
        params.sel = &selector;
        params.quantizer_params = &quantizer_params;

        std::vector<float> distances(pool);
        std::vector<idx_t> candidates(pool);
        io_uring ring;
        if (io_uring_queue_init(std::max(64, pool + 8), &ring, 0) < 0) die("cannot set up io_uring");
        std::vector<uint8_t *> buffers(pool);
        for (auto &b : buffers) {
          if (posix_memalign((void **) &b, BLOCK, BLOCK)) die("out of memory");
        }
        std::vector<std::pair<float, uint32_t>> exact;
        exact.reserve(pool);
        std::vector<float> table(nprobe == 0 ? (size_t) pieces * codes_per_piece : 0);
        std::vector<std::pair<float, idx_t>> scanned;
        if (nprobe == 0) scanned.reserve(scan_capacity);

        ready++;
        while (!go.load(std::memory_order_acquire)) {
        }
        for (size_t q; (q = next.fetch_add(1)) < nq;) {
          const double start = now();
          const float *x = queries + q * dim;
          evaluator.eval(filters[q].get(), bits.data());
          if (nprobe == 0) {
            index->pq.compute_distance_table(x, table.data());
            scanned.clear();
            for (size_t w = 0; w < bits.size(); w++) {
              for (uint64_t b = bits[w]; b; b &= b - 1) {
                const idx_t id = (idx_t) (w * 64 + (size_t) __builtin_ctzll(b));
                const idx_t place = index->direct_map.get(id);
                const uint8_t *code = list_codes[faiss::lo_listno(place)] + (size_t) faiss::lo_offset(place) * pieces;
                float s = 0;
                for (int m = 0; m < pieces; m++) s += table[(size_t) m * codes_per_piece + code[m]];
                scanned.emplace_back(s, id);
              }
            }
            const size_t top = std::min<size_t>(pool, scanned.size());
            std::partial_sort(scanned.begin(), scanned.begin() + top, scanned.end());
            for (int j = 0; j < pool; j++) candidates[j] = (size_t) j < top ? scanned[j].second : -1;
          } else {
            index->search(1, x, pool, distances.data(), candidates.data(), &params);
          }

          int count = 0;
          for (int j = 0; j < pool; j++) {
            if (candidates[j] < 0) continue;
            io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            io_uring_prep_read(sqe, fd, buffers[count], BLOCK, (off_t) (candidates[j] / per_block) * BLOCK);
            sqe->user_data = (uint64_t) count;
            candidates[count++] = candidates[j];
          }
          if (count) {
            io_uring_submit(&ring);
            io_uring_cqe *cqe;
            if (io_uring_wait_cqe_nr(&ring, &cqe, (unsigned) count)) die("io_uring wait failed");
            unsigned head, done = 0;
            io_uring_for_each_cqe(&ring, head, cqe) {
              if (cqe->res != (int) BLOCK) io_errors++;
              done++;
            }
            io_uring_cq_advance(&ring, done);
            if ((int) done != count) die("missing completions");
          }
          exact.clear();
          for (int j = 0; j < count; j++) {
            const float *y = (const float *) (buffers[j] + (size_t) (candidates[j] % per_block) * vector_bytes);
            float s = 0;
            for (size_t e = 0; e < dim; e++) {
              const float u = x[e] - y[e];
              s += u * u;
            }
            exact.emplace_back(s, (uint32_t) candidates[j]);
          }
          const int top = std::min<int>(K, (int) exact.size());
          std::partial_sort(exact.begin(), exact.begin() + top, exact.end());
          for (int j = 0; j < top; j++) results[q * K + j] = exact[j].second;
          reads[q] = (uint32_t) count;
          latency[q] = now() - start;
        }
        finish[t] = now();
        io_uring_queue_exit(&ring);
        for (uint8_t *b : buffers) free(b);
      });
    }
    while (ready.load() < threads) {
    }
    const double start = now();
    go.store(true, std::memory_order_release);
    for (auto &w : workers) w.join();
    const double wall = *std::max_element(finish.begin(), finish.end()) - start;

    std::vector<double> sorted(latency);
    std::sort(sorted.begin(), sorted.end());
    double mean = 0, read_sum = 0;
    for (size_t i = 0; i < nq; i++) mean += latency[i], read_sum += reads[i];
    const auto percentile = [&](double p) { return sorted[std::min(nq - 1, (size_t) std::ceil(p * (double) nq) - 1)] * 1e6; };
    std::printf("RES nprobe=%d pool=%d queries=%zu qps=%.2f mean_us=%.1f p50_us=%.1f p99_us=%.1f reads=%.2f io_errors=%ld\n", nprobe, pool, nq, (double) nq / wall,
                mean / (double) nq * 1e6, percentile(0.5), percentile(0.99), read_sum / (double) nq, io_errors.load());
    if (dump != "-") {
      const std::string path = dump + "_np" + std::to_string(nprobe) + "_L" + std::to_string(pool) + ".bin";
      std::ofstream f(path, std::ios::binary);
      const int32_t head[2] = {(int32_t) nq, K};
      f.write((const char *) head, sizeof(head));
      f.write((const char *) results.data(), (std::streamsize) (results.size() * 4));
      if (!f) die("cannot write " + path);
    }
    std::fflush(stdout);
  }
};

}  // namespace

int main(int argc, char **argv) {
  std::map<std::string, std::string> args;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strncmp(argv[i], "--", 2)) die(std::string("bad argument ") + argv[i]);
    args[argv[i] + 2] = argv[i + 1];
  }
  if (!args.count("index") || !args.count("labels")) die("usage: faiss_ssd --index P --labels L.spmat [--range NAME=FILE.u32] [--threads T]");
  Server server;
  server.threads = args.count("threads") ? std::stoi(args["threads"]) : (int) std::thread::hardware_concurrency();
  server.load(args["index"]);
  server.filter.n = server.n;
  server.filter.load_labels(args["labels"]);
  if (args.count("range")) {
    const auto eq = args["range"].find('=');
    if (eq == std::string::npos) die("--range takes NAME=FILE");
    server.filter.load_range(args["range"].substr(0, eq), args["range"].substr(eq + 1));
  }
  std::printf("READY vectors=%lld partitions=%d\n", (long long) server.n, server.nlist);
  std::fflush(stdout);

  for (std::string line; std::getline(std::cin, line);) {
    std::stringstream ss(line);
    std::vector<std::string> words;
    for (std::string w; ss >> w;) words.push_back(w);
    if (words.empty()) continue;
    if (words[0] == "quit") break;
    if (words[0] != "run" || words.size() != 5) die("bad command: " + line);
    server.run(words[1], words[2], words[3], words[4]);
  }
  return 0;
}
