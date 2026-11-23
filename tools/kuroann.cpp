// kuroann: build an index, run queries with a plan or a fixed setting, or apply updates.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "kuroann/index.h"

using namespace kuroann;

namespace {

// The id that --out writes after the last result of a query with fewer than k results.
constexpr uint32_t kNoResult = 0xFFFFFFFFu;

struct Args {
  std::map<std::string, std::string> kv;
  std::vector<std::pair<std::string, std::string>> all;
  std::string get(const std::string &k, const std::string &def = "") const {
    auto it = kv.find(k);
    return it == kv.end() ? def : it->second;
  }
  std::string need(const std::string &k) const {
    auto it = kv.find(k);
    if (it == kv.end()) throw std::invalid_argument("missing --" + k);
    return it->second;
  }
  uint32_t u32(const std::string &k, uint32_t def) const {
    auto it = kv.find(k);
    return it == kv.end() ? def : (uint32_t) std::stoul(it->second);
  }
};

Args parse(int argc, char **argv) {
  Args a;
  for (int i = 2; i < argc; i += 2) {
    const std::string k = argv[i];
    if (k.rfind("--", 0) != 0 || i + 1 >= argc) throw std::invalid_argument("bad argument " + k);
    a.kv[k.substr(2)] = argv[i + 1];
    a.all.push_back({k.substr(2), argv[i + 1]});
  }
  return a;
}

struct Queries {
  uint32_t n = 0, dim = 0;
  std::vector<float> x;
};

Queries read_queries(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  Queries q;
  f.read((char *) &q.n, 4);
  f.read((char *) &q.dim, 4);
  q.x.resize((size_t) q.n * q.dim);
  f.seekg(0, std::ios::end);
  const size_t payload = (size_t) f.tellg() - 8;
  f.seekg(8);
  if (payload == q.x.size() * 4) {
    f.read((char *) q.x.data(), (std::streamsize) payload);
  } else if (payload == q.x.size()) {
    std::vector<uint8_t> raw(payload);
    f.read((char *) raw.data(), (std::streamsize) payload);
    for (size_t i = 0; i < payload; i++) q.x[i] = (float) raw[i];
  } else {
    throw std::runtime_error(path + ": size does not match its header");
  }
  return q;
}

std::vector<Predicate> read_filters(const Index &index, const std::string &path, uint32_t n) {
  std::vector<Predicate> out;
  out.reserve(n);
  if (path.empty()) {
    for (uint32_t i = 0; i < n; i++) out.emplace_back(index, "ALL");
    return out;
  }
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path);
  for (std::string line; std::getline(f, line);) {
    if (line.empty() || line[0] == '#') continue;
    out.emplace_back(index, line);
  }
  if (out.size() != n) throw std::runtime_error(path + ": expected one expression per query");
  return out;
}

struct Truth {
  uint32_t n = 0, k = 0;
  std::vector<uint32_t> ids;
  std::vector<float> dist;
};

Truth read_truth(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  Truth t;
  f.read((char *) &t.n, 4);
  f.read((char *) &t.k, 4);
  t.ids.resize((size_t) t.n * t.k);
  t.dist.resize(t.ids.size());
  f.read((char *) t.ids.data(), (std::streamsize) (t.ids.size() * 4));
  f.read((char *) t.dist.data(), (std::streamsize) (t.dist.size() * 4));
  if (!f) throw std::runtime_error(path + ": truncated");
  return t;
}

// A returned vector counts when its distance does not exceed the k-th true distance, so tied vectors are interchangeable.
double tie_recall(const Truth &t, uint32_t k, const std::vector<Hit> &hits, const std::vector<uint32_t> &counts) {
  double scale = 0;
  for (uint32_t i = 0; i < t.n; i++) scale += std::fabs(t.dist[(size_t) i * t.k + k - 1]);
  scale /= std::max<uint32_t>(1, t.n);
  double sum = 0;
  for (uint32_t i = 0; i < t.n; i++) {
    const double limit = (double) t.dist[(size_t) i * t.k + k - 1] * (1.0 + 1e-5) + 1e-6 * scale;
    uint32_t hit = 0;
    for (uint32_t x = 0; x < counts[i]; x++) {
      if ((double) hits[(size_t) i * k + x].distance <= limit) hit++;
    }
    sum += (double) std::min(hit, k) / k;
  }
  return sum / std::max<uint32_t>(1, t.n);
}

int cmd_build(const Args &a) {
  BuildOptions o;
  o.partitions = a.u32("partitions", 0);
  o.pq_pieces = a.u32("pq", 64);
  o.train_size = a.u32("train", o.train_size);
  o.iterations = a.u32("iterations", o.iterations);
  o.seed = a.u32("seed", 1);
  o.threads = a.u32("threads", 0);
  std::vector<AttributeSource> attrs;
  for (const auto &kv : a.all) {
    if (kv.first != "range" && kv.first != "string") continue;
    const size_t e = kv.second.find('=');
    if (e == std::string::npos) throw std::invalid_argument("--" + kv.first + " takes name=path");
    AttributeSource s;
    s.kind = kv.first == "range" ? AttributeSource::Kind::kRange : AttributeSource::Kind::kString;
    s.name = kv.second.substr(0, e);
    s.path = kv.second.substr(e + 1);
    attrs.push_back(s);
  }
  build_index(a.need("base"), a.get("labels"), attrs, a.need("index"), o);
  std::printf("built %s\n", a.need("index").c_str());
  return 0;
}

std::vector<std::pair<uint32_t, uint32_t>> parse_settings(const std::string &s) {
  std::vector<std::pair<uint32_t, uint32_t>> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, ',');) {
    const size_t c = item.find(':');
    if (c == std::string::npos) throw std::invalid_argument("--settings takes partitions:pool,...");
    out.push_back({(uint32_t) std::stoul(item.substr(0, c)), (uint32_t) std::stoul(item.substr(c + 1))});
  }
  return out;
}

int cmd_search(const Args &a) {
  CountOptions count;
  count.samples = a.u32("count-samples", count.samples);
  count.min_work = a.u32("count-min-work", (uint32_t) count.min_work);
  count.local_partitions = a.u32("local-partitions", 0);
  const std::string ratio = a.get("local-ratio", "2");
  size_t used = 0;
  try {
    count.local_ratio = std::stod(ratio, &used);
  } catch (const std::exception &) {
    used = 0;
  }
  if (used == 0 || used != ratio.size()) throw std::invalid_argument("--local-ratio takes a number");
  Index index(a.need("index"), count);
  const Queries q = read_queries(a.need("queries"));
  if (q.dim != index.dimension()) throw std::invalid_argument("the queries have another dimension");
  const std::vector<Predicate> filters = read_filters(index, a.get("filters"), q.n);
  const uint32_t k = a.u32("k", 10), threads = a.u32("threads", std::thread::hardware_concurrency());
  const uint32_t rounds = a.u32("rounds", 1);
  const std::string plan = a.get("plan");
  if (plan.empty() && count.local_partitions > 0) throw std::invalid_argument("--local-partitions works with --plan");
  std::vector<std::pair<uint32_t, uint32_t>> settings;
  uint32_t max_pool = k;
  if (!plan.empty()) {
    index.load_plan(plan);
    max_pool = std::max(max_pool, index.plan_max_pool());
    settings.push_back({0, 0});
  } else {
    settings = parse_settings(a.need("settings"));
    for (const auto &s : settings) max_pool = std::max(max_pool, s.second);
  }
  const std::string route = a.get("route", "auto");
  const float alpha = std::stof(a.get("alpha", "0"));
  const std::string table = a.get("table");
  const uint32_t step = a.u32("read-step", 32);
  Truth truth;
  const std::string truth_path = a.get("groundtruth");
  if (!truth_path.empty()) {
    truth = read_truth(truth_path);
    if (truth.n != q.n || truth.k < k) throw std::invalid_argument("the ground truth does not match the queries");
  }

  std::vector<Hit> hits((size_t) q.n * k);
  std::vector<uint32_t> counts(q.n);
  std::vector<double> latency(q.n);
  std::vector<SearchStats> stats(q.n);
  for (const auto &s : settings) {
    Setting setting;
    setting.route = route == "exhaustive" ? Route::kExhaustive : route == "partition" ? Route::kPartition : Route::kAuto;
    setting.partitions = s.first;
    setting.pool = s.second;
    setting.alpha = alpha;
    setting.table = table;
    setting.read_step = step;
    // One thread and one Searcher per worker for all rounds, so that round 0 also warms up each worker's scratch lists.
    std::mutex mutex;
    std::condition_variable changed;
    int started = -1;
    bool stop = false;
    std::exception_ptr failure;
    uint64_t parked = 0, finished = 0;
    std::vector<std::chrono::steady_clock::time_point> finish(threads);
    std::vector<uint64_t> errors(threads, 0);
    std::vector<std::thread> pool;
    for (uint32_t t = 0; t < threads; t++) {
      pool.emplace_back([&, t]() {
        // A worker that fails keeps taking part in the rounds, so that the main thread can stop all of them and report the error.
        std::unique_ptr<Searcher> searcher;
        std::exception_ptr error;
        try {
          searcher = std::make_unique<Searcher>(index, k, max_pool);
        } catch (...) {
          error = std::current_exception();
        }
        for (uint32_t round = 0; round <= rounds; round++) {
          {
            std::unique_lock<std::mutex> lock(mutex);
            parked++;
            changed.notify_all();
            changed.wait(lock, [&]() { return stop || started >= (int) round; });
            if (stop) return;
          }
          if (!error) {
            try {
              for (uint32_t i = t; i < q.n; i += threads) {
                const auto t0 = std::chrono::steady_clock::now();
                const float *v = q.x.data() + (size_t) i * q.dim;
                Hit *out = &hits[(size_t) i * k];
                counts[i] = plan.empty() ? searcher->search(v, filters[i], setting, out, &stats[i])
                                         : searcher->search(v, filters[i], out, &stats[i]);
                latency[i] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                for (uint32_t h = counts[i]; h < k; h++) out[h] = Hit{kNoResult, std::numeric_limits<float>::infinity()};
              }
            } catch (...) {
              error = std::current_exception();
            }
          }
          finish[t] = std::chrono::steady_clock::now();
          errors[t] = searcher ? searcher->io_errors() : 0;
          std::lock_guard<std::mutex> lock(mutex);
          if (error && !failure) failure = error;
          finished++;
          changed.notify_all();
        }
      });
    }
    for (uint32_t round = 0; round <= rounds; round++) {     // round 0 warms up and is not reported
      std::chrono::steady_clock::time_point start;
      std::exception_ptr error;
      {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [&]() { return parked == (uint64_t) threads * (round + 1); });
        start = std::chrono::steady_clock::now();
        started = (int) round;
        changed.notify_all();
        changed.wait(lock, [&]() { return finished == (uint64_t) threads * (round + 1); });
        error = failure;
      }
      const bool short_reads = std::any_of(errors.begin(), errors.end(), [](uint64_t e) { return e != 0; });
      if (error || short_reads) {
        {
          std::lock_guard<std::mutex> lock(mutex);
          stop = true;
          changed.notify_all();
        }
        for (auto &th : pool) th.join();
        if (error) std::rethrow_exception(error);
        throw std::runtime_error("the SSD returned short reads");
      }
      if (round == 0) continue;
      const double seconds = std::chrono::duration<double>(*std::max_element(finish.begin(), finish.end()) - start).count();
      std::vector<double> sorted(latency);
      std::sort(sorted.begin(), sorted.end());
      double mean = 0, reads = 0, scanned = 0;
      for (uint32_t i = 0; i < q.n; i++) {
        mean += latency[i];
        reads += stats[i].reads;
        scanned += stats[i].scanned;
      }
      const double den = std::max<uint32_t>(1, q.n);
      std::printf("partitions=%u pool=%u round=%u qps=%.1f mean_us=%.1f p50_us=%.1f p99_us=%.1f reads=%.2f scanned=%.1f", s.first,
                  s.second, round, q.n / seconds, mean / den, sorted[q.n / 2],
                  sorted[std::min<size_t>(q.n - 1, (size_t) (q.n * 0.99))], reads / den, scanned / den);
      if (!truth_path.empty()) std::printf(" recall=%.4f", tie_recall(truth, k, hits, counts));
      std::printf("\n");
      std::fflush(stdout);
    }
    for (auto &th : pool) th.join();
  }
  const std::string out = a.get("out");
  if (!out.empty()) {
    std::ofstream f(out, std::ios::binary);
    const int32_t header[2] = {(int32_t) q.n, (int32_t) k};
    f.write((const char *) header, 8);
    for (const Hit &h : hits) f.write((const char *) &h.id, 4);
    for (const Hit &h : hits) f.write((const char *) &h.distance, 4);
    f.close();
    if (!f) throw std::runtime_error("cannot write " + out);
  }
  const std::string per_query = a.get("query-stats");
  if (!per_query.empty()) {
    std::ofstream f(per_query);
    f << "query\tlatency_us\treads\tcount\trow\texhaustive\tpartitions\tscanned\tlocal\n";
    for (uint32_t i = 0; i < q.n; i++) {
      f << i << '\t' << latency[i] << '\t' << stats[i].reads << '\t' << stats[i].count << '\t' << stats[i].plan_row << '\t'
        << (int) stats[i].exhaustive << '\t' << stats[i].partitions << '\t' << stats[i].scanned << '\t'
        << (int) stats[i].local << '\n';
    }
    f.close();
    if (!f) throw std::runtime_error("cannot write " + per_query);
  }
  return 0;
}

std::vector<std::vector<int32_t>> read_label_rows(const std::string &path, uint32_t rows) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  int64_t h[3];
  f.read((char *) h, 24);
  if (!f || h[0] != (int64_t) rows) throw std::runtime_error(path + ": expected one row per vector");
  if (h[1] < 0 || h[2] < 0) throw std::runtime_error(path + ": bad header");
  std::vector<int64_t> start((size_t) rows + 1);
  std::vector<int32_t> labels((size_t) h[2]);
  f.read((char *) start.data(), (std::streamsize) (start.size() * 8));
  f.read((char *) labels.data(), (std::streamsize) (labels.size() * 4));
  if (!f) throw std::runtime_error(path + ": truncated");
  if (start[0] != 0 || start[rows] != h[2]) throw std::runtime_error(path + ": row starts do not cover the entries");
  for (uint32_t r = 0; r < rows; r++) {
    if (start[r + 1] < start[r]) throw std::runtime_error(path + ": row starts decrease");
  }
  std::vector<std::vector<int32_t>> out(rows);
  for (uint32_t r = 0; r < rows; r++) out[r].assign(labels.begin() + start[r], labels.begin() + start[r + 1]);
  return out;
}

int cmd_update(const Args &a) {
  Index index(a.need("index"));
  const std::string added = a.get("insert"), gone = a.get("remove");
  if (added.empty() && gone.empty()) throw std::invalid_argument("--insert or --remove is required");
  if (!added.empty()) {
    const Queries v = read_queries(added);
    if (v.dim != index.dimension()) throw std::invalid_argument("the new vectors have another dimension");
    std::vector<std::vector<int32_t>> labels;
    const std::string label_path = a.get("insert-labels");
    if (!label_path.empty()) labels = read_label_rows(label_path, v.n);
    const std::vector<uint32_t> ids = index.insert(v.x.data(), v.n, labels);
    if (!ids.empty()) std::printf("inserted %zu vectors with ids %u to %u\n", ids.size(), ids.front(), ids.back());
  }
  if (!gone.empty()) {
    std::ifstream f(gone, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open " + gone);
    std::vector<uint32_t> ids((size_t) f.tellg() / 4);
    f.seekg(0);
    f.read((char *) ids.data(), (std::streamsize) (ids.size() * 4));
    if (!f) throw std::runtime_error("cannot read " + gone);
    size_t removed = 0;
    for (uint32_t id : ids) removed += index.remove(id);
    std::printf("removed %zu of %zu ids\n", removed, ids.size());
  }
  index.merge();
  index.save();
  std::printf("%u vectors\n", index.size());
  return 0;
}

void usage() {
  std::fprintf(stderr,
               "kuroann build  --base B --index P [--labels S] [--range name=file] [--string name=file]\n"
               "               [--pq 64] [--partitions N] [--train N] [--iterations N] [--seed N] [--threads N]\n"
               "kuroann search --index P --queries Q [--filters F] [--k 10] [--threads N] [--rounds 1]\n"
               "               (--plan PLAN | --settings partitions:pool,... [--route auto|exhaustive|partition]\n"
               "                [--table T --alpha A] [--read-step 32])\n"
               "               [--groundtruth G] [--out R] [--query-stats S] [--count-samples 16384] [--count-min-work 262144]\n"
               "               [--local-partitions 0] [--local-ratio 2]\n"
               "kuroann update --index P [--insert V [--insert-labels S]] [--remove IDS]\n");
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  try {
    const std::string cmd = argv[1];
    const Args a = parse(argc, argv);
    if (cmd == "build") return cmd_build(a);
    if (cmd == "search") return cmd_search(a);
    if (cmd == "update") return cmd_update(a);
    usage();
    return 2;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "kuroann: %s\n", e.what());
    return 1;
  }
}
