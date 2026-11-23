#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <exception>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "kuroann/index.h"

namespace py = pybind11;
using kuroann::Index;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

}  // namespace

PYBIND11_MODULE(kuroann, m) {
  m.doc() = "KuroANN: filtered vector search on SSDs that sets how much each query reads from its match count";

  m.def(
      "build",
      [](const std::string &base, const std::string &prefix, const std::string &labels,
         const std::map<std::string, std::string> &ranges, const std::map<std::string, std::string> &strings, uint32_t pq,
         uint32_t partitions, uint32_t threads) {
        std::vector<kuroann::AttributeSource> attributes;
        for (const auto &kv : ranges) attributes.push_back({kuroann::AttributeSource::Kind::kRange, kv.first, kv.second});
        for (const auto &kv : strings) attributes.push_back({kuroann::AttributeSource::Kind::kString, kv.first, kv.second});
        kuroann::BuildOptions o;
        o.pq_pieces = pq;
        o.partitions = partitions;
        o.threads = threads;
        py::gil_scoped_release release;
        kuroann::build_index(base, labels, attributes, prefix, o);
      },
      py::arg("base"), py::arg("prefix"), py::arg("labels") = "", py::arg("ranges") = std::map<std::string, std::string>(),
      py::arg("strings") = std::map<std::string, std::string>(), py::arg("pq") = 64, py::arg("partitions") = 0, py::arg("threads") = 0,
      "Build an index from a .fbin file, an optional label matrix, and optional attribute files (field name to path)");

  py::class_<Index>(m, "Index")
      .def(py::init([](const std::string &prefix, uint32_t local_partitions, double local_ratio) {
             kuroann::CountOptions count;
             count.local_partitions = local_partitions;
             count.local_ratio = local_ratio;
             py::gil_scoped_release release;
             return std::make_unique<Index>(prefix, count);
           }),
           py::arg("prefix"), py::arg("local_partitions") = 0, py::arg("local_ratio") = 2.0)
      .def("load_plan", &Index::load_plan, py::arg("path"))
      .def(
          "search",
          [](const Index &self, FloatArray queries, const std::vector<std::string> &filters, uint32_t k, uint32_t threads) {
            if (queries.ndim() != 2 || (uint32_t) queries.shape(1) != self.dimension()) {
              throw std::invalid_argument("queries must be (n, dim) float32");
            }
            const size_t nq = (size_t) queries.shape(0);
            if (!filters.empty() && filters.size() != nq) throw std::invalid_argument("expected one filter per query");
            std::vector<kuroann::Predicate> predicates;
            predicates.reserve(nq);
            for (size_t i = 0; i < nq; i++) predicates.emplace_back(self, filters.empty() ? "ALL" : filters[i]);
            py::array_t<int64_t> ids({(py::ssize_t) nq, (py::ssize_t) k});
            py::array_t<float> distances({(py::ssize_t) nq, (py::ssize_t) k});
            py::array_t<uint32_t> reads((py::ssize_t) nq);
            const float *q = queries.data();
            const uint32_t dim = self.dimension();
            int64_t *out_ids = ids.mutable_data();
            float *out_distances = distances.mutable_data();
            uint32_t *out_reads = reads.mutable_data();
            {
              py::gil_scoped_release release;
              if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
              threads = (uint32_t) std::min<size_t>(threads, std::max<size_t>(1, nq));
              std::atomic<size_t> next{0};
              std::vector<std::thread> pool;
              std::vector<std::exception_ptr> errors(threads);
              for (uint32_t t = 0; t < threads; t++) {
                pool.emplace_back([&, t]() {
                  try {
                    kuroann::Searcher searcher(self, k, std::max(k, self.plan_max_pool()));
                    std::vector<kuroann::Hit> hits(k);
                    kuroann::SearchStats stats;
                    for (size_t i; (i = next.fetch_add(1)) < nq;) {
                      const uint32_t got = searcher.search(q + i * dim, predicates[i], hits.data(), &stats);
                      for (uint32_t h = 0; h < k; h++) {
                        out_ids[i * k + h] = h < got ? (int64_t) hits[h].id : -1;
                        out_distances[i * k + h] = h < got ? hits[h].distance : 0.0f;
                      }
                      out_reads[i] = stats.reads;
                    }
                    if (searcher.io_errors()) throw std::runtime_error("the SSD returned short reads");
                  } catch (...) {
                    errors[t] = std::current_exception();
                  }
                });
              }
              for (auto &th : pool) th.join();
              for (const auto &e : errors) {
                if (e) std::rethrow_exception(e);
              }
            }
            return py::make_tuple(ids, distances, reads);
          },
          py::arg("queries"), py::arg("filters") = std::vector<std::string>(), py::arg("k") = 10, py::arg("threads") = 0,
          "Follow the plan. Returns (ids, squared L2 distances, reads per query), nearest first; a query with fewer than k matches pads ids with -1")
      .def(
          "insert",
          [](Index &self, FloatArray vectors, const std::vector<std::vector<int32_t>> &labels) {
            if (vectors.ndim() != 2 || (uint32_t) vectors.shape(1) != self.dimension()) {
              throw std::invalid_argument("vectors must be (n, dim) float32");
            }
            const float *v = vectors.data();
            const size_t n = (size_t) vectors.shape(0);
            py::gil_scoped_release release;
            return self.insert(v, n, labels);
          },
          py::arg("vectors"), py::arg("labels") = std::vector<std::vector<int32_t>>(), "Add vectors; returns their ids")
      .def("remove", &Index::remove, py::arg("id"), "Remove one vector; False when the id is absent")
      .def("merge", &Index::merge, py::call_guard<py::gil_scoped_release>())
      .def("save", &Index::save, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("size", &Index::size)
      .def_property_readonly("dimension", &Index::dimension)
      .def_property_readonly("partitions", &Index::partitions)
      .def_property_readonly("pq_pieces", &Index::pq_pieces)
      .def_property_readonly("labels", &Index::labels);
}
