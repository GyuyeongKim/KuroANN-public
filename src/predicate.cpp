#include "predicate.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace kuroann {

namespace {

bool like_match(const std::string &s, const std::string &p) {
  size_t i = 0, j = 0, star = std::string::npos, mark = 0;
  while (i < s.size()) {
    if (j < p.size() && (p[j] == '_' || p[j] == s[i])) {
      i++;
      j++;
    } else if (j < p.size() && p[j] == '%') {
      star = j++;
      mark = i;
    } else if (star != std::string::npos) {
      j = star + 1;
      i = ++mark;
    } else {
      return false;
    }
  }
  while (j < p.size() && p[j] == '%') j++;
  return j == p.size();
}

[[noreturn]] void bad(const std::string &what) { throw std::invalid_argument("filter expression: " + what); }

int parse_node(Tree &t, const Index::Impl &index, const std::vector<std::string> &tok, size_t &i, bool negate) {
  if (i >= tok.size()) bad("truncated");
  const std::string s = tok[i++];
  if (s == "NOT") return parse_node(t, index, tok, i, !negate);
  if (s == "AND" || s == "OR") {
    if (i >= tok.size()) bad("missing arity");
    const int k = std::atoi(tok[i++].c_str());
    if (k < 1) bad("arity " + std::to_string(k));
    const int id = (int) t.nodes.size();
    t.nodes.emplace_back();
    t.nodes[id].type = ((s == "AND") != negate) ? kAnd : kOr;
    for (int j = 0; j < k; j++) {
      const int c = parse_node(t, index, tok, i, negate);
      t.nodes[id].kids.push_back(c);
    }
    return id;
  }
  Node x;
  x.negated = negate;
  auto need = [&](size_t m) {
    if (i + m > tok.size()) bad(s + " needs " + std::to_string(m) + " arguments");
  };
  if (s == "ALL") {
    x.type = kAll;
  } else if (s == "L") {
    need(1);
    x.type = kLabel;
    x.label = std::atoi(tok[i++].c_str());
    if (x.label < 0 || x.label >= index.nlabels) bad("label " + std::to_string(x.label) + " out of range");
  } else if (s == "R") {
    need(3);
    x.type = kRangeLeaf;
    const std::string f = tok[i++];
    for (size_t k = 0; k < index.ranges.size(); k++) {
      if (index.ranges[k].name == f) x.field = (int) k;
    }
    if (x.field < 0) bad("no range field " + f);
    x.lo = (uint32_t) std::strtoul(tok[i].c_str(), nullptr, 10);
    x.hi = (uint32_t) std::strtoul(tok[i + 1].c_str(), nullptr, 10);
    i += 2;
  } else if (s == "S") {
    need(3);
    x.type = kStringLeaf;
    const std::string f = tok[i++];
    for (size_t k = 0; k < index.strings.size(); k++) {
      if (index.strings[k].name == f) x.field = (int) k;
    }
    if (x.field < 0) bad("no string field " + f);
    const std::string op = tok[i++];
    x.op = op == "eq" ? 0 : op == "in" ? 1 : op == "like" ? 2 : -1;
    if (x.op < 0) bad("string operator " + op);
    x.arg = tok[i++];
    if (x.op == 1) {
      size_t a = 0;
      for (;;) {
        size_t b = x.arg.find(',', a);
        if (b == std::string::npos) {
          x.args.push_back(x.arg.substr(a));
          break;
        }
        x.args.push_back(x.arg.substr(a, b - a));
        a = b + 1;
      }
    }
  } else {
    bad("unknown token " + s);
  }
  t.nodes.push_back(x);
  return (int) t.nodes.size() - 1;
}

// Candidates of a node in [lo, hi): ascending, unique, a superset of its members in the range.
void driver_list(EvalContext &c, int v, uint32_t lo, uint32_t hi, std::vector<uint32_t> &out, int depth) {
  const Index::Impl &ix = *c.index;
  const Node &x = c.tree->nodes[v];
  const uint32_t n = (uint32_t) ix.n;
  out.clear();
  if (x.type == kAnd) {
    int b = -1;
    for (int k : x.kids) {
      if (b < 0 || c.driver[k] < c.driver[b]) b = k;
    }
    driver_list(c, b, lo, hi, out, depth + 1);
    return;
  }
  if (x.type == kOr) {
    bool scan = false;
    for (int k : x.kids) {
      if (c.driver[k] >= (double) n) scan = true;
    }
    if (scan) {
      out.resize(hi - lo);
      std::iota(out.begin(), out.end(), lo);
      return;
    }
    std::vector<uint32_t> &a = c.pool[2 * depth];
    std::vector<uint32_t> &t = c.pool[2 * depth + 1];
    bool first = true;
    for (int k : x.kids) {
      driver_list(c, k, lo, hi, a, depth + 1);
      if (first) {
        out.swap(a);
        first = false;
        continue;
      }
      t.resize(out.size() + a.size());
      t.resize((size_t) (std::set_union(out.begin(), out.end(), a.begin(), a.end(), t.begin()) - t.begin()));
      out.swap(t);
    }
    return;
  }
  if (x.type == kAll) {
    if (!x.negated) {
      out.resize(hi - lo);
      std::iota(out.begin(), out.end(), lo);
    }
    return;
  }
  const bool full = (lo == 0 && hi == n);
  if (x.negated) {
    if (x.type == kLabel) {
      const uint32_t *s = ix.sidx.data() + ix.siptr[x.label];
      const uint32_t *e = ix.sidx.data() + ix.siptr[x.label + 1];
      c.cursor[v] = std::lower_bound(s, e, lo);
      c.end[v] = e;
    }
    for (uint32_t p = lo; p < hi; p++) {
      if (test(c, v, p)) out.push_back(p);
    }
    return;
  }
  if (x.type == kLabel) {
    const uint32_t *s = ix.sidx.data() + ix.siptr[x.label];
    const uint32_t *e = ix.sidx.data() + ix.siptr[x.label + 1];
    const uint32_t *b = full ? s : std::lower_bound(s, e, lo);
    const uint32_t *f = full ? e : std::lower_bound(b, e, hi);
    out.assign(b, f);
    return;
  }
  if (x.type == kRangeLeaf) {
    const RangeField &f = ix.ranges[x.field];
    if (x.lo >= x.hi) return;
    if (full && c.count[v] * 32 < (double) n) {
      const uint32_t *b = std::lower_bound(f.sorted.data(), f.sorted.data() + n, x.lo);
      const uint32_t *e = std::lower_bound(b, f.sorted.data() + n, x.hi);
      out.assign(f.perm.data() + (b - f.sorted.data()), f.perm.data() + (e - f.sorted.data()));
      std::sort(out.begin(), out.end());
      return;
    }
    for (uint32_t p = lo; p < hi; p++) {
      const uint32_t val = f.col[p];
      if (val >= x.lo && val < x.hi) out.push_back(p);
    }
    return;
  }
  const StringField &f = ix.strings[x.field];
  if (full && c.count[v] * 32 < (double) n) {
    for (uint32_t d : c.values[v]) out.insert(out.end(), f.pos.data() + f.ptr[d], f.pos.data() + f.ptr[d + 1]);
    std::sort(out.begin(), out.end());
    return;
  }
  const uint8_t *b = c.bits[v].data();
  for (uint32_t p = lo; p < hi; p++) {
    if (b[f.col[p]]) out.push_back(p);
  }
}

}  // namespace

Tree parse_expression(const Index::Impl &index, const std::string &expression, bool negate) {
  std::vector<std::string> tok;
  std::istringstream in(expression);
  for (std::string t; in >> t;) tok.push_back(t);
  if (tok.empty()) bad("empty");
  Tree t;
  size_t i = 0;
  t.root = parse_node(t, index, tok, i, negate);
  if (i != tok.size()) bad("trailing tokens");
  return t;
}

void prepare(EvalContext &c, const Index::Impl &ix, const Tree &t) {
  c.index = &ix;
  c.tree = &t;
  const size_t n = t.nodes.size();
  c.count.assign(n, 0);
  c.lower.assign(n, 0);
  c.upper.assign(n, 0);
  c.driver.assign(n, 0);
  c.cursor.assign(n, nullptr);
  c.end.assign(n, nullptr);
  if (c.bits.size() < n) {
    c.bits.resize(n);
    c.values.resize(n);
  }
  if (c.pool.size() < 2 * n + 4) c.pool.resize(2 * n + 4);
  // Bounds follow the current members. Driver sizes follow the packed arrays, which are what a driver lists.
  const Updates &updates = ix.updates;
  const double positions = (double) ix.n;
  const double total = updates.pending ? (double) updates.live_total : positions;
  for (int v = (int) n - 1; v >= 0; v--) {
    const Node &x = t.nodes[v];
    if (x.type == kAnd) {
      double s = 0, u = total, d = positions;
      for (int k : x.kids) {
        s += c.lower[k];
        u = std::min(u, c.upper[k]);
        d = std::min(d, c.driver[k]);
      }
      c.lower[v] = std::max(0.0, s - (double) (x.kids.size() - 1) * total);
      c.upper[v] = u;
      c.driver[v] = d;
      continue;
    }
    if (x.type == kOr) {
      double l = 0, s = 0, d = 0;
      for (int k : x.kids) {
        l = std::max(l, c.lower[k]);
        s += c.upper[k];
        d += c.driver[k];
      }
      c.lower[v] = l;
      c.upper[v] = std::min(total, s);
      c.driver[v] = std::min(positions, d);
      continue;
    }
    double m = 0;
    if (x.type == kAll) {
      m = positions;
    } else if (x.type == kLabel) {
      m = (double) (ix.siptr[x.label + 1] - ix.siptr[x.label]);
    } else if (x.type == kRangeLeaf) {
      const RangeField &f = ix.ranges[x.field];
      if (x.lo < x.hi) {
        const uint32_t *b = std::lower_bound(f.sorted.data(), f.sorted.data() + ix.n, x.lo);
        m = (double) (std::lower_bound(b, f.sorted.data() + ix.n, x.hi) - b);
      }
    } else {
      const StringField &f = ix.strings[x.field];
      const size_t values = f.dict.size();
      std::vector<uint8_t> &b = c.bits[v];
      b.assign(values, 0);
      c.values[v].clear();
      for (size_t d = 0; d < values; d++) {
        const std::string &s = f.dict[d];
        bool hit;
        if (x.op == 0) {
          hit = (s == x.arg);
        } else if (x.op == 1) {
          hit = false;
          for (const std::string &a : x.args) {
            if (a == s) {
              hit = true;
              break;
            }
          }
        } else {
          hit = like_match(s, x.arg);
        }
        if (hit) {
          b[d] = 1;
          c.values[v].push_back((uint32_t) d);
          m += (double) (f.ptr[d + 1] - f.ptr[d]);
        }
      }
    }
    c.count[v] = m;
    const double members = !updates.pending ? m : x.type == kAll ? total : (double) updates.live[x.label];
    const double e = x.negated ? total - members : members;
    c.lower[v] = e;
    c.upper[v] = e;
    if (x.type == kAll) {
      c.driver[v] = x.negated ? 0 : positions;
    } else {
      c.driver[v] = x.negated ? positions : m;
    }
  }
}

void seek(EvalContext &c, uint32_t lo) {
  const Tree &t = *c.tree;
  const Index::Impl &ix = *c.index;
  for (size_t v = 0; v < t.nodes.size(); v++) {
    if (t.nodes[v].type == kLabel) {
      const int32_t a = t.nodes[v].label;
      const uint32_t *s = ix.sidx.data() + ix.siptr[a];
      const uint32_t *e = ix.sidx.data() + ix.siptr[a + 1];
      c.cursor[v] = lo ? std::lower_bound(s, e, lo) : s;
      c.end[v] = e;
    }
  }
}

static inline bool test_node(EvalContext &c, int v, uint32_t p) {
  const Node &x = c.tree->nodes[v];
  const Index::Impl &ix = *c.index;
  bool r;
  switch (x.type) {
    case kLabel: {
      const uint32_t *&q = c.cursor[v];
      const uint32_t *e = c.end[v];
      if (q < e && *q < p) {
        size_t st = 1;
        while (q + st < e && q[st] < p) st <<= 1;
        q = std::lower_bound(q + (st >> 1) + 1, std::min(q + st + 1, e), p);
      }
      r = (q < e && *q == p);
      break;
    }
    case kRangeLeaf: {
      const uint32_t val = ix.ranges[x.field].col[p];
      r = (val >= x.lo && val < x.hi);
      break;
    }
    case kStringLeaf:
      r = c.bits[v][ix.strings[x.field].col[p]] != 0;
      break;
    case kAll:
      r = true;
      break;
    case kAnd:
      for (int k : x.kids) {
        if (!test_node(c, k, p)) return false;
      }
      return true;
    default:
      for (int k : x.kids) {
        if (test_node(c, k, p)) return true;
      }
      return false;
  }
  return r != x.negated;
}

bool test(EvalContext &c, int v, uint32_t p) { return test_node(c, v, p); }

namespace {

bool is_leaf(const Node &x) { return x.type == kLabel || x.type == kRangeLeaf || x.type == kStringLeaf; }

// Members of leaf v in [lo, hi): exact for a label, prorated for a field.
double leaf_members(const EvalContext &c, int v, uint32_t lo, uint32_t hi) {
  const Index::Impl &ix = *c.index;
  const Node &x = c.tree->nodes[v];
  const double width = (double) (hi - lo);
  double m;
  if (x.type == kLabel) {
    const uint32_t *s = ix.sidx.data() + ix.siptr[x.label];
    const uint32_t *e = ix.sidx.data() + ix.siptr[x.label + 1];
    m = (lo == 0 && hi == (uint32_t) ix.n) ? (double) (e - s) : (double) (std::lower_bound(s, e, hi) - std::lower_bound(s, e, lo));
  } else {
    m = c.count[v] * width / (double) ix.n;
  }
  return x.negated ? width - m : m;
}

// Work of listing the members of leaf v, as driver_list does it.
double listing_work(const EvalContext &c, int v, uint32_t lo, uint32_t hi, double members) {
  const Node &x = c.tree->nodes[v];
  const uint32_t n = (uint32_t) c.index->n;
  if (x.negated) return (double) (hi - lo);
  if (x.type == kLabel) return members;
  if (lo == 0 && hi == n && c.count[v] * 32 < (double) n) return members * std::log2(members + 2.0);
  return (double) (hi - lo);
}

// Work of filtering m candidates by leaf v. A label leaf gallops through its slice.
double filtering_work(const EvalContext &c, int v, double m, double members) {
  return c.tree->nodes[v].type == kLabel ? m * std::log2(members / (m + 1.0) + 2.0) : m;
}

// Keeps the candidates that satisfy leaf v.
void keep(EvalContext &c, int v, uint32_t lo, uint32_t hi, std::vector<uint32_t> &a) {
  const Index::Impl &ix = *c.index;
  const Node &x = c.tree->nodes[v];
  size_t o = 0;
  if (x.type == kLabel) {
    const uint32_t *s = ix.sidx.data() + ix.siptr[x.label];
    const uint32_t *e = ix.sidx.data() + ix.siptr[x.label + 1];
    const uint32_t *q = lo ? std::lower_bound(s, e, lo) : s;
    const uint32_t *f = hi < (uint32_t) ix.n ? std::lower_bound(q, e, hi) : e;
    for (const uint32_t p : a) {
      if (q < f && *q < p) {
        size_t st = 1;
        while (q + st < f && q[st] < p) st <<= 1;
        q = std::lower_bound(q + (st >> 1) + 1, std::min(q + st + 1, f), p);
      }
      if ((q < f && *q == p) != x.negated) a[o++] = p;
    }
  } else if (x.type == kRangeLeaf) {
    const uint32_t *col = ix.ranges[x.field].col.data();
    for (const uint32_t p : a) {
      const uint32_t val = col[p];
      if ((val >= x.lo && val < x.hi) != x.negated) a[o++] = p;
    }
  } else {
    const uint32_t *col = ix.strings[x.field].col.data();
    const uint8_t *b = c.bits[v].data();
    for (const uint32_t p : a) {
      if ((b[col[p]] != 0) != x.negated) a[o++] = p;
    }
  }
  a.resize(o);
}

// An AND of leaves, one leaf at a time: the members of the leaf with the least listing and filtering work, then one pass per
// other leaf, the most selective first. A large label under a wide range then starts from the label's slice, not from a scan of
// every position for the range followed by a test of the whole tree per candidate.
void and_of_leaves(EvalContext &c, const Node &x, uint32_t lo, uint32_t hi, std::vector<uint32_t> &out) {
  const size_t k = x.kids.size();
  std::vector<double> &members = c.members;
  members.resize(k);
  for (size_t i = 0; i < k; i++) members[i] = leaf_members(c, x.kids[i], lo, hi);
  size_t b = 0;
  double best = 0;
  for (size_t i = 0; i < k; i++) {
    double w = listing_work(c, x.kids[i], lo, hi, members[i]);
    for (size_t j = 0; j < k; j++) {
      if (j != i) w += filtering_work(c, x.kids[j], members[i], members[j]);
    }
    if (i == 0 || w < best) {
      b = i;
      best = w;
    }
  }
  driver_list(c, x.kids[b], lo, hi, out, 1);
  std::vector<int> &order = c.order;
  order.clear();
  for (size_t j = 0; j < k; j++) {
    if (j != b) order.push_back((int) j);
  }
  std::sort(order.begin(), order.end(), [&](int p, int q) { return members[p] < members[q]; });
  for (int j : order) {
    if (out.empty()) break;
    keep(c, x.kids[j], lo, hi, out);
  }
}

}  // namespace

void evaluate(EvalContext &c, uint32_t lo, uint32_t hi, std::vector<uint32_t> &out) {
  const Node &root = c.tree->nodes[c.tree->root];
  if (is_leaf(root)) {   // a leaf lists exactly its members
    driver_list(c, c.tree->root, lo, hi, out, 1);
    return;
  }
  if (root.type == kAnd && std::all_of(root.kids.begin(), root.kids.end(), [&](int k) { return is_leaf(c.tree->nodes[k]); })) {
    and_of_leaves(c, root, lo, hi, out);
    return;
  }
  driver_list(c, c.tree->root, lo, hi, c.candidates, 1);
  seek(c, lo);
  out.clear();
  const int r = c.tree->root;
  for (uint32_t p : c.candidates) {
    if (test_node(c, r, p)) out.push_back(p);
  }
}

double exact_count(EvalContext &pos, EvalContext &neg, bool &ready) {
  const uint32_t n = (uint32_t) pos.index->n;
  if (pos.driver[pos.tree->root] <= neg.driver[neg.tree->root]) {
    evaluate(pos, 0u, n, pos.all);
    ready = true;
    return (double) pos.all.size();
  }
  evaluate(neg, 0u, n, neg.all);
  ready = false;
  return (double) n - (double) neg.all.size();
}

}  // namespace kuroann
