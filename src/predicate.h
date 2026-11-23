#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "internal.h"

namespace kuroann {

enum NodeType { kLabel, kRangeLeaf, kStringLeaf, kAll, kAnd, kOr };

struct Node {
  NodeType type = kLabel;
  bool negated = false;
  int32_t label = 0;
  uint32_t lo = 0, hi = 0;
  int field = -1;
  int op = 0;                     // string: 0 eq, 1 in, 2 like
  std::string arg;
  std::vector<std::string> args;
  std::vector<int> kids;
};

// An expression in negation normal form. Kids have larger indices than their parent.
struct Tree {
  std::vector<Node> nodes;
  int root = 0;
};

// Per-thread evaluation state of one tree.
struct EvalContext {
  const Index::Impl *index = nullptr;
  const Tree *tree = nullptr;
  std::vector<double> count, lower, upper, driver;   // leaf members, bounds on a node's members, candidate list size
  std::vector<const uint32_t *> cursor, end;         // label leaves
  std::vector<std::vector<uint8_t>> bits;            // string leaves: dictionary hits
  std::vector<std::vector<uint32_t>> values;         // string leaves: dictionary ids that hit
  std::vector<std::vector<uint32_t>> pool;
  std::vector<uint32_t> candidates, out, all;
  std::vector<double> members;                       // AND of leaves: members of each leaf in the range
  std::vector<int> order;                            // AND of leaves: the order of the filtering passes
};

Tree parse_expression(const Index::Impl &index, const std::string &expression, bool negate);

// Leaf counts, bounds and driver sizes, bottom-up.
void prepare(EvalContext &c, const Index::Impl &index, const Tree &tree);
// Label cursors at the first member >= lo.
void seek(EvalContext &c, uint32_t lo);
// Membership of position p. p must not decrease between calls without a seek.
bool test(EvalContext &c, int node, uint32_t p);
// Members in [lo, hi), ascending.
void evaluate(EvalContext &c, uint32_t lo, uint32_t hi, std::vector<uint32_t> &out);
// Exact count. E's members stay in positive.all when ready is true.
double exact_count(EvalContext &positive, EvalContext &negative, bool &ready);

}  // namespace kuroann
