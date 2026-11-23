"""Read an index's files and evaluate filter expressions in NumPy, independently of the C++ engine."""
import os

import numpy as np


class IndexFiles:
    def __init__(self, prefix):
        meta = np.fromfile(prefix + ".meta", dtype=np.int32)
        self.n, self.dim, self.nlist, self.m, self.dsub, self.ksub = (int(x) for x in meta[:6])
        self.element_bytes = int(meta[6]) if len(meta) > 6 else 4
        self.ids = int(meta[7]) if len(meta) > 7 else self.n
        self.prefix = prefix
        self.coarse = np.fromfile(prefix + ".coarse", dtype=np.float32).reshape(self.nlist, self.dim)
        self.book = np.fromfile(prefix + ".book", dtype=np.float32).reshape(self.m, self.ksub, self.dsub)
        self.lptr = np.fromfile(prefix + ".lptr", dtype=np.int64)
        self.lids = np.memmap(prefix + ".lids", dtype=np.int32, mode="r", shape=(self.n,))
        self.codes = np.memmap(prefix + ".codes", dtype=np.uint8, mode="r", shape=(self.n, self.m))
        self.siptr = np.fromfile(prefix + ".siptr", dtype=np.int64)
        self.sidx = np.memmap(prefix + ".sidx", dtype=np.uint32, mode="r", shape=(int(self.siptr[-1]),)) if self.siptr[-1] else np.zeros(0, np.uint32)
        vector_bytes = self.dim * self.element_bytes
        self.per_block = 4096 // vector_bytes
        self.vector_bytes = vector_bytes
        self.blocks = np.memmap(prefix + ".vectors", dtype=np.uint8, mode="r")
        self.ranges, self.strings = {}, {}
        fields = prefix + ".attr.fields"
        if os.path.exists(fields):
            for line in open(fields):
                kind, name = line.split()
                base = f"{prefix}.attr.{name}"
                col = np.memmap(base + ".col", dtype=np.uint32, mode="r", shape=(self.n,))
                if kind == "range":
                    self.ranges[name] = col
                else:
                    self.strings[name] = (col, open(base + ".dict").read().split("\n")[:-1])

    def vectors(self, ids):
        """Raw vectors of the given ids as float32."""
        ids = np.asarray(ids, dtype=np.int64)
        offsets = (ids // self.per_block) * 4096 + (ids % self.per_block) * self.vector_bytes
        out = np.empty((len(ids), self.dim), dtype=np.float32)
        dtype = np.float32 if self.element_bytes == 4 else np.uint8
        order = np.argsort(offsets)
        for i in order:
            start = int(offsets[i])
            out[i] = np.frombuffer(self.blocks[start:start + self.vector_bytes], dtype=dtype)
        return out

    def first_vectors(self, start, stop):
        """Raw vectors of a run of ids as float32."""
        count = stop - start
        if self.per_block == 1:
            raw = np.asarray(self.blocks[start * 4096:stop * 4096]).reshape(count, 4096)[:, :self.vector_bytes]
            data = raw.copy()
        else:
            first = start // self.per_block
            last = (stop + self.per_block - 1) // self.per_block
            raw = np.asarray(self.blocks[first * 4096:last * 4096]).reshape(last - first, 4096)
            data = raw[:, :self.per_block * self.vector_bytes].reshape(-1, self.vector_bytes)
            data = data[start - first * self.per_block:start - first * self.per_block + count].copy()
        if self.element_bytes == 4:
            return data.reshape(count, -1).view(np.float32)
        return data.astype(np.float32)


def read_vectors(path):
    n, d = (int(x) for x in np.fromfile(path, dtype=np.uint32, count=2))
    payload = os.path.getsize(path) - 8
    dtype = np.float32 if payload == n * d * 4 else np.uint8
    return np.fromfile(path, dtype=dtype, offset=8).reshape(n, d).astype(np.float32)


def write_vectors(path, x):
    with open(path, "wb") as f:
        np.array(x.shape, dtype=np.uint32).tofile(f)
        x.astype(np.float32).tofile(f)


def read_lines(path):
    return [line.strip() for line in open(path) if line.strip() and not line.startswith("#")]


def _parse(tokens, i):
    t = tokens[i]
    i += 1
    if t == "ALL":
        return ("ALL",), i
    if t == "NOT":
        child, i = _parse(tokens, i)
        return ("NOT", child), i
    if t in ("AND", "OR"):
        k = int(tokens[i])
        i += 1
        kids = []
        for _ in range(k):
            child, i = _parse(tokens, i)
            kids.append(child)
        return (t, kids), i
    if t == "L":
        return ("L", int(tokens[i])), i + 1
    if t == "R":
        return ("R", tokens[i], int(tokens[i + 1]), int(tokens[i + 2])), i + 3
    if t == "S":
        return ("S", tokens[i], tokens[i + 1], tokens[i + 2]), i + 3
    raise ValueError(f"unknown token {t}")


def parse(expression):
    tokens = expression.split()
    tree, i = _parse(tokens, 0)
    if i != len(tokens):
        raise ValueError("trailing tokens: " + expression)
    return tree


def like_match(s, p):
    i = j = mark = 0
    star = -1
    while i < len(s):
        if j < len(p) and (p[j] == "_" or p[j] == s[i]):
            i += 1
            j += 1
        elif j < len(p) and p[j] == "%":
            star = j
            j += 1
            mark = i
        elif star != -1:
            j = star + 1
            mark += 1
            i = mark
        else:
            return False
    while j < len(p) and p[j] == "%":
        j += 1
    return j == len(p)


class Mask:
    """Evaluates expressions on a set of positions. Label and string leaves are cached across expressions."""

    def __init__(self, files, positions=None):
        self.f = files
        self.positions = positions
        self.count = files.n if positions is None else len(positions)
        self.cache = {}

    def _label(self, label):
        key = ("L", label)
        if key not in self.cache:
            posting = self.f.sidx[self.f.siptr[label]:self.f.siptr[label + 1]]
            m = np.zeros(self.count, dtype=bool)
            if self.positions is None:
                m[np.asarray(posting, dtype=np.int64)] = True
            elif len(posting):
                at = np.searchsorted(posting, self.positions)
                at[at >= len(posting)] = len(posting) - 1
                m = np.asarray(posting)[at] == self.positions
            self.cache[key] = m
        return self.cache[key]

    def _string(self, name, op, arg):
        key = ("S", name, op, arg)
        if key not in self.cache:
            col, values = self.f.strings[name]
            if op == "eq":
                hit = [v == arg for v in values]
            elif op == "in":
                wanted = set(arg.split(","))
                hit = [v in wanted for v in values]
            else:
                hit = [like_match(v, arg) for v in values]
            hit = np.array(hit, dtype=bool)
            self.cache[key] = hit[col if self.positions is None else col[self.positions]]
        return self.cache[key]

    def __call__(self, t):
        k = t[0]
        if k == "ALL":
            return np.ones(self.count, dtype=bool)
        if k == "L":
            return self._label(t[1])
        if k == "R":
            col = self.f.ranges[t[1]]
            v = col if self.positions is None else col[self.positions]
            return (v >= t[2]) & (v < t[3])
        if k == "S":
            return self._string(t[1], t[2], t[3])
        if k == "NOT":
            return ~self(t[1])
        masks = [self(c) for c in t[1]]
        out = masks[0].copy()
        for m in masks[1:]:
            if k == "AND":
                out &= m
            else:
                out |= m
        return out
