#!/usr/bin/env python3
"""Synthetic clustered vectors with labels and a price, plus sample and test queries with filters.
usage: make_synthetic.py OUT_DIR [--n 200000] [--dim 128]
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("out")
ap.add_argument("--n", type=int, default=200000)
ap.add_argument("--dim", type=int, default=128)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(0)
clusters = 256
centers = rng.normal(size=(clusters, a.dim)) * 4.0
which = rng.integers(clusters, size=a.n)
base = (centers[which] + rng.normal(size=(a.n, a.dim))).astype(np.float32)


def write_vectors(path, x):
    with open(path, "wb") as f:
        np.array(x.shape, dtype=np.uint32).tofile(f)
        x.tofile(f)


write_vectors(f"{a.out}/base.fbin", base)

nlabels = 14
rows, columns = [], []
for label in range(nlabels):
    members = np.flatnonzero(rng.random(a.n) < 0.5 ** (label + 1))
    rows.append(members)
    columns.append(np.full(len(members), label, dtype=np.int32))
members = list(rows)
rows = np.concatenate(rows)
columns = np.concatenate(columns)
order = np.lexsort((columns, rows))
rows, columns = rows[order], columns[order]
ptr = np.zeros(a.n + 1, dtype=np.int64)
ptr[1:] = np.cumsum(np.bincount(rows, minlength=a.n))
with open(f"{a.out}/labels.spmat", "wb") as f:
    np.array([a.n, nlabels, len(columns)], dtype=np.int64).tofile(f)
    ptr.tofile(f)
    columns.tofile(f)
    np.ones(len(columns), dtype=np.float32).tofile(f)
price = rng.integers(0, 100000, size=a.n, dtype=np.uint32)
price.tofile(f"{a.out}/price.u32")
MIN_MATCHES = 30   # well above the 10 neighbors a query asks for, so that every query has an exact answer


def filters(count):
    out = []
    while len(out) < count:
        label = int(rng.integers(0, nlabels - 2))
        kind = int(rng.integers(3))
        if kind == 0:
            expr, matches = f"L {label}", len(members[label])
        elif kind == 1:
            high = int(rng.integers(20000, 100000))
            expr, matches = f"AND 2 L {label} R price 0 {high}", int(np.count_nonzero(price[members[label]] < high))
        else:
            high = int(rng.integers(200, 100000))
            expr, matches = f"R price 0 {high}", int(np.count_nonzero(price < high))
        if matches >= MIN_MATCHES:
            out.append(expr)
    return out


for name, count in (("sample", 2000), ("test", 1000)):
    picks = rng.integers(clusters, size=count)
    q = (centers[picks] + rng.normal(size=(count, a.dim))).astype(np.float32)
    write_vectors(f"{a.out}/{name}.fbin", q)
    with open(f"{a.out}/{name}.expr", "w") as f:
        f.write("".join(e + "\n" for e in filters(count)))
print("wrote", a.out)
