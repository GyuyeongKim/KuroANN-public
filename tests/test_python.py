#!/usr/bin/env python3
"""Checks the Python module against a brute-force oracle: search, insert, remove, merge, save, reload."""
import os
import sys
import tempfile

import numpy as np

import kuroann


def write_inputs(directory, x, labels, columns):
    base = os.path.join(directory, "base.fbin")
    with open(base, "wb") as f:
        np.array(x.shape, dtype=np.uint32).tofile(f)
        x.tofile(f)
    path = os.path.join(directory, "labels.spmat")
    start = np.cumsum([0] + [len(row) for row in labels], dtype=np.int64)
    with open(path, "wb") as f:
        np.array([len(labels), columns, start[-1]], dtype=np.int64).tofile(f)
        start.tofile(f)
        np.array([l for row in labels for l in row], dtype=np.int32).tofile(f)
        np.ones(start[-1], dtype=np.float32).tofile(f)
    return base, path


def exact(x, alive, member, q, k):
    d = ((x - q) ** 2).sum(1)
    ids = np.flatnonzero(alive & member)
    return ids[np.lexsort((ids, d[ids]))][:k]


def main():
    rng = np.random.default_rng(3)
    n, added, dim, k = 8000, 500, 32, 10
    centers = rng.normal(size=(32, dim)) * 4
    x = (centers[rng.integers(32, size=n + added)] + rng.normal(size=(n + added, dim))).astype(np.float32)
    labels = [[l for l in range(4) if rng.random() < 0.5 ** (l + 1)] for _ in range(n + added)]
    has = np.array([[l in row for l in range(4)] for row in labels])
    queries = x[rng.integers(n + added, size=8)] + rng.normal(size=(8, dim)).astype(np.float32) * 0.3
    filters = ["ALL", "L 0", "L 3", "NOT L 0", "AND 2 L 0 L 1", "OR 2 L 2 L 3", "AND 2 L 1 NOT L 2", "L 1"]
    members = [np.ones(n + added, bool), has[:, 0], has[:, 3], ~has[:, 0], has[:, 0] & has[:, 1], has[:, 2] | has[:, 3], has[:, 1] & ~has[:, 2], has[:, 1]]
    alive = np.zeros(n + added, bool)
    alive[:n] = True

    def compare(index, what):
        ids, distances, reads = index.search(queries, filters, k=k, threads=2)
        for i in range(len(queries)):
            want = exact(x, alive, members[i], queries[i], k)
            assert list(ids[i][:len(want)]) == list(want), f"{what}: {filters[i]}"
            assert (ids[i][len(want):] == -1).all(), what
            true = ((x[want] - queries[i]) ** 2).sum(1)
            assert np.allclose(distances[i][:len(want)], true, rtol=1e-3, atol=1e-3), what
            assert reads[i] == min(int((alive & members[i]).sum()), n + added), what
        print("ok  ", what)

    with tempfile.TemporaryDirectory(prefix="kuroann_py_", dir=".") as directory:
        base, spmat = write_inputs(directory, x[:n], labels[:n], 4)
        prefix = os.path.join(directory, "index")
        kuroann.build(base, prefix, labels=spmat, pq=8, partitions=16)
        plan = os.path.join(directory, "plan.txt")
        with open(plan, "w") as f:
            f.write(f"vmax={n * 10} np=0 L={n + added} alpha=0\n")
        index = kuroann.Index(prefix)
        assert (index.size, index.dimension, index.partitions, index.pq_pieces, index.labels) == (n, dim, 16, 8, 4)
        index.load_plan(plan)
        compare(index, "search")
        ids = index.insert(x[n:], [labels[i] for i in range(n, n + added)])
        assert list(ids) == list(range(n, n + added))
        alive[n:] = True
        assert index.size == n + added
        compare(index, "search after insert")
        for i in range(0, n + added, 7):
            assert index.remove(i)
            alive[i] = False
        assert not index.remove(0)
        assert index.size == int(alive.sum())
        compare(index, "search after remove")
        index.merge()
        compare(index, "search after merge")
        index.save()
        again = kuroann.Index(prefix)
        again.load_plan(plan)
        assert again.size == int(alive.sum())
        compare(again, "search after reload")
        try:
            again.search(queries[:1], ["L 9"], k=k)
        except ValueError:
            print("ok   a bad filter raises ValueError")
        else:
            raise AssertionError("a bad filter was accepted")
    print("all passed")


if __name__ == "__main__":
    sys.exit(main())
