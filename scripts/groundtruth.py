#!/usr/bin/env python3
"""Exact filtered nearest neighbors for one query set, in one pass over the vectors.

Writes OUT.gtpf (u32 queries, u32 k, u32 ids, f32 squared L2 distances; ties by id) and OUT.counts (i64 matches per query).
A query with fewer than k matches fails.

usage: groundtruth.py --index P --queries Q.fbin [--filters F.expr] --out OUT [--k 10] [--chunk 1000000] [--workers N]
"""
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kuroann_files import IndexFiles, Mask, parse, read_lines, read_vectors

DEPTH = 64  # kept per query per chunk, so that ties at the k-th distance can be resolved exactly
G = {}


def work(rows):
    out = []
    for r in rows:
        ids = np.flatnonzero(G["mask"](G["trees"][r]) & G["present"])
        c = len(ids)
        if c == 0:
            out.append((r, None, None, 0))
            continue
        d = G["D"][r, ids]
        if c > DEPTH:
            keep = np.argpartition(d, DEPTH - 1)[:DEPTH]
            ids, d = ids[keep], d[keep]
        out.append((r, ids.astype(np.int64) + G["start"], d.astype(np.float32), c))
    return out


def compute(index, queries, filters, out, k=10, chunk=1000000, workers=None):
    f = IndexFiles(index)
    Q = read_vectors(queries)
    lines = read_lines(filters) if filters else ["ALL"] * len(Q)
    if len(lines) != len(Q):
        raise SystemExit(f"{len(Q)} queries but {len(lines)} expressions")
    trees = [parse(e) for e in lines]
    workers = workers or os.cpu_count()
    nq = len(Q)
    best_d = np.full((nq, DEPTH), np.inf, dtype=np.float32)
    best_i = np.full((nq, DEPTH), -1, dtype=np.int64)
    counts = np.zeros(nq, dtype=np.int64)
    norms = (Q ** 2).sum(1)
    batches = [list(range(i, min(nq, i + 64))) for i in range(0, nq, 64)]
    lids = np.asarray(f.lids, dtype=np.int64)
    position_of = np.full(f.ids, -1, dtype=np.int64)
    position_of[lids] = np.arange(f.n, dtype=np.int64)
    for s in range(0, f.ids, chunk):
        e = min(f.ids, s + chunk)
        X = f.first_vectors(s, e)
        G.clear()
        G.update(D=norms[:, None] + (X ** 2).sum(1)[None, :] - 2.0 * (Q @ X.T), trees=trees, start=s, mask=Mask(f, position_of[s:e]), present=position_of[s:e] >= 0)
        del X
        with mp.get_context("fork").Pool(workers) as pool:
            for res in pool.imap_unordered(work, batches):
                for r, ids, d, c in res:
                    counts[r] += c
                    if ids is None:
                        continue
                    bd = np.concatenate([best_d[r], d])
                    bi = np.concatenate([best_i[r], ids])
                    keep = np.lexsort((bi, bd))[:DEPTH]
                    best_d[r], best_i[r] = bd[keep], bi[keep]
        G.clear()
        print(f"chunk {s}-{e}", flush=True)
    if (counts < k).any():
        raise SystemExit(f"{int((counts < k).sum())} queries match fewer than {k} vectors")
    ids_out = np.zeros((nq, k), dtype=np.uint32)
    dist_out = np.zeros((nq, k), dtype=np.float32)
    for i in range(nq):
        c = best_i[i][best_i[i] >= 0]
        x = f.vectors(c).astype(np.float64)
        d = ((x - Q[i].astype(np.float64)[None, :]) ** 2).sum(1)
        o = np.lexsort((c, d))[:k]
        ids_out[i], dist_out[i] = c[o], d[o]
    with open(out + ".gtpf", "wb") as w:
        np.array([nq, k], dtype=np.uint32).tofile(w)
        ids_out.tofile(w)
        dist_out.tofile(w)
    counts.tofile(out + ".counts")
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--filters")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=1000000)
    ap.add_argument("--workers", type=int)
    a = ap.parse_args()
    counts = compute(a.index, a.queries, a.filters, a.out, a.k, a.chunk, a.workers)
    print(f"{a.out}.gtpf: {len(counts)} queries, matches min/median/max {counts.min()}/{int(np.median(counts))}/{counts.max()}")


if __name__ == "__main__":
    main()
