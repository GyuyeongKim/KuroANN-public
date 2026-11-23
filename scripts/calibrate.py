#!/usr/bin/env python3
"""Calibrate a plan on sample queries: the table from match counts to route, partitions, pool, and stopping.

Steps: exact ground truth of the sample; count bands; a stop table per band and route; a timed sweep per band; one row per band,
the fastest setting whose recall clears the target by a margin of Z binomial standard errors. Between the count ranges of two
bands, the row is the fastest setting without a stop table that clears the target at both. Below the first range an exhaustive
row serves, and above the last range a partition row serves.

Writes OUT/plan_<target>.txt, OUT/tables/, OUT/selection.json. Run it alone on the machine: the sweep measures throughput.

usage: calibrate.py --index P --queries Q.fbin --filters F.expr --out OUT --kuroann build/kuroann [--targets 0.9,0.95,0.99]
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import re
import struct
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import groundtruth
from kuroann_files import IndexFiles, Mask, parse, read_lines, read_vectors, write_vectors

BOUNDS = [0, 16, 64, 256, 1024, 2048]
ALPHAS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.25]
EDGES = [1e-4, 1e-3, 1e-2, 5e-2, 2e-1]
MAX_POOL = 1536
LARGE_POOLS = (2048, 3072, 4096, 6144, 8192)
EXHAUSTIVE_ALPHAS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1)
PARTITION_ALPHAS = (0.001, 0.005, 0.02, 0.1)
BROAD_TABLE_QUERIES = 100  # exhaustive pools of the broad bands are costly, and 100 queries give enough samples per rank
G = {}
LINE = re.compile(r"partitions=(\d+) pool=(\d+) round=\d+ qps=([\d.]+) .*? reads=([\d.]+) .*?recall=([\d.]+)")


def table_query(i):
    f, q = G["f"], G["Q"][i]
    pos = np.flatnonzero(Mask(f)(G["trees"][i])).astype(np.int64)
    if G["np"] > 0 and len(pos):
        dc = (f.coarse ** 2).sum(1) - 2.0 * (f.coarse @ q)
        cells = np.argpartition(dc, min(G["np"], f.nlist) - 1)[:G["np"]]
        keep = np.zeros(f.nlist, dtype=bool)
        keep[cells] = True
        pos = pos[keep[np.searchsorted(f.lptr, pos, side="right") - 1]]
    if len(pos) == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.int64)
    lut = ((f.book - q.reshape(f.m, 1, f.dsub)) ** 2).sum(axis=2, dtype=np.float32)
    d = np.empty(len(pos), dtype=np.float32)
    rows = np.arange(f.m)[None, :]
    for s in range(0, len(pos), 262144):
        d[s:s + 262144] = lut[rows, np.asarray(f.codes[pos[s:s + 262144]])].sum(axis=1, dtype=np.float32)
    k = min(MAX_POOL, len(d))
    top = np.argpartition(d, k - 1)[:k] if len(d) > k else np.arange(len(d))
    top = top[np.argsort(d[top], kind="stable")]
    x = f.vectors(np.asarray(f.lids[pos[top]]))
    exact = ((x - q[None, :]) ** 2).sum(1, dtype=np.float32)
    return exact - d[top], np.searchsorted(BOUNDS, np.arange(k), side="right") - 1


def write_table(f, trees, Q, partitions, out, workers):
    G.update(f=f, trees=trees, Q=Q, np=partitions)
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(table_query, range(len(Q)), chunksize=2)
    errors = np.concatenate([r[0] for r in res])
    buckets = np.concatenate([r[1] for r in res])
    table = np.zeros((len(ALPHAS), len(BOUNDS) - 1), dtype=np.float32)
    for b in range(len(BOUNDS) - 1):
        e = errors[buckets == b]
        for a, alpha in enumerate(ALPHAS):
            table[a, b] = np.quantile(e, alpha) if len(e) else -1e30
    with open(out, "wb") as w:
        w.write(struct.pack("<IIII", 0x44334331, len(BOUNDS) - 1, len(ALPHAS), 0))
        np.array(BOUNDS, dtype=np.int32).tofile(w)
        np.array(ALPHAS, dtype=np.float32).tofile(w)
        table.tofile(w)


def split_bands(n, counts, min_queries):
    edges = [0] + [e * n for e in EDGES] + [n + 1]
    band = np.searchsorted(edges, counts, side="right") - 1
    groups = [list(np.flatnonzero(band == b)) for b in range(len(edges) - 1)]
    merged = []
    for g in groups:
        if merged and len(merged[-1]) < min_queries:
            merged[-1] = merged[-1] + g
        elif g:
            merged.append(g)
    if len(merged) > 1 and len(merged[-1]) < min_queries:
        last = merged.pop()
        merged[-1] = merged[-1] + last
    return [sorted(g) for g in merged]


def grids(nlist):
    s = nlist / 32768
    partitions = sorted({max(8, int(round(x * s))) for x in (64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)})
    fixed_e = [(64, L) for L in (32, 40, 48, 56, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 768, 1024, 1536)]
    fixed_p = [(p, L) for p in partitions for L in (64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536) if L <= p * 64]
    stop_e = [(64, L) for L in (128, 256, 512, 1024, 1536)]
    stop_p = [(p, L) for p in partitions for L in (128, 256, 512, 1024, 1536) if L <= p * 64]
    return fixed_e, fixed_p, stop_e, stop_p


def sweep(a, band_dir, route, settings, alpha=0.0, table=None):
    cmd = [a.kuroann, "search", "--index", a.index, "--queries", f"{band_dir}/queries.fbin", "--filters", f"{band_dir}/filters.expr",
           "--groundtruth", f"{band_dir}/truth.gtpf", "--threads", str(a.threads), "--route", route, "--rounds", "1",
           "--settings", ",".join(f"{p}:{L}" for p, L in settings)]
    if alpha > 0:
        cmd += ["--alpha", str(alpha), "--table", table]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    rows = [LINE.search(line) for line in out.splitlines() if line.startswith("partitions=")]
    if len(rows) != len(settings) or not all(rows):
        raise SystemExit("unexpected output: " + out[:400])
    return [(int(m[1]), int(m[2]), float(m[5]), float(m[3]), float(m[4])) for m in rows]


def select(rows, bands, target, z, k):
    """rows: {key: {band: (recall, qps, reads)}}; key = (arm, partitions, pool, route, alpha, table)"""
    def margin(b):
        return z * math.sqrt(target * (1 - target) / (k * bands[b]["queries"]))

    def clears(key, b):
        return b in rows[key] and rows[key][b][0] >= target + margin(b) - 1e-9

    def fastest(b, keep=lambda key: True):
        c = [(v[b][1], key) for key, v in rows.items() if keep(key) and clears(key, b)]
        return max(c)[1] if c else None

    def fixed_between(x, y):
        c = [(min(rows[key][x][1], rows[key][y][1]), key) for key in rows if key[4] == 0 and clears(key, x) and clears(key, y)]
        return max(c)[1] if c else None

    def text(vmax, key):
        np_ = 0 if key[3] == "exhaustive" else key[1]
        s = f"vmax={vmax:.0f} np={np_} L={key[2]} alpha={key[4]:g}"
        return s + (f" table={key[5]}" if key[4] > 0 else "")

    order = sorted(bands, key=lambda b: bands[b]["median"])
    own = {}
    for b in order:
        own[b] = fastest(b)
        if own[b] is None:
            raise SystemExit(f"no setting reaches {target} on band {b}; widen the grid or lower the target")
    lines = [f"# target {target}, margin z {z}"]
    first = order[0]
    if own[first][3] != "exhaustive":
        e = fastest(first, lambda key: key[3] == "exhaustive")
        if e is not None:
            lines.append(text(bands[first]["low"] - 1, e))
    for i, b in enumerate(order):
        if i > 0 and bands[order[i - 1]]["high"] < bands[b]["low"] - 1:
            between = fixed_between(order[i - 1], b)
            if between is None:
                raise SystemExit(f"{target}: no fixed setting clears both {order[i - 1]} and {b}")
            lines.append(text(bands[b]["low"] - 1, between))
        if i == len(order) - 1:
            if own[b][3] == "exhaustive":
                p = fastest(b, lambda key: key[3] == "partition")
                if p is not None:
                    lines.append(text(bands[b]["high"], own[b]))
                    lines.append(text(1e15, p))
                    continue
            lines.append(text(1e15, own[b]))
        else:
            lines.append(text(bands[b]["high"], own[b]))
    out, last = [], -1.0
    for line in lines:
        if line.startswith("#"):
            out.append(line)
            continue
        v = float(line.split()[0].split("=")[1])
        if v > last:
            out.append(line)
            last = v
    return out, {b: list(own[b]) for b in order}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--filters", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kuroann", required=True)
    ap.add_argument("--targets", default="0.9,0.95,0.99")
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--min-band", type=int, default=50)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    f = IndexFiles(a.index)
    Q = read_vectors(a.queries)
    lines = read_lines(a.filters)
    truth = os.path.join(a.out, "truth")
    counts = groundtruth.compute(a.index, a.queries, a.filters, truth, a.k, workers=a.threads)
    n_q, k_q = (int(x) for x in np.fromfile(truth + ".gtpf", dtype=np.uint32, count=2))
    gt_ids = np.fromfile(truth + ".gtpf", dtype=np.uint32, offset=8, count=n_q * k_q).reshape(n_q, k_q)
    gt_d = np.fromfile(truth + ".gtpf", dtype=np.float32, offset=8 + 4 * n_q * k_q, count=n_q * k_q).reshape(n_q, k_q)

    groups = split_bands(f.n, counts, a.min_band)
    bands = {}
    fixed_e, fixed_p, stop_e, stop_p = grids(f.nlist)
    rows = {}
    for number, g in enumerate(groups):
        band = f"band{number}"
        d = os.path.join(a.out, band)
        os.makedirs(d, exist_ok=True)
        write_vectors(f"{d}/queries.fbin", Q[g])
        with open(f"{d}/filters.expr", "w") as w:
            w.write("".join(lines[i] + "\n" for i in g))
        with open(f"{d}/truth.gtpf", "wb") as w:
            np.array([len(g), k_q], dtype=np.uint32).tofile(w)
            gt_ids[g].tofile(w)
            gt_d[g].tofile(w)
        c = counts[g]
        bands[band] = dict(queries=len(g), low=int(c.min()), high=int(c.max()), median=float(np.median(c)))
        print(band, bands[band], flush=True)
        trees = [parse(lines[i]) for i in g]
        broad = bands[band]["median"] >= 0.01 * f.n
        keep = min(len(g), BROAD_TABLE_QUERIES) if broad else len(g)
        exh = f"tables/{band}_exhaustive.bin"
        part = f"tables/{band}_partition.bin"
        write_table(f, trees[:keep], Q[g][:keep], 0, os.path.join(a.out, exh), a.threads)
        write_table(f, trees, Q[g], max(1, f.nlist // 64), os.path.join(a.out, part), a.threads)
        arms = [("fixed", "exhaustive", fixed_e, 0.0, None), ("fixed", "partition", fixed_p, 0.0, None)]
        arms += [("stop", "exhaustive", stop_e, al, exh) for al in EXHAUSTIVE_ALPHAS]
        arms += [("stop", "partition", stop_p, al, part) for al in PARTITION_ALPHAS]
        for arm, route, settings, alpha, table in arms:
            res = sweep(a, d, route, settings, alpha, os.path.join(a.out, table) if table else None)
            for p, L, recall, qps, reads in res:
                rows.setdefault((arm, p, L, route, alpha, table or ""), {})[band] = (recall, qps, reads)
            print(band, arm, route, alpha, "done", flush=True)

    # Broad filters on some data need larger pools to reach a high target. A band that falls short gets them on the partition route, and so does the
    # band below it, so that a fixed setting can serve the range between the two. Stopping works at these pools too, because the last bucket of a
    # stop table covers every deeper rank.
    targets = [float(x) for x in a.targets.split(",")]
    names = list(bands)
    top = max(targets)
    short = [b for b in names if max(v[b][0] for v in rows.values() if b in v) < top + a.z * math.sqrt(top * (1 - top) / (a.k * bands[b]["queries"]))]
    if short:
        partitions = sorted({p for p, _ in fixed_p if f.nlist // 32 <= p <= f.nlist * 3 // 16})
        for band in names[max(0, names.index(short[0]) - 1):]:
            wide = [(p, L) for p in partitions for L in LARGE_POOLS]
            part = f"tables/{band}_partition.bin"
            for arm, alpha, table in [("fixed", 0.0, None)] + [("stop", al, part) for al in PARTITION_ALPHAS]:
                res = sweep(a, os.path.join(a.out, band), "partition", wide, alpha, os.path.join(a.out, table) if table else None)
                for p, L, recall, qps, reads in res:
                    rows.setdefault((arm, p, L, "partition", alpha, table or ""), {})[band] = (recall, qps, reads)
            print(band, "larger pools done", flush=True)

    selection = {"bands": bands, "rows": {}}
    for t in targets:
        plan, own = select(rows, bands, t, a.z, a.k)
        with open(os.path.join(a.out, f"plan_{int(round(t * 100))}.txt"), "w") as w:
            w.write("\n".join(plan) + "\n")
        selection["rows"][str(t)] = own
        print("\n".join(plan), flush=True)
    with open(os.path.join(a.out, "selection.json"), "w") as w:
        json.dump(selection, w, indent=1)


if __name__ == "__main__":
    main()
