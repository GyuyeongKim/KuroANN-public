#!/usr/bin/env python3
"""Measure one system on every evaluation run of a dataset, one setting at a time, and write one JSON line per run and setting.

KuroANN follows its plans for recall 0.90, 0.95, and 0.99. PipeANN-Filter sweeps its search list L, and its sweep of a band stops
once each of its runs reaches recall 0.999 or falls below 10 queries per second. Faiss-SSD runs every nprobe and rerank pool. Recall
counts only returned vectors that match the filter, and a distance tie counts as a hit, as in kuroann search.

usage: bench.py kuroann|pipeann|faiss --data DIR --out OUT.jsonl --bin BINARY --index P [--plans DIR] [--threads T]
DIR holds base.fbin, base.spmat, the range field, and runs/ (see scripts/datasets.py), with the exact answers in runs/<name>.gtpf.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from kuroann_files import parse, read_lines

PIPEANN_LISTS = [[10, 15, 20, 30, 40, 60, 80, 100], [150, 200, 300, 400, 600, 800], [1000, 1500, 2000, 3000], [4000, 6000, 8000]]
FAISS_PROBES = (64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192)
FAISS_POOLS = (32, 64, 128, 256, 512, 1024, 2048)
FAISS_LARGE_POOLS = (4096, 8192)  # for filters over 1% with many probes, where codes can need them to reach recall 0.99
RECALL_END = 0.999
QPS_END = 10
PIPEANN_ROW = re.compile(r"^\s*(\d+)\s+(\d+)(.*)$")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
FAISS_ROW = re.compile(r"^RES nprobe=(\d+) pool=(\d+) queries=(\d+) qps=([\d.]+) mean_us=([\d.]+) p50_us=([\d.]+) p99_us=([\d.]+) reads=([\d.]+) io_errors=(\d+)$")
KUROANN_ROW = re.compile(r"qps=([\d.]+) .*?reads=([\d.]+) .*?recall=([\d.]+)")


class Data:
    def __init__(self, directory):
        self.dir = directory
        self.spec = json.load(open(f"{directory}/runs/runs.json"))
        self.runs = [r for r in self.spec["runs"] if r["set"] == "eval"]
        self.n, self.dim = (int(x) for x in np.fromfile(f"{directory}/base.fbin", dtype=np.uint32, count=2))
        self.base = np.memmap(f"{directory}/base.fbin", dtype=np.float32, mode="r", offset=8, shape=(self.n, self.dim))
        rows, _, entries = (int(x) for x in np.fromfile(f"{directory}/base.spmat", dtype=np.int64, count=3))
        self.row = np.memmap(f"{directory}/base.spmat", dtype=np.int64, mode="r", offset=24, shape=(rows + 1,))
        self.labels = np.memmap(f"{directory}/base.spmat", dtype=np.int32, mode="r", offset=24 + 8 * (rows + 1), shape=(entries,))
        self.field = self.spec["range_field"]
        self.values = np.memmap(f"{directory}/{self.field}.u32", dtype=np.uint32, mode="r")

    def path(self, run, suffix):
        return f"{self.dir}/runs/{run['name']}{suffix}"

    def member(self, tree, i):
        kind = tree[0]
        if kind == "L":
            a, b = int(self.row[i]), int(self.row[i + 1])
            at = a + int(np.searchsorted(self.labels[a:b], tree[1]))
            return at < b and int(self.labels[at]) == tree[1]
        if kind == "R":
            return tree[2] <= int(self.values[i]) < tree[3]
        if kind == "AND":
            return all(self.member(t, i) for t in tree[1])
        if kind == "OR":
            return any(self.member(t, i) for t in tree[1])
        if kind == "NOT":
            return not self.member(tree[1], i)
        raise ValueError(f"unsupported filter {kind}")

    def recall(self, run, ids, k=10):
        with open(self.path(run, ".gtpf"), "rb") as f:
            nq, truth_k = (int(x) for x in np.fromfile(f, dtype=np.uint32, count=2))
            np.fromfile(f, dtype=np.uint32, count=nq * truth_k)
            kth = np.fromfile(f, dtype=np.float32, count=nq * truth_k).reshape(nq, truth_k)[:, k - 1].astype(np.float64)
        queries = np.fromfile(self.path(run, "_query.fbin"), dtype=np.float32, offset=8).reshape(nq, self.dim)
        filters = [parse(e) for e in read_lines(self.path(run, ".expr"))]
        scale = float(np.abs(kth).mean())
        hits = 0
        for q in range(nq):
            found = [int(i) for i in ids[q, :k] if int(i) < self.n and self.member(filters[q], int(i))]
            if found:
                d = ((np.asarray(self.base[np.sort(found)], dtype=np.float32) - queries[q]) ** 2).sum(axis=1, dtype=np.float32)
                hits += min(k, int((d.astype(np.float64) <= kth[q] * (1 + 1e-5) + 1e-6 * scale).sum()))
        return hits / (nq * k)


def read_ids(path):
    with open(path, "rb") as f:
        nq, k = (int(x) for x in np.fromfile(f, dtype=np.int32, count=2))
        return np.fromfile(f, dtype=np.uint32, count=nq * k).reshape(nq, k)


def emit(log, **row):
    log.write(json.dumps(row) + "\n")
    log.flush()
    print(row["run"], row["setting"], f"recall={row['recall']:.4f} qps={row['qps']:.1f} reads={row['reads']:.1f}", flush=True)


def run_kuroann(a, data, log):
    for target in ("90", "95", "99"):
        for r in data.runs:
            out = subprocess.run([a.bin, "search", "--index", a.index, "--queries", data.path(r, "_query.fbin"), "--filters", data.path(r, ".expr"),
                                  "--plan", f"{a.plans}/plan_{target}.txt", "--groundtruth", data.path(r, ".gtpf"), "--threads", str(a.threads), "--rounds", "3"],
                                 check=True, capture_output=True, text=True).stdout
            m = KUROANN_ROW.search(out.strip().splitlines()[-1])
            emit(log, run=r["name"], setting=f"plan{target}", recall=float(m[3]), qps=float(m[1]), reads=float(m[2]), queries=r["n"])


def by_band(runs):
    bands = {}
    for r in runs:
        bands.setdefault(r["band"], []).append(r)
    return [bands[b] for b in sorted(bands)]


def run_pipeann(a, data, log):
    env = dict(os.environ)
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(a.out))) as tmp:
        for runs in by_band(data.runs):
            for lists in PIPEANN_LISTS:
                ended = True  # all runs of a band get the same settings, because plot.py pools them
                for r in runs:
                    argv = [a.bin, "float", a.index, str(a.threads), "4", data.path(r, "_query.fbin"), data.path(r, ".gtpf"), "10", "l2", "pq", data.path(r, ".pipeann.json"), "0"] + [str(x) for x in lists]
                    env.pop("PF_RESULT_PREFIX", None)
                    subprocess.run(argv, env=env, check=True, capture_output=True, text=True)
                    env["PF_RESULT_PREFIX"] = f"{tmp}/{r['name']}"
                    stdout = subprocess.run(argv, env=env, check=True, capture_output=True, text=True).stdout
                    rows = []
                    for line in stdout.splitlines():
                        m = PIPEANN_ROW.match(line)
                        v = NUMBER.findall(m[3]) if m else []
                        if len(v) >= 4:  # QPS, mean latency, counters whose wide columns can touch each other, mean reads, recall
                            rows.append((int(m[1]), float(v[0]), float(v[-2])))
                    if len(rows) != len(lists):
                        raise SystemExit(f"{r['name']}: {len(rows)} result rows for {len(lists)} settings")
                    stop = False
                    for L, qps, reads in rows:
                        path = f"{tmp}/{r['name']}_L{L}.bin"
                        recall = data.recall(r, read_ids(path))
                        os.unlink(path)
                        emit(log, run=r["name"], setting=f"L{L}", recall=recall, qps=qps, reads=reads, queries=r["n"])
                        stop |= recall >= RECALL_END or qps < QPS_END
                    ended &= stop
                if ended:
                    break


def run_faiss(a, data, log):
    partitions = int(np.fromfile(a.index + ".meta", dtype=np.int32)[2])
    scale = partitions / 32768
    probes = [0] + sorted({max(8, int(round(x * scale))) for x in FAISS_PROBES})
    argv = [a.bin, "--index", a.index, "--labels", f"{a.data}/base.spmat", "--range", f"{data.field}={a.data}/{data.field}.u32", "--threads", str(a.threads)]
    server = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    for line in server.stdout:
        if line.startswith("READY"):
            break
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(a.out))) as tmp:
        for runs in by_band(data.runs):
            for nprobe in probes:
                large = runs[0]["band"] >= 3 and nprobe * 32 >= partitions
                for pool in FAISS_POOLS + (FAISS_LARGE_POOLS if large else ()):
                    ended = True
                    for r in runs:
                        prefix = f"{tmp}/{r['name']}"
                        # the first pass warms up, and the second is the measurement
                        server.stdin.write(f"run {data.path(r, '_query.fbin')} {data.path(r, '.expr')} {prefix} {nprobe}:{pool},{nprobe}:{pool}\n")
                        server.stdin.flush()
                        rows = []
                        for line in server.stdout:
                            if line.startswith("END"):
                                break
                            m = FAISS_ROW.match(line.strip())
                            if m:
                                rows.append(m)
                        if len(rows) != 2 or int(rows[1][9]):
                            raise SystemExit(f"{r['name']} {nprobe}:{pool}: bad result")
                        path = f"{prefix}_np{nprobe}_L{pool}.bin"
                        recall = data.recall(r, read_ids(path))
                        os.unlink(path)
                        qps = float(rows[1][4])
                        emit(log, run=r["name"], setting=f"np{nprobe}_L{pool}", recall=recall, qps=qps, reads=float(rows[1][8]), queries=r["n"])
                        ended &= qps < QPS_END
                    if ended:
                        break
    server.stdin.write("quit\n")
    server.stdin.flush()
    server.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system", choices=["kuroann", "pipeann", "faiss"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--plans")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    a = ap.parse_args()
    data = Data(a.data)
    with open(a.out + ".tmp", "w") as log:
        {"kuroann": run_kuroann, "pipeann": run_pipeann, "faiss": run_faiss}[a.system](a, data, log)
    os.replace(a.out + ".tmp", a.out)


if __name__ == "__main__":
    main()
