#!/usr/bin/env python3
"""Compare the systems per filter-width band, and draw the figure of the README.

KuroANN serves a band with its plan for each target. A baseline serves it with one setting, its fastest whose recall reaches KuroANN's.
Within a band, recall is weighted by queries, and throughput is the total queries over the total time, over the filter forms of the band.

usage: plot.py RESULTS_DIR
Reads RESULTS_DIR/<dataset>/<system>.jsonl, and writes RESULTS_DIR/summary.csv and RESULTS_DIR/throughput_by_band.svg.
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATASETS = [("pubmed", "PubMed-24M"), ("amazon", "Amazon-35M")]
BANDS = ["<0.01%", "0.01–0.1%", "0.1–1%", "1–5%", "5–20%", ">20%"]
TARGETS = ["90", "95", "99"]
SYSTEMS = {"kuroann": ("KuroANN", "#0062CC", "o", "-"), "pipeann-filter": ("PipeANN-Filter", "#C05A8A", "s", "--"), "faiss-ssd": ("Faiss-SSD", "#7F7F7F", "D", ":")}


def bands_of(path):
    """{band: {setting: (recall, qps)}}, over the settings measured on every run of the band."""
    runs = defaultdict(dict)
    for line in open(path):
        x = json.loads(line)
        runs[x["run"]][x["setting"]] = x
    grouped = defaultdict(list)
    for run in runs:
        grouped[int(re.match(r"eval_b(\d+)_", run)[1])].append(run)
    out = {}
    for band, names in grouped.items():
        out[band] = {}
        for setting in set.intersection(*[set(runs[r]) for r in names]):
            rows = [runs[r][setting] for r in names]
            total = sum(x["queries"] for x in rows)
            out[band][setting] = (sum(x["queries"] * x["recall"] for x in rows) / total, total / sum(x["queries"] / x["qps"] for x in rows))
    return out


def summarize(directory):
    kuroann = bands_of(f"{directory}/kuroann.jsonl")
    rows = []
    for band in sorted(kuroann):
        for target in TARGETS:
            recall, qps = kuroann[band][f"plan{target}"]
            row = dict(band=BANDS[band], band_index=band, target=target, recall=recall, kuroann=qps)
            for system in ("pipeann-filter", "faiss-ssd"):
                path = f"{directory}/{system}.jsonl"
                if os.path.exists(path):
                    points = bands_of(path).get(band, {})
                    reached = [q for r, q in points.values() if r >= recall]
                    row[system] = max(reached) if reached else None
            rows.append(row)
    return rows


def main():
    results = sys.argv[1]
    present = [(key, title) for key, title in DATASETS if os.path.exists(f"{results}/{key}/kuroann.jsonl")]
    fig, axes = plt.subplots(len(present), 3, figsize=(12, 2.6 * len(present)), squeeze=False, sharex=True)
    with open(f"{results}/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "band", "target", "kuroann_recall", "kuroann_qps", "pipeann_filter_qps", "faiss_ssd_qps"])
        for i, (key, title) in enumerate(present):
            rows = summarize(f"{results}/{key}")
            for row in rows:
                values = [row.get(s) for s in ("pipeann-filter", "faiss-ssd")]
                w.writerow([key, row["band"], row["target"], f"{row['recall']:.4f}", f"{row['kuroann']:.0f}"] + ["" if v is None else f"{v:.0f}" for v in values])
                print(key, row["band"], row["target"], f"recall {row['recall']:.4f} KuroANN {row['kuroann']:.0f}", values, flush=True)
            for j, target in enumerate(TARGETS):
                ax = axes[i][j]
                part = [r for r in rows if r["target"] == target]
                x = [r["band_index"] for r in part]
                for system, (label, color, marker, line) in SYSTEMS.items():
                    y = [r["kuroann"] if system == "kuroann" else r.get(system) for r in part]
                    ax.plot(x, [np.nan if v is None else v for v in y], color=color, marker=marker, linestyle=line, label=label, linewidth=1.6, markersize=5)
                ax.set_yscale("log")
                ax.grid(True, which="major", color="#E3E6EA", linewidth=0.6)
                ax.set_title(f"{title}, recall {int(target) / 100:.2f}", fontsize=10)
                ax.set_xticks(range(len(BANDS)))
                ax.set_xticklabels(BANDS, fontsize=7)
                if j == 0:
                    ax.set_ylabel("Queries per second", fontsize=9)
                if i == len(present) - 1:
                    ax.set_xlabel("Vectors that match the filter", fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{results}/throughput_by_band.svg")


if __name__ == "__main__":
    main()
