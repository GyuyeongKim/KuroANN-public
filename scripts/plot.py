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
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = [("pubmed", "PubMed-24M"), ("amazon", "Amazon-35M")]
BANDS = ["<0.01%", "0.01–0.1%", "0.1–1%", "1–5%", "5–20%", ">20%"]
TARGETS = ["90", "95", "99"]
SYSTEMS = {"kuroann": ("KuroANN", "#E879B0", ""), "pipeann-filter": ("PipeANN-Filter", "#1E1B24", ""), "faiss-ssd": ("Faiss-SSD", "#B4ACC2", "")}


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
    plt.rcParams.update({"font.family": ["Helvetica", "Arial", "DejaVu Sans"], "svg.fonttype": "path"})
    fig, axes = plt.subplots(len(present), 3, figsize=(12, 3.6 * len(present)), squeeze=False, sharey="row")
    width = 0.27
    with open(f"{results}/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "band", "target", "kuroann_recall", "kuroann_qps", "pipeann_filter_qps", "faiss_ssd_qps"])
        for i, (key, title) in enumerate(present):
            rows = summarize(f"{results}/{key}")
            for row in rows:
                values = [row.get(s) for s in ("pipeann-filter", "faiss-ssd")]
                w.writerow([key, row["band"], row["target"], f"{row['recall']:.4f}", f"{row['kuroann']:.0f}"] + ["" if v is None else f"{v:.0f}" for v in values])
                print(key, row["band"], row["target"], f"recall {row['recall']:.4f} KuroANN {row['kuroann']:.0f}", values, flush=True)
            low = min(v for r in rows for v in [r["kuroann"]] + [r.get(s) for s in ("pipeann-filter", "faiss-ssd")] if v)
            high = max(r["kuroann"] for r in rows)
            for j, target in enumerate(TARGETS):
                ax = axes[i][j]
                part = [r for r in rows if r["target"] == target]
                for k, (system, (label, color, _)) in enumerate(SYSTEMS.items()):
                    if system != "kuroann" and not any(system in r for r in part):   # not measured in this run
                        continue
                    for r in part:
                        x = r["band_index"] + (k - 1) * width
                        v = r["kuroann"] if system == "kuroann" else r.get(system)
                        if v is None:   # no setting of this system reached KuroANN's recall in this band
                            ax.plot([x], [low / 1.6], marker="x", color=color, markersize=6, markeredgewidth=1.6, clip_on=False, zorder=4)
                        else:
                            ax.bar([x], [v], width, color=color, edgecolor="white", linewidth=0.4, zorder=3,
                                   label=label if r is part[0] else None)
                ax.set_yscale("log")
                ax.set_ylim(low / 2, high * 1.6)
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.grid(True, axis="y", which="major", color="#E5E3E9", linewidth=0.6, zorder=0)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                ax.set_title(f"{title}, target recall {int(target) / 100:.2f}", fontsize=12)
                ax.set_xticks(range(len(BANDS)))
                ax.set_xticklabels([b.replace("%", "") for b in BANDS], fontsize=10)
                ax.tick_params(axis="y", labelsize=12)
                if j == 0:
                    ax.set_ylabel("Throughput (QPS)", fontsize=11)
                ax.set_xlabel("Vectors that match the filter (%)", fontsize=11)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if any(row.get(s, 0) is None for key, _ in present for row in summarize(f"{results}/{key}") for s in SYSTEMS):
        handles.append(plt.Line2D([], [], marker="x", color="#4B3B63", linestyle="none", markersize=6, markeredgewidth=1.6))
        labels.append("Recall of KuroANN not reached")
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{results}/throughput_by_band.svg", facecolor="white")


if __name__ == "__main__":
    main()
