#!/usr/bin/env python3
"""Prepare a dataset for scripts/reproduce.sh: vectors, labels, a range field, filtered queries, and their exact answers.

pubmed: Qdrant/PubMed-MV on Hugging Face (BGE-M3 dense vectors of PubMed abstracts and 10,000 bioRxiv titles as queries), with the
year, publication types, and MeSH terms of each abstract from the NLM baseline.

amazon: Amazon Reviews'23 item metadata, embedded with BLaIR (one single-thread process per core and per 200,000 items), with
categories, ratings, and prices as filters, and the mean of a shopper's recent items from the 5-core test split as the query.

A query keeps one filter, drawn with a fixed seed. 30% of the queries calibrate the plan, and the rest are the evaluation runs,
one per filter-width band and filter form. Every step keeps its output, so running the script again only does what is missing.

Writes to DIR: base.fbin, base.spmat, labels.json, <field>.u32, <field>.bin, and runs/. For each run, runs/ holds <name>_query.fbin,
<name>.expr, and for PipeANN-Filter, <name>_qt.spmat, <name>_qr.spmat, <name>.pipeann.json. The calibration queries are in
runs/calibration.fbin and runs/calibration.expr.

usage: datasets.py pubmed|amazon DIR [--limit N] [--workers W]
"""
import argparse
import csv
import glob
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import re
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict

import numpy as np

EDGES = [0, 0.0001, 0.001, 0.01, 0.05, 0.2, 1.01]
SEED = 42
PIPEANN_PREFIX = "pipeann"


def done(path):
    return os.path.exists(path)


def write_spmat(path, columns, row, labels, values=None):
    with open(path + ".tmp", "wb") as f:
        np.array([len(row) - 1, columns, len(labels)], dtype=np.int64).tofile(f)
        np.asarray(row, dtype=np.int64).tofile(f)
        np.asarray(labels, dtype=np.int32).tofile(f)
        (np.ones(len(labels), dtype=np.float32) if values is None else np.asarray(values, dtype=np.float32)).tofile(f)
    os.replace(path + ".tmp", path)


class Filters:
    """Label lists and the range field, by vector id, to count the matches of a filter."""

    def __init__(self, directory, field):
        rows, columns, entries = (int(x) for x in np.fromfile(f"{directory}/base.spmat", dtype=np.int64, count=3))
        row = np.memmap(f"{directory}/base.spmat", dtype=np.int64, mode="r", offset=24, shape=(rows + 1,))
        labels = np.memmap(f"{directory}/base.spmat", dtype=np.int32, mode="r", offset=24 + 8 * (rows + 1), shape=(entries,))
        self.n = rows
        self.start = np.r_[0, np.cumsum(np.bincount(labels, minlength=columns))]
        order = np.argsort(labels, kind="stable")
        owners = np.repeat(np.arange(rows, dtype=np.uint32), np.diff(row))
        self.ids = owners[order]
        self.values = np.fromfile(f"{directory}/{field}.u32", dtype=np.uint32)
        self.cache = {}

    def members(self, label):
        return self.ids[self.start[label]:self.start[label + 1]]

    def count_label(self, label):
        return int(self.start[label + 1] - self.start[label])

    def count_range(self, lo, hi):
        key = ("R", lo, hi)
        if key not in self.cache:
            self.cache[key] = int(((self.values >= lo) & (self.values < hi)).sum())
        return self.cache[key]

    def count_label_range(self, label, lo, hi):
        key = ("LR", label, lo, hi)
        if key not in self.cache:
            v = self.values[self.members(label)]
            self.cache[key] = int(((v >= lo) & (v < hi)).sum())
        return self.cache[key]

    def count_labels(self, a, b):
        key = ("LL", a, b)
        if key not in self.cache:
            self.cache[key] = len(np.intersect1d(self.members(a), self.members(b), assume_unique=True))
        return self.cache[key]


# PubMed


# Downloads are pinned to the versions that the results in the README were measured with.
PUBMED_REPO = "datasets/Qdrant/PubMed-MV@1c1d90c20a9f16ecfb1dfd3c5fe661801f16b58d"
PUBMED_COLUMNS = ["PMID", "bge_m3_dense_embedding"]
PUBMED_QUERIES = PUBMED_REPO + "/queries/bf_queries_dense_cosine_k1000.parquet"
PUBMED_DIM = 1024
NLM_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline"
# NLM replaces its baseline every December and removes the previous one, so a later year can be chosen with NLM_BASELINE.
NLM_BASELINE = os.environ.get("NLM_BASELINE", "pubmed26n")
PUBLICATION_TYPES = ["Randomized Controlled Trial", "Clinical Trial", "Review", "Systematic Review", "Meta-Analysis", "Observational Study", "Case Reports",
                     "Multicenter Study", "Comparative Study", "Practice Guideline", "Evaluation Study", "Validation Study"]
# MeSH check tags say little about a topic.
CHECK_TAGS = {"Humans", "Animals", "Male", "Female", "Adult", "Aged", "Middle Aged", "Aged, 80 and over", "Young Adult", "Adolescent", "Child", "Child, Preschool",
              "Infant", "Infant, Newborn", "Mice", "Rats", "Pregnancy", "Retrospective Studies", "Prospective Studies", "Cross-Sectional Studies", "Treatment Outcome",
              "Risk Factors", "Time Factors", "Cohort Studies", "Follow-Up Studies", "Surveys and Questionnaires"}
YEARS = [2023, 2022, 2020, 2018, 2015, 2010, 2000]


def fetch_pubmed_file(job):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    k, path, out = job
    vectors, ids = f"{out}/parts/{k}.npy", f"{out}/parts/{k}.pmid.npy"
    if done(vectors) and done(ids):
        return k, int(np.load(ids, mmap_mode="r").shape[0])
    fs = HfFileSystem()
    for attempt in range(8):
        try:
            with fs.open(path, "rb", block_size=4 * 1024 * 1024) as f:
                table = pq.ParquetFile(f).read(columns=PUBMED_COLUMNS)
            flat = table.column(PUBMED_COLUMNS[1]).combine_chunks().values.to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
            if len(flat) != table.num_rows * PUBMED_DIM:
                raise ValueError(f"{path}: rows of different lengths")
            np.save(vectors + ".tmp.npy", flat.reshape(table.num_rows, PUBMED_DIM))
            np.save(ids + ".tmp.npy", table.column(PUBMED_COLUMNS[0]).to_numpy().astype(np.int64))
            os.replace(vectors + ".tmp.npy", vectors)
            os.replace(ids + ".tmp.npy", ids)
            return k, table.num_rows
        except Exception as e:  # network errors and rate limits
            print(f"retry {k} attempt {attempt}: {e!r}", file=sys.stderr, flush=True)
            time.sleep(min(120, 5 * 2 ** attempt))
    raise RuntimeError(f"cannot fetch {path}")


def fetch_pubmed(out, limit, workers):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    if not done(f"{out}/base.fbin"):
        os.makedirs(f"{out}/parts", exist_ok=True)
        fs = HfFileSystem()
        files = []
        for shard in range(10):
            files += [e["name"] for e in fs.ls(f"{PUBMED_REPO}/data/{shard:03d}", detail=True) if e["name"].endswith(".parquet")]
        files = sorted(files)
        random.Random(SEED).shuffle(files)
        # Files hold about 3,000 rows. A limit takes a prefix of the shuffled files.
        if limit:
            files = files[:min(len(files), limit // 2000 + 1)]
        print(f"{len(files)} files", flush=True)
        with mp.Pool(workers) as pool:
            rows = dict(pool.imap_unordered(fetch_pubmed_file, [(k, p, out) for k, p in enumerate(files)]))
        total = min(limit, sum(rows.values())) if limit else sum(rows.values())
        written = 0
        pmids = []
        with open(f"{out}/base.fbin.tmp", "wb") as f:
            np.array([total, PUBMED_DIM], dtype=np.uint32).tofile(f)
            for k in range(len(files)):
                take = min(rows[k], total - written)
                vectors = np.load(f"{out}/parts/{k}.npy")[:take]
                if np.any(np.abs(np.linalg.norm(vectors, axis=1) - 1) > 1e-3):
                    raise SystemExit(f"file {k}: vectors are not unit length")
                vectors.tofile(f)
                pmids.append(np.load(f"{out}/parts/{k}.pmid.npy")[:take])
                written += take
        np.save(f"{out}/pmid.npy", np.concatenate(pmids))
        os.replace(f"{out}/base.fbin.tmp", f"{out}/base.fbin")
        subprocess.run(["rm", "-rf", f"{out}/parts"], check=True)
        print(f"base.fbin: {written} vectors", flush=True)
    if not done(f"{out}/query.fbin"):
        with HfFileSystem().open(PUBMED_QUERIES, "rb", block_size=8 * 1024 * 1024) as f:
            table = pq.ParquetFile(f).read(columns=["hit_ids", "bge_m3_dense_embedding"])
        q = table.column("bge_m3_dense_embedding").combine_chunks().values.to_numpy(zero_copy_only=False).astype(np.float32).reshape(table.num_rows, PUBMED_DIM)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        pq.write_table(table.drop_columns(["bge_m3_dense_embedding"]), f"{out}/queries.parquet")
        with open(f"{out}/query.fbin.tmp", "wb") as f:
            np.array(q.shape, dtype=np.uint32).tofile(f)
            q.tofile(f)
        os.replace(f"{out}/query.fbin.tmp", f"{out}/query.fbin")


def fetch_nlm_file(job):
    url, directory, name = job
    path = f"{directory}/{name}"
    if done(path):
        return
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{url}/{name}", timeout=600) as r:
                data = r.read()
            with urllib.request.urlopen(f"{url}/{name}.md5", timeout=60) as r:
                expected = re.search(r"[0-9a-f]{32}", r.read().decode()).group(0)
            if hashlib.md5(data).hexdigest() != expected:
                raise ValueError("checksum differs")
            with open(path + ".tmp", "wb") as f:
                f.write(data)
            os.replace(path + ".tmp", path)
            return
        except Exception as e:
            print(f"retry {name} attempt {attempt}: {e!r}", file=sys.stderr, flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"cannot fetch {name}")


def year_of(article):
    for path in ("Journal/JournalIssue/PubDate/Year", "ArticleDate/Year"):
        y = article.findtext(path)
        if y and y[:4].isdigit():
            return int(y[:4])
    m = article.findtext("Journal/JournalIssue/PubDate/MedlineDate")
    if m:
        g = re.search(r"(1[89]|20)\d\d", m)
        if g:
            return int(g.group(0))
    return 0


def parse_nlm_file(path):
    from lxml import etree

    rows = []
    with gzip.open(path, "rb") as f:
        for _, e in etree.iterparse(f, tag="PubmedArticle"):
            citation = e.find("MedlineCitation")
            article = citation.find("Article")
            rows.append(dict(pmid=int(citation.findtext("PMID")), year=year_of(article),
                             pubtypes=[p.text for p in article.iterfind("PublicationTypeList/PublicationType")],
                             mesh=[h.find("DescriptorName").text for h in citation.iterfind("MeshHeadingList/MeshHeading")]))
            e.clear()
            while e.getprevious() is not None:
                del e.getparent()[0]
    return rows


def fetch_nlm(directory, workers):
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = f"{directory}/nlm.parquet"
    if done(table):
        return table
    nlm = f"{directory}/nlm"
    os.makedirs(nlm, exist_ok=True)
    with urllib.request.urlopen(NLM_URL + "/", timeout=120) as r:
        names = sorted(n for n in set(re.findall(r"pubmed[0-9a-z]+\.xml\.gz", r.read().decode())) if n.startswith(NLM_BASELINE))
    if not names:
        raise SystemExit(f"the NLM baseline {NLM_BASELINE} is no longer at {NLM_URL}. Set NLM_BASELINE to the current one; its years, "
                         "publication types, and MeSH terms differ, so the filters and results will differ from the README.")
    with mp.Pool(min(workers, 4)) as pool:
        list(pool.imap_unordered(fetch_nlm_file, [(NLM_URL, nlm, name) for name in names]))
    schema = pa.schema([("pmid", pa.int64()), ("year", pa.int32()), ("pubtypes", pa.list_(pa.string())), ("mesh", pa.list_(pa.string()))])
    writer = pq.ParquetWriter(table + ".tmp", schema, compression="zstd")
    with mp.Pool(workers) as pool:
        for rows in pool.imap(parse_nlm_file, sorted(glob.glob(f"{nlm}/pubmed*.xml.gz"))):
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    writer.close()
    os.replace(table + ".tmp", table)
    subprocess.run(["rm", "-rf", nlm], check=True)
    return table


def label_pubmed(out, table, min_count=10):
    """Labels are publication types and MeSH terms, ordered by count. A later NLM file wins for a repeated PMID."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if done(f"{out}/base.spmat"):
        return
    pmid = np.load(f"{out}/pmid.npy")
    n = len(pmid)
    meta = pq.read_table(table, columns=["pmid", "year", "pubtypes", "mesh"])
    meta_ids = meta.column("pmid").to_numpy()
    order = np.argsort(meta_ids, kind="stable")
    order = order[np.r_[meta_ids[order][1:] != meta_ids[order][:-1], True]]
    sorted_ids = meta_ids[order]
    at = np.minimum(np.searchsorted(sorted_ids, pmid), len(sorted_ids) - 1)
    found = sorted_ids[at] == pmid
    rows = order[at]
    print(f"{int(found.sum())} of {n} vectors have an NLM record", flush=True)
    year = meta.column("year").to_numpy().astype(np.uint32)[rows]
    year[~found] = 0

    def flatten(column, prefix):
        a = column.cast(pa.list_(pa.large_string())).combine_chunks()
        d = a.values.dictionary_encode()
        return a.offsets.to_numpy().astype(np.int64), d.indices.to_numpy().astype(np.int32), [prefix + s for s in d.dictionary.to_pylist()]

    pt_offset, pt_index, pt_names = flatten(meta.column("pubtypes"), "PT:")
    mh_offset, mh_index, mh_names = flatten(meta.column("mesh"), "MH:")
    names = pt_names + mh_names

    def take(offset, index, shift):
        lengths = offset[rows + 1] - offset[rows]
        lengths[~found] = 0
        ptr = np.zeros(n + 1, dtype=np.int64)
        ptr[1:] = np.cumsum(lengths)
        flat = np.empty(int(lengths.sum()), dtype=np.int32)
        first = offset[rows]
        for i in np.nonzero(lengths)[0]:
            flat[ptr[i]:ptr[i + 1]] = index[first[i]:first[i] + lengths[i]]
        return ptr, flat + shift

    pp, pf = take(pt_offset, pt_index, 0)
    mp2, mf = take(mh_offset, mh_index, len(pt_names))
    counts = np.bincount(np.r_[pf, mf], minlength=len(names))
    kept = np.nonzero(counts >= min_count)[0]
    kept = kept[np.argsort(-counts[kept], kind="stable")]
    new = -np.ones(len(names), dtype=np.int32)
    new[kept] = np.arange(len(kept), dtype=np.int32)

    lengths = np.diff(pp) + np.diff(mp2)
    owners = np.repeat(np.arange(n, dtype=np.int64), lengths)
    ids = np.empty(int(lengths.sum()), dtype=np.int32)
    starts = np.zeros(n + 1, dtype=np.int64)
    starts[1:] = np.cumsum(lengths)
    pt_lengths = np.diff(pp)
    for i in np.nonzero(lengths)[0]:
        a, b = starts[i], starts[i] + pt_lengths[i]
        ids[a:b] = pf[pp[i]:pp[i + 1]]
        ids[b:starts[i + 1]] = mf[mp2[i]:mp2[i + 1]]
    ids = new[ids]
    keep = ids >= 0
    owners, ids = owners[keep], ids[keep]
    o = np.lexsort((ids, owners))
    owners, ids = owners[o], ids[o]
    unique = np.r_[True, (owners[1:] != owners[:-1]) | (ids[1:] != ids[:-1])]
    owners, ids = owners[unique], ids[unique]
    row = np.zeros(n + 1, dtype=np.int64)
    row[1:] = np.cumsum(np.bincount(owners, minlength=n))
    json.dump({"names": [names[i] for i in kept], "counts": np.bincount(ids, minlength=len(kept)).tolist()}, open(f"{out}/labels.json", "w"))
    write_range(out, "year", year)
    write_spmat(f"{out}/base.spmat", len(kept), row, ids)


def write_range(out, field, values):
    values.tofile(f"{out}/{field}.u32")
    with open(f"{out}/{field}.bin", "wb") as f:
        np.array([len(values), 1], dtype=np.int32).tofile(f)
        values.tofile(f)


def workload_pubmed(out):
    """One filter per query. A MeSH term comes from the query's own top 100 unfiltered neighbors, weighted by how many carry it."""
    import pyarrow.parquet as pq

    path = f"{out}/workload.json"
    if done(path):
        return json.load(open(path))
    rng = np.random.default_rng(SEED)
    names = json.load(open(f"{out}/labels.json"))["names"]
    label_of = {name: i for i, name in enumerate(names)}
    filters = Filters(out, "year")
    n = filters.n
    types = [label_of["PT:" + p] for p in PUBLICATION_TYPES if "PT:" + p in label_of]
    skip = {label_of["MH:" + c] for c in CHECK_TAGS if "MH:" + c in label_of}
    row = np.memmap(f"{out}/base.spmat", dtype=np.int64, mode="r", offset=24, shape=(n + 1,))
    labels = np.memmap(f"{out}/base.spmat", dtype=np.int32, mode="r", offset=24 + 8 * (n + 1), shape=(int(row[n]),))
    pmid = np.load(f"{out}/pmid.npy")
    order = np.argsort(pmid, kind="stable")
    sorted_pmid = pmid[order]
    hits = pq.read_table(f"{out}/queries.parquet", columns=["hit_ids"]).column("hit_ids").to_pylist()

    def mesh_of(i):
        h = np.array([int(x) for x in hits[i][:100]], dtype=np.int64)
        at = np.searchsorted(sorted_pmid, h)
        at[at >= n] = n - 1
        rows = order[at][sorted_pmid[at] == h]
        if len(rows) == 0:
            return None
        ids = np.concatenate([labels[row[r]:row[r + 1]] for r in rows])
        u, c = np.unique(ids, return_counts=True)
        ok = np.array([names[x].startswith("MH:") and x not in skip for x in u], dtype=bool)
        if not ok.any():
            return None
        u, c = u[ok], c[ok].astype(float)
        return int(rng.choice(u, p=c / c.sum()))

    forms = ["pt", "year", "pt_year", "mesh", "mesh_year", "mesh_pt"]
    out_rows = []
    for i in range(len(hits)):
        f = forms[int(rng.integers(len(forms)))]
        pt = int(rng.choice(types))
        y = int(rng.choice(YEARS))
        m = mesh_of(i) if "mesh" in f else None
        if "mesh" in f and m is None:
            f = "pt"
        if f == "pt":
            e, form, tags, bounds, count = f"L {pt}", "labels", [pt], None, filters.count_label(pt)
        elif f == "year":
            e, form, tags, bounds, count = f"R year {y} 3000", "range", None, [y, 3000], filters.count_range(y, 3000)
        elif f == "pt_year":
            e, form, tags, bounds, count = f"AND 2 L {pt} R year {y} 3000", "labelrange", [pt], [y, 3000], filters.count_label_range(pt, y, 3000)
        elif f == "mesh":
            e, form, tags, bounds, count = f"L {m}", "labels", [m], None, filters.count_label(m)
        elif f == "mesh_year":
            e, form, tags, bounds, count = f"AND 2 L {m} R year {y} 3000", "labelrange", [m], [y, 3000], filters.count_label_range(m, y, 3000)
        else:
            count = filters.count_label(m) if m == pt else filters.count_labels(*sorted((m, pt)))
            e, form, tags, bounds = f"AND 2 L {m} L {pt}", "labels", sorted({m, pt}), None
        if count >= 10:
            out_rows.append(dict(q=i, expr=e, form=form, tags=tags, bounds=bounds, count=count))
    return finish_workload(out, rng, n, out_rows, "year", f"{out}/query.fbin")


# Amazon


AMAZON_REPO = "McAuley-Lab/Amazon-Reviews-2023"
AMAZON_REVISION = "2b6d039ed471f2ba5fd2acb718bf33b0a7e5598e"
AMAZON_MODEL = "hyp1231/blair-roberta-base"
AMAZON_MODEL_REVISION = "88adfe2b621cb202dee9aabb19b59de8f622844c"
AMAZON_BLOCK = 200000
AMAZON_BUDGETS = [10, 25, 50, 100, 200, 500]
AMAZON_HISTORY = 10
AMAZON_QUERIES = 10000
AMAZON_ITEMS = 35077399  # items with a title in the full metadata


def price_of(v):
    if isinstance(v, (int, float)):
        return float(v) if v > 0 else math.nan
    if isinstance(v, str):
        try:
            x = float(v.replace("$", "").replace(",", "").strip())
            return x if x > 0 else math.nan
        except ValueError:
            return math.nan
    return math.nan


def text_of(d):
    parts = [d["title"]] + (d.get("features") or []) + (d.get("description") or [])
    return " ".join(p for p in parts if isinstance(p, str) and p)[:2000]


def amazon_users(out, rng):
    """A fixed sample of twice the query count from the shoppers of the 5-core test split."""
    csv.field_size_limit(1 << 28)
    users, seen = [], 0
    for csv_path in sorted(glob.glob(f"{out}/source/benchmark/5core/last_out_w_his/*.test.csv")):
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                seen += 1
                if len(users) < AMAZON_QUERIES * 2:
                    users.append(r)
                else:
                    j = int(rng.integers(seen))
                    if j < AMAZON_QUERIES * 2:
                        users[j] = r
    return users


def parse_amazon(out, limit):
    """One row per item with a title, in the sorted order of the files. A repeated parent_asin keeps its first row.
    A limit keeps the items of the first sampled shoppers, about half of the limit, so that the workload has queries, and a uniform
    sample of the other items for the rest."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = f"{out}/items.parquet"
    if done(table):
        return table
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=AMAZON_REPO, repo_type="dataset", revision=AMAZON_REVISION, local_dir=f"{out}/source", max_workers=16,
                      allow_patterns=["raw/meta_categories/*.jsonl", "benchmark/5core/last_out_w_his/*.test.csv"])
    keep, threshold = None, 0
    if limit:
        users = amazon_users(out, np.random.default_rng(SEED))[:max(1, limit // (2 * (AMAZON_HISTORY + 1)))]
        keep = {a for u in users for a in u["history"].split()[-AMAZON_HISTORY:] + [u["parent_asin"]]}
        threshold = max(0, limit - len(keep)) / AMAZON_ITEMS * 2 ** 32
    schema = pa.schema([("asin", pa.string()), ("cat", pa.string()), ("price", pa.float32()), ("rating", pa.float32()), ("nrating", pa.int32()),
                        ("path", pa.list_(pa.string())), ("text", pa.string())])
    writer = pq.ParquetWriter(table + ".tmp", schema, compression="zstd")
    seen = set()
    for path in sorted(glob.glob(f"{out}/source/raw/meta_categories/meta_*.jsonl")):
        category = os.path.basename(path)[5:-6]
        rows = []
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                title = d.get("title")
                if not isinstance(title, str) or not title.strip() or d["parent_asin"] in seen:
                    continue
                if keep is not None and d["parent_asin"] not in keep and int(hashlib.md5(d["parent_asin"].encode()).hexdigest()[:8], 16) >= threshold:
                    continue
                seen.add(d["parent_asin"])
                rows.append(dict(asin=d["parent_asin"], cat=category, price=price_of(d.get("price")), rating=d.get("average_rating") or 0.0,
                                 nrating=d.get("rating_number") or 0, path=[c for c in (d.get("categories") or []) if isinstance(c, str)], text=text_of(d)))
                if len(rows) == AMAZON_BLOCK:
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                    rows = []
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    writer.close()
    os.replace(table + ".tmp", table)
    return table


EMBED_BLOCK = """
import os, sys
import numpy as np, pyarrow.parquet as pq, torch
from transformers import AutoTokenizer, AutoModel
table, out, model_name, revision, block, size = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
pf = pq.ParquetFile(table)
lo, hi = block * size, min((block + 1) * size, pf.metadata.num_rows)
texts, pos = [], 0
for g in range(pf.metadata.num_row_groups):
    n = pf.metadata.row_group(g).num_rows
    if pos + n > lo and pos < hi:
        texts += pf.read_row_group(g, columns=["text"]).column(0).to_pylist()[max(lo - pos, 0):hi - pos]
    pos += n
tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
model = AutoModel.from_pretrained(model_name, revision=revision).eval()
order = np.argsort([len(t) for t in texts])
vectors = None
with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
    for i in range(0, len(texts), 16):
        at = order[i:i + 16]
        batch = tokenizer([texts[j] for j in at], padding=True, truncation=True, max_length=64, return_tensors="pt")
        v = torch.nn.functional.normalize(model(**batch).last_hidden_state[:, 0].float(), dim=1).numpy()
        if vectors is None:
            vectors = np.zeros((len(texts), v.shape[1]), dtype=np.float32)
        vectors[at] = v
np.save(out + ".tmp.npy", vectors)
os.replace(out + ".tmp.npy", out)
"""


def embed_worker(job):
    table, parts, blocks, size, core, lock = job
    env = dict(os.environ, TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    while True:
        with lock:
            block = next((b for b in range(blocks) if not done(f"{parts}/{b:06d}.npy") and not done(f"{parts}/.{b:06d}.lock")), None)
            if block is None:
                return
            open(f"{parts}/.{block:06d}.lock", "w").close()
        subprocess.run(["taskset", "-c", str(core), sys.executable, "-c", EMBED_BLOCK, table, f"{parts}/{block:06d}.npy", AMAZON_MODEL, AMAZON_MODEL_REVISION, str(block),
                        str(size)],
                       env=env, check=True)
        os.unlink(f"{parts}/.{block:06d}.lock")


def embed_amazon(out, table, workers):
    """A process per block, because a long-lived process grows by about 2 GB per block."""
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download

    if done(f"{out}/base.fbin"):
        return
    rows = pq.ParquetFile(table).metadata.num_rows
    size = min(AMAZON_BLOCK, math.ceil(rows / workers))  # a small set still uses every core
    blocks = math.ceil(rows / size)
    parts = f"{out}/parts"
    os.makedirs(parts, exist_ok=True)
    for path in glob.glob(f"{parts}/.*.lock"):
        os.unlink(path)
    snapshot_download(AMAZON_MODEL, revision=AMAZON_MODEL_REVISION)
    manager = mp.Manager()
    lock = manager.Lock()
    with mp.Pool(workers) as pool:
        pool.map(embed_worker, [(table, parts, blocks, size, core, lock) for core in sorted(os.sched_getaffinity(0))[:workers]], chunksize=1)
    written = 0
    with open(f"{out}/base.fbin.tmp", "wb") as f:
        np.zeros(2, dtype=np.uint32).tofile(f)
        for b in range(blocks):
            v = np.load(f"{parts}/{b:06d}.npy")
            if np.any(np.abs(np.linalg.norm(v, axis=1) - 1) > 1e-3):
                raise SystemExit(f"block {b}: vectors are not unit length")
            dim = v.shape[1]
            v.astype(np.float32).tofile(f)
            written += len(v)
        f.seek(0)
        np.array([written, dim], dtype=np.uint32).tofile(f)
    if written != rows:
        raise SystemExit(f"{written} vectors for {rows} items")
    os.replace(f"{out}/base.fbin.tmp", f"{out}/base.fbin")
    subprocess.run(["rm", "-rf", parts], check=True)


def label_amazon(out, table):
    """Labels are the top category, every node of the category path, and rating and popularity marks. The range field is cents.
    A label needs 100 items in the full set, and as many in proportion in a limited set, but at least the 10 that a query returns."""
    import pyarrow.parquet as pq

    if done(f"{out}/base.spmat"):
        return
    t = pq.read_table(table, columns=["cat", "price", "rating", "nrating", "path"])
    n = t.num_rows
    min_count = max(10, round(100 * n / AMAZON_ITEMS))
    cat, path = t.column("cat").to_pylist(), t.column("path").to_pylist()
    price = t.column("price").to_numpy(zero_copy_only=False)
    rating, ratings = t.column("rating").to_numpy(), t.column("nrating").to_numpy()
    vocabulary, counts = {}, []

    def label_id(s):
        i = vocabulary.get(s)
        if i is None:
            i = vocabulary[s] = len(vocabulary)
            counts.append(0)
        counts[i] += 1
        return i

    rows = []
    for i in range(n):
        r = [label_id("CAT:" + cat[i])]
        r += [label_id("PATH:" + c) for c in dict.fromkeys(path[i])]  # in path order, so that label ids do not depend on string hashing
        if rating[i] >= 4.0:
            r.append(label_id("RATE4"))
        if rating[i] >= 4.5:
            r.append(label_id("RATE45"))
        if ratings[i] >= 100:
            r.append(label_id("POP100"))
        rows.append(r)
    counts = np.array(counts)
    names = list(vocabulary)
    kept = np.nonzero(counts >= min_count)[0]
    kept = kept[np.argsort(-counts[kept], kind="stable")]
    new = -np.ones(len(names), dtype=np.int32)
    new[kept] = np.arange(len(kept), dtype=np.int32)
    row = np.zeros(n + 1, dtype=np.int64)
    flat = []
    for i, r in enumerate(rows):
        x = sorted(int(new[j]) for j in r if new[j] >= 0)
        flat.extend(x)
        row[i + 1] = row[i] + len(x)
    ids = np.array(flat, dtype=np.int32)
    json.dump({"names": [names[i] for i in kept], "counts": np.bincount(ids, minlength=len(kept)).tolist()}, open(f"{out}/labels.json", "w"))
    # PipeANN-Filter binds ranges as float32, so the limit keeps every value exact.
    cents = np.minimum(np.rint(np.nan_to_num(price, nan=0.0) * 100), (1 << 24) - 1).astype(np.uint32)
    write_range(out, "price", cents)
    write_spmat(f"{out}/base.spmat", len(kept), row, ids)


def workload_amazon(out, table):
    """The query is the unit mean of a shopper's last items, and the filter narrows by what they bought next."""
    import pyarrow.parquet as pq

    path = f"{out}/workload.json"
    if done(path):
        return json.load(open(path))
    rng = np.random.default_rng(SEED)
    t = pq.read_table(table, columns=["asin", "cat", "price", "path"])
    asins = t.column("asin").to_pylist()
    row_of = {a: i for i, a in enumerate(asins)}
    cat, path_of = t.column("cat").to_pylist(), t.column("path").to_pylist()
    price = t.column("price").to_numpy(zero_copy_only=False)
    label_of = {name: i for i, name in enumerate(json.load(open(f"{out}/labels.json"))["names"])}
    n, dim = (int(x) for x in np.fromfile(f"{out}/base.fbin", dtype=np.uint32, count=2))
    base = np.memmap(f"{out}/base.fbin", dtype=np.float32, mode="r", offset=8, shape=(n, dim))
    filters = Filters(out, "price")
    users = amazon_users(out, rng)
    forms = ["cat", "node", "budget", "cat_budget", "node_budget", "node_rate"]
    rows, queries = [], []
    for u in users:
        if len(rows) >= AMAZON_QUERIES:
            break
        history = [row_of[a] for a in u["history"].split() if a in row_of][-AMAZON_HISTORY:]
        following = row_of.get(u["parent_asin"])
        if len(history) < 3 or following is None:
            continue
        v = np.asarray(base[np.sort(history)]).mean(axis=0)
        v /= np.linalg.norm(v)
        f = forms[int(rng.integers(len(forms)))]
        c = label_of.get("CAT:" + cat[following])
        nodes = [label_of["PATH:" + x] for x in path_of[following] if "PATH:" + x in label_of]
        node = nodes[-1] if nodes else None
        if "node" in f and node is None:
            f = "cat"
        if "cat" in f and c is None:
            continue
        p = price[following]
        hi = None
        if "budget" in f:
            over = [b for b in AMAZON_BUDGETS if p == p and b >= p]
            b = over[0] if over else int(rng.choice(AMAZON_BUDGETS))
            hi = b * 100 + 1  # half-open, so "under $b" includes $b.00
        if f == "cat":
            e, form, tags, bounds, count = f"L {c}", "labels", [c], None, filters.count_label(c)
        elif f == "node":
            e, form, tags, bounds, count = f"L {node}", "labels", [node], None, filters.count_label(node)
        elif f == "budget":
            e, form, tags, bounds, count = f"R price 1 {hi}", "range", None, [1, hi], filters.count_range(1, hi)
        elif f == "cat_budget":
            e, form, tags, bounds, count = f"AND 2 L {c} R price 1 {hi}", "labelrange", [c], [1, hi], filters.count_label_range(c, 1, hi)
        elif f == "node_budget":
            e, form, tags, bounds, count = f"AND 2 L {node} R price 1 {hi}", "labelrange", [node], [1, hi], filters.count_label_range(node, 1, hi)
        else:
            mark = label_of["RATE4"]
            count = filters.count_label(node) if node == mark else filters.count_labels(*sorted((node, mark)))
            e, form, tags, bounds = f"AND 2 L {node} L {mark}", "labels", sorted({node, mark}), None
        if count >= 10:
            rows.append(dict(q=len(queries), expr=e, form=form, tags=tags, bounds=bounds, count=count))
            queries.append(v.astype(np.float32))
    q = np.array(queries, dtype=np.float32)
    with open(f"{out}/query.fbin", "wb") as f:
        np.array(q.shape, dtype=np.uint32).tofile(f)
        q.tofile(f)
    return finish_workload(out, rng, n, rows, "price", f"{out}/query.fbin")


# Both datasets


def finish_workload(out, rng, n, rows, field, query_path):
    counts = np.array([r["count"] for r in rows])
    band = np.searchsorted(np.array(EDGES) * n, counts, side="right") - 1
    calibration = set(rng.permutation(len(rows))[:int(0.3 * len(rows))].tolist())
    for j, r in enumerate(rows):
        r["band"] = int(band[j])
        r["set"] = "cal" if j in calibration else "eval"
    workload = dict(n=n, field=field, query_path=query_path, rows=rows)
    json.dump(workload, open(f"{out}/workload.json", "w"))
    print(f"{len(rows)} queries, bands {np.bincount(band, minlength=len(EDGES) - 1).tolist()}", flush=True)
    return workload


def write_runs(out, workload):
    """One evaluation run per band and form. The calibration queries stay together, because calibration splits them by count."""
    runs_dir = f"{out}/runs"
    if done(f"{runs_dir}/runs.json"):
        return
    os.makedirs(runs_dir, exist_ok=True)
    field = workload["field"]
    n_q, dim = (int(x) for x in np.fromfile(workload["query_path"], dtype=np.uint32, count=2))
    queries = np.fromfile(workload["query_path"], dtype=np.float32, offset=8).reshape(n_q, dim)
    columns = int(np.fromfile(f"{out}/base.spmat", dtype=np.int64, count=2)[1])
    sql = {"labels": "array_contains_all(labels, $$qt)", "range": f"{field} = $$qr", "labelrange": f"array_contains_all(labels, $$qt) and {field} = $$qr"}
    attributes = [{"name": "labels", "key": 0, "type": "label", "file": f"{os.path.abspath(out)}/{PIPEANN_PREFIX}/index.label.0"},
                  {"name": field, "key": 1, "type": "range", "file": f"{os.path.abspath(out)}/{PIPEANN_PREFIX}/index.label.1"}]
    groups = defaultdict(list)
    for r in workload["rows"]:
        groups[(r["set"], r["band"], r["form"])].append(r)
    runs = []
    calibration = [r for r in workload["rows"] if r["set"] == "cal"]
    with open(f"{runs_dir}/calibration.fbin", "wb") as f:
        np.array([len(calibration), dim], dtype=np.uint32).tofile(f)
        queries[[r["q"] for r in calibration]].tofile(f)
    with open(f"{runs_dir}/calibration.expr", "w") as f:
        f.write("".join(r["expr"] + "\n" for r in calibration))
    for (kind, band, form), rows in sorted(groups.items()):
        if kind != "eval":
            continue
        name = f"eval_b{band}_{form}"
        with open(f"{runs_dir}/{name}_query.fbin", "wb") as f:
            np.array([len(rows), dim], dtype=np.uint32).tofile(f)
            queries[[r["q"] for r in rows]].tofile(f)
        with open(f"{runs_dir}/{name}.expr", "w") as f:
            f.write("".join(r["expr"] + "\n" for r in rows))
        config = {"attr_indexes": attributes, "filter": sql[form], "bindings": {}}
        if "$$qt" in sql[form]:
            tags = [r["tags"] for r in rows]
            write_spmat(f"{runs_dir}/{name}_qt.spmat", columns, np.r_[0, np.cumsum([len(t) for t in tags])], [x for t in tags for x in t])
            config["bindings"]["qt"] = f"{os.path.abspath(runs_dir)}/{name}_qt.spmat"
        if "$$qr" in sql[form]:
            write_spmat(f"{runs_dir}/{name}_qr.spmat", 2, np.arange(0, 2 * len(rows) + 1, 2), [0, 1] * len(rows), [x for r in rows for x in r["bounds"]])
            config["bindings"]["qr"] = f"{os.path.abspath(runs_dir)}/{name}_qr.spmat"
        json.dump(config, open(f"{runs_dir}/{name}.pipeann.json", "w"), indent=1)
        runs.append(dict(name=name, set="eval", band=band, form=form, n=len(rows)))

    json.dump(dict(range_field=field, runs=runs), open(f"{runs_dir}/runs.json.tmp", "w"), indent=1)
    os.replace(f"{runs_dir}/runs.json.tmp", f"{runs_dir}/runs.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["pubmed", "amazon"])
    ap.add_argument("directory")
    ap.add_argument("--limit", type=int, default=0, help="use about this many vectors, for a quick run")
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    a = ap.parse_args()
    out = a.directory
    os.makedirs(out, exist_ok=True)
    if a.dataset == "pubmed":
        fetch_pubmed(out, a.limit, a.workers)
        label_pubmed(out, fetch_nlm(out, a.workers))
        workload = workload_pubmed(out)
    else:
        table = parse_amazon(out, a.limit)
        embed_amazon(out, table, a.workers)
        label_amazon(out, table)
        workload = workload_amazon(out, table)
    write_runs(out, workload)


if __name__ == "__main__":
    main()
