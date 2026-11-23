#!/usr/bin/env bash
# Prepares the datasets, builds the indexes of KuroANN and the baselines, calibrates the plans, measures recall and throughput,
# and draws the figure. A finished step is kept, so running the script again only does what is missing.
#
#   DATA       dataset and index directory (default data)
#   OUT        results directory (default results)
#   DATASETS   subset of "pubmed amazon"
#   SYSTEMS    subset of "kuroann pipeann faiss"
#   THREADS    search and build threads (default: all CPUs)
#   LIMIT      use about this many vectors per dataset, for a quick run
#   PYTHON     Python with numpy, pyarrow, huggingface_hub, lxml, torch, transformers, and matplotlib (default python3)
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:-data}
OUT=${OUT:-results}
DATASETS=${DATASETS:-"pubmed amazon"}
SYSTEMS=${SYSTEMS:-"kuroann pipeann faiss"}
THREADS=${THREADS:-$(nproc)}
LIMIT=${LIMIT:-0}
PYTHON=${PYTHON:-python3}
run=()
if command -v numactl > /dev/null && [ "$(numactl --hardware | awk '/available:/ {print $2}')" -gt 1 ]; then
  run=(numactl --interleave=all)
fi
want() { [[ " $SYSTEMS " == *" $1 "* ]]; }
measure() {  # dataset system bench-arguments... -> $OUT/<dataset>/<system>.jsonl
  local ds=$1 system=$2 out=$OUT/$1/$2.jsonl
  shift 2
  [ -s "$out" ] && return
  sync
  "${run[@]}" "$PYTHON" baselines/bench.py "${system%-*}" --data "$DATA/$ds" --out "$out" --threads "$THREADS" "$@"
}

[ -x build/kuroann ] || { cmake -B build > /dev/null && cmake --build build -j > /dev/null; }
if want pipeann || want faiss; then
  BASELINES=$PWD/baselines/build bash baselines/build.sh
fi

for ds in $DATASETS; do
  case $ds in
    pubmed) pq=128 field=year ;;
    amazon) pq=96 field=price ;;
    *) echo "unknown dataset $ds" >&2; exit 2 ;;
  esac
  d=$DATA/$ds
  mkdir -p "$d" "$OUT/$ds"
  [ -s "$d/runs/runs.json" ] || "$PYTHON" scripts/datasets.py "$ds" "$d" --limit "$LIMIT"

  # Faiss-SSD and the exact answers read KuroANN's index, so the index comes first.
  index=$d/kuroann/index
  if [ ! -s "$index.vectors" ]; then
    mkdir -p "$d/kuroann"
    "${run[@]}" build/kuroann build --base "$d/base.fbin" --index "$index" --pq $pq --labels "$d/base.spmat" \
      --range "$field=$d/$field.u32" --threads "$THREADS"
  fi
  for name in $("$PYTHON" -c "import json,sys;print(' '.join(r['name'] for r in json.load(open(sys.argv[1]))['runs']))" "$d/runs/runs.json"); do
    [ -s "$d/runs/$name.gtpf" ] || "$PYTHON" scripts/groundtruth.py --index "$index" --queries "$d/runs/${name}_query.fbin" \
      --filters "$d/runs/$name.expr" --out "$d/runs/$name" --workers "$THREADS"
  done

  if want kuroann; then
    if [ ! -s "$d/kuroann/plan/plan_99.txt" ]; then
      sync
      "${run[@]}" "$PYTHON" scripts/calibrate.py --index "$index" --queries "$d/runs/calibration.fbin" --filters "$d/runs/calibration.expr" \
        --out "$d/kuroann/plan" --kuroann build/kuroann --threads "$THREADS"
    fi
    measure "$ds" kuroann --bin build/kuroann --index "$index" --plans "$d/kuroann/plan"
  fi
  if want pipeann; then
    tools=baselines/build/PipeANN/build/tests
    if [ ! -s "$d/pipeann/index_disk.index" ]; then
      mkdir -p "$d/pipeann"
      # A build that stopped, for example out of memory, continues from its saved shards.
      resume=()
      if [ -e "$d/pipeann/index_disk.index_tempFiles_subshard-0_ids_uint32.bin" ]; then resume=(env BUILD_RESUME_SHARDS=1); fi
      "${run[@]}" "${resume[@]}" "$tools/build_disk_index_filtered" float "$d/base.fbin" "$d/pipeann/index" 96 96 128 $pq 90 "$THREADS" l2 pq \
        label_spmat "$d/base.spmat" range "$d/$field.bin"
    fi
    measure "$ds" pipeann-filter --bin "$tools/search_disk_index_filtered" --index "$d/pipeann/index"
  fi
  if want faiss; then
    measure "$ds" faiss-ssd --bin baselines/build/faiss_ssd --index "$index"
  fi
done
"$PYTHON" scripts/plot.py "$OUT"
