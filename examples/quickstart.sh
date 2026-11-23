#!/bin/bash
# Build an index on synthetic data, calibrate a plan, and search with filters.
set -eu
HERE=$(cd "$(dirname "$0")/.." && pwd)
K="$HERE/build/kuroann"
D="${1:-$HERE/data/quickstart}"
python3 "$HERE/examples/make_synthetic.py" "$D"
"$K" build --base "$D/base.fbin" --labels "$D/labels.spmat" --range price="$D/price.u32" --index "$D/index" --pq 32
python3 "$HERE/scripts/calibrate.py" --index "$D/index" --queries "$D/sample.fbin" --filters "$D/sample.expr" \
  --out "$D/plan" --kuroann "$K" --targets 0.99
python3 "$HERE/scripts/groundtruth.py" --index "$D/index" --queries "$D/test.fbin" --filters "$D/test.expr" --out "$D/test"
"$K" search --index "$D/index" --queries "$D/test.fbin" --filters "$D/test.expr" --plan "$D/plan/plan_99.txt" \
  --groundtruth "$D/test.gtpf" --rounds 3
