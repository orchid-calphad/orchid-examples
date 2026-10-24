#!/bin/sh
# Quick path: gate table + the paper's three-diagram overlay from the cached TDBs (about 3-4 min on one CPU core).
# Usage: sh run_quick.sh [OUT_DIR]   (default: figures/; gate.json is written next to the figures)
set -e
cd "$(dirname "$0")"
OUT=${1:-figures}
PY=${PYTHON:-python}
$PY scripts/gate.py "$OUT"
$PY scripts/fig_overlay.py "$OUT"
