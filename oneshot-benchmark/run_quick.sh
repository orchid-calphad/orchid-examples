#!/bin/sh
# Quick path: figure from the cached scores and phase map, and the grid-scoring path checked on the cached GRACE map
# (about 30 s, no pycalphad). Usage: sh run_quick.sh [OUT_DIR]   (default figures/)
set -e
cd "$(dirname "$0")"
OUT=${1:-figures}
PY=${PYTHON:-python}
mkdir -p "$OUT"
$PY scripts/figure.py --out "$OUT"
$PY scripts/npz_to_grid.py results/map_cuzr_grace.npz "$OUT/grace_grid.csv"
$PY scripts/score_grid.py "$OUT/grace_grid.csv" --label grace_grid -o "$OUT/grace_grid_score.json"
rm -f "$OUT/grace_grid.csv"
