#!/bin/sh
# Quick path: the negative-result figure from the cached summary, and the gate of the SGTE starting model (about 40 s).
# Usage: sh run_quick.sh [OUT_DIR]   (default figures/)
set -e
cd "$(dirname "$0")"
OUT=${1:-figures}
PY=${PYTHON:-python}
mkdir -p "$OUT"
$PY scripts/gate.py -o "$OUT/gate_sgte_start.json"
$PY scripts/figure.py "$OUT"
