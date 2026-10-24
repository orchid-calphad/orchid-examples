#!/bin/sh
# Quick path: saturation statistics + figure from the cached MLIP predictions (about 30 s; standard library + matplotlib).
# Usage: sh run_quick.sh [OUT_DIR]   (default: results/summary.json and figures/)
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python}
if [ -n "$1" ]; then $PY scripts/analyze.py "$1" && $PY scripts/make_figure.py "$1"
else $PY scripts/analyze.py && $PY scripts/make_figure.py; fi
