#!/bin/sh
# Quick path: execute the notebook offline on the shipped Cu-Zr slice (about 20 s). Set ORCHID_DATA to use the full
# release instead (same outputs). Usage: sh run_quick.sh [OUT_DIR]  (figures; default figures/; the notebook file is not rewritten)
set -e
cd "$(dirname "$0")"
export ORCHID_DATA=${ORCHID_DATA:-sample-release}
export QUICKSTART_OUT=${1:-figures}
export QUICKSTART_EXPORT=${QUICKSTART_EXPORT:-export}
${PYTHON:-python} scripts/run_notebook.py --no-save
