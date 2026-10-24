#!/bin/sh
# Full path: rebuild the evidence from the pinned ORCHID release and score every model of the Cu-Zr test with the same
# evidence, grid and solver settings (about 10-15 min on one CPU core). Run from anywhere; PYTHON selects the interpreter.
set -e
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
$PY scripts/build_evidence.py
$PY scripts/score.py tdb/CuZr-PhaseForge-MLIP.tdb --label cuzr_grace --names-as CuZr-94Zen-LB.tdb
$PY scripts/score.py tdb/CuZr-94Zen-LB.tdb --label cuzr_sgte_94Zen
$PY scripts/score.py tdb/CuZr-PhaseForge-final.tdb --label cuzr_phaseforgeplus_final --names-as CuZr-94Zen-LB.tdb \
  --in-sample-note "IN-SAMPLE: fitted (orchid-phaseforge-demo runs/final2/F10b_s3, data/a4d_F10) to ORCHID Cu-Zr ZPF/invariant data, the Dataset 6 assessed points in 94Zen names and a dense grid over the same assessed fields"
for f in tdb/reference_panel/CuZr-*.tdb; do
  b=$(basename "$f" .tdb)
  $PY scripts/score.py "$f" --label "ref_$b" -o "results/reference_panel/$b.json"
done
$PY scripts/compare.py
