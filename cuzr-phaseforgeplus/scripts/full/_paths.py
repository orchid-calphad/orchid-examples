"""Paths for the full (compute) path of the PhaseForge+ example.

Everything the full path writes goes under <example>/work/, in the layout of the original working repository
(orchid-phaseforge-demo): work/runs, work/data, work/results, work/tdb, work/structures. On first use work/tdb and
work/results are seeded with this example's TDBs and MLIP sigma, so that the fit can be rerun without the MLIP steps.

Inputs from outside this repository:
  ORCHID_DATA        local orchid-data checkout at the pinned release (README: 778cc6bfa67db292e3f917f5ac6d691e0f37fc1a);
                     required by build_data.py, printed_invariants.py and sigma_mlip.py.
  ORCHID_DENSE_GRID  optional: the dense assessed-field ZPF grid used by the paper's final fit (build_a4.py step 3).
                     It was derived for the demonstration from the field map of the assessed Cu-Zr diagram and is NOT
                     part of the ORCHID release, so it is not distributed here (available on request); without it
                     build_a4.py writes the same data without that grid and the fit is not the paper's configuration.
  PHASEFORGEPLUS_SRC optional: a PhaseForgePlus checkout (its example/PhaseForge ATAT SQS, for validate_sqs.py only).
"""
import os, shutil, sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2]
ROOT = EXAMPLE / "work"
SGTE_TDB = EXAMPLE / "tdb" / "CuZr-94Zen-LB.tdb"
RELEASE = "778cc6bfa67db292e3f917f5ac6d691e0f37fc1a"
sys.path.insert(0, str(EXAMPLE / "scripts"))          # model.py, pd_map.py

for sub in ("runs", "data", "results", "tdb", "structures"):
    (ROOT / sub).mkdir(parents=True, exist_ok=True)
for f in ("CuZr-PhaseForge-MLIP.tdb", "CuZr-PhaseForge-MLIP-B2b.tdb", "CuZr-PhaseForge-final.tdb", "mlip_fit.json"):
    if not (ROOT / "tdb" / f).exists():
        shutil.copy(EXAMPLE / "tdb" / f, ROOT / "tdb" / f)
if not (ROOT / "results" / "sigma_mlip.json").exists():
    shutil.copy(EXAMPLE / "inputs" / "sigma_mlip.json", ROOT / "results" / "sigma_mlip.json")


def orchid(rel):
    root = os.environ.get("ORCHID_DATA")
    if not root:
        sys.exit(f"set ORCHID_DATA to an orchid-data checkout at commit {RELEASE} (git clone "
                 "https://github.com/orchid-calphad/orchid-data && git -C orchid-data checkout " + RELEASE + ")")
    return Path(root).expanduser() / rel


DENSE_GRID = Path(os.environ["ORCHID_DENSE_GRID"]) if os.environ.get("ORCHID_DENSE_GRID") else None
PFP_SRC = Path(os.environ["PHASEFORGEPLUS_SRC"]) if os.environ.get("PHASEFORGEPLUS_SRC") else None
