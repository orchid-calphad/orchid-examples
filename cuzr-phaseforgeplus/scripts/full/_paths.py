"""Paths for the full (compute) path of the PhaseForge+ example.

Everything the full path writes goes under <example>/work/, in the layout of the original working repository
(orchid-phaseforge-demo): work/runs, work/data, work/results, work/tdb, work/structures. On first use work/tdb and
work/results are seeded with this example's TDBs and MLIP sigma, so that the fit can be rerun without the MLIP steps.

Inputs from outside this repository:
  ORCHID_DATA        local orchid-data checkout at the pinned release (README: 778cc6bfa67db292e3f917f5ac6d691e0f37fc1a);
                     required by build_data.py, printed_invariants.py and sigma_mlip.py.
  ORCHID_DENSE_DATA  optional: orchid-data checkout at DENSE_RELEASE (default: ORCHID_DATA, if it holds the file). The dense
                     assessed-field grid of the final fit (177 ZPF points, build_a4.py step 3) is released as
                     espei/assessed/CU-ZR/CuZr-94Zen-LB.CU-ZR-p371.dense.json from that commit on; it is not in the
                     older release RELEASE (the other final-fit data come from RELEASE), so the two can be separate
                     checkouts. The file is verified against DENSE_SHA256.
  ORCHID_DENSE_GRID  optional override: another file with the same layout (then no hash check).
  PHASEFORGEPLUS_SRC optional: a PhaseForgePlus checkout (its example/PhaseForge ATAT SQS, for validate_sqs.py only).
"""
import os, shutil, sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("ORCHID_WORK", str(EXAMPLE / "work"))).expanduser().resolve()
SGTE_TDB = EXAMPLE / "tdb" / "CuZr-94Zen-LB.tdb"
RELEASE = "778cc6bfa67db292e3f917f5ac6d691e0f37fc1a"
DENSE_RELEASE = "e6b9b3f649e612ed8432f5487304c78910ab7604"
DENSE_REL = "espei/assessed/CU-ZR/CuZr-94Zen-LB.CU-ZR-p371.dense.json"
DENSE_SHA256 = "62b78ff8f2efcbced4c7c84f63fe93447ee5b38c3b2a94b824cd436bfad4f714"
sys.path.insert(0, str(EXAMPLE / "scripts"))          # model.py, pd_map.py

for sub in ("runs", "data", "results", "tdb", "structures"):
    (ROOT / sub).mkdir(parents=True, exist_ok=True)
for f in ("CuZr-PhaseForge-MLIP.tdb", "CuZr-PhaseForge-MLIP-B2b.tdb", "CuZr-PhaseForge-final.tdb", "mlip_fit.json"):
    if os.environ.get("ORCHID_SEED_CACHED", "1") == "1" and not (ROOT / "tdb" / f).exists():
        shutil.copy(EXAMPLE / "tdb" / f, ROOT / "tdb" / f)
if os.environ.get("ORCHID_SEED_CACHED", "1") == "1" and not (ROOT / "results" / "sigma_mlip.json").exists():
    shutil.copy(EXAMPLE / "inputs" / "sigma_mlip.json", ROOT / "results" / "sigma_mlip.json")


def orchid(rel):
    root = os.environ.get("ORCHID_DATA")
    if not root:
        sys.exit(f"set ORCHID_DATA to an orchid-data checkout at commit {RELEASE} (git clone "
                 "https://github.com/orchid-calphad/orchid-data && git -C orchid-data checkout " + RELEASE + ")")
    return Path(root).expanduser() / rel


def dense_grid():
    """The 177-point dense assessed-field dataset of the final fit, from the ORCHID release (sha256 verified)."""
    import hashlib
    if os.environ.get("ORCHID_DENSE_GRID"):
        return Path(os.environ["ORCHID_DENSE_GRID"])
    for var in ("ORCHID_DENSE_DATA", "ORCHID_DATA"):
        root = os.environ.get(var)
        f = Path(root).expanduser() / DENSE_REL if root else None
        if f and f.exists():
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            if h != DENSE_SHA256:
                sys.exit(f"{f}: sha256 {h[:12]} differs from the pinned {DENSE_SHA256[:12]} (orchid-data {DENSE_RELEASE})")
            return f
    sys.exit(f"the dense assessed-field grid is in orchid-data from commit {DENSE_RELEASE}: set ORCHID_DENSE_DATA (or "
             "ORCHID_DATA) to a checkout of that commit")


PFP_SRC = Path(os.environ["PHASEFORGEPLUS_SRC"]) if os.environ.get("PHASEFORGEPLUS_SRC") else None
