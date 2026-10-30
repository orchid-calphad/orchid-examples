"""Paths for the full (compute) path of the PhaseForge+ example.

Everything the full path writes goes under <example>/work/, in the layout of the original working repository
(orchid-phaseforge-demo): work/runs, work/data, work/results, work/tdb, work/structures. On first use work/tdb and
work/results are seeded with this example's TDBs and MLIP sigma, so that the fit can be rerun without the MLIP steps.

Inputs from outside this repository:
  ORCHID_DATA        local orchid-data checkout at the pinned release (README: 778cc6bfa67db292e3f917f5ac6d691e0f37fc1a);
                     required by build_data.py, printed_invariants.py and sigma_mlip.py.
  ORCHID_DENSE_DATA  optional: orchid-data checkout at dense_release (default: ORCHID_DATA, if it holds the file). The dense
                     assessed-field grid of the final fit (177 ZPF points, build_a4.py step 3) is released as
                     espei/assessed/CU-ZR/CuZr-94Zen-LB.CU-ZR-p371.dense.json from that commit on; it is not in the
                     older release (the other final-fit data come from release), so the two can be separate
                     checkouts. The file is verified against dense_sha256.
  ORCHID_DENSE_GRID  optional override: another file with the same layout (then no hash check).
  PHASEFORGEPLUS_SRC optional: a PhaseForgePlus checkout (its example/PhaseForge ATAT SQS, for validate_sqs.py only).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

example = Path(__file__).resolve().parents[2]
root = Path(os.environ.get("ORCHID_WORK", str(example / "work"))).expanduser().resolve()
sgte_tdb = example / "tdb" / "CuZr-94Zen-LB.tdb"
release = "778cc6bfa67db292e3f917f5ac6d691e0f37fc1a"
dense_release = "e6b9b3f649e612ed8432f5487304c78910ab7604"
dense_rel = "espei/assessed/CU-ZR/CuZr-94Zen-LB.CU-ZR-p371.dense.json"
dense_sha256 = "62b78ff8f2efcbced4c7c84f63fe93447ee5b38c3b2a94b824cd436bfad4f714"
sys.path.insert(0, str(example / "scripts"))  # model.py, pd_map.py

for sub in ("runs", "data", "results", "tdb", "structures"):
    (root / sub).mkdir(parents=True, exist_ok=True)
for f in ("CuZr-PhaseForge-MLIP.tdb", "CuZr-PhaseForge-MLIP-B2b.tdb", "CuZr-PhaseForge-final.tdb", "mlip_fit.json"):
    if os.environ.get("ORCHID_SEED_CACHED", "1") == "1" and not (root / "tdb" / f).exists():
        shutil.copy(example / "tdb" / f, root / "tdb" / f)
if os.environ.get("ORCHID_SEED_CACHED", "1") == "1" and not (root / "results" / "sigma_mlip.json").exists():
    shutil.copy(example / "inputs" / "sigma_mlip.json", root / "results" / "sigma_mlip.json")


def orchid(rel: str) -> Path:
    """Path of a file inside the ORCHID data checkout.

    Args:
        rel (str): Path relative to the orchid-data root.

    Returns:
        The file path under ORCHID_DATA. Exits the process if ORCHID_DATA is not set.
    """
    data_root = os.environ.get("ORCHID_DATA")
    if not data_root:
        sys.exit(
            f"set ORCHID_DATA to an orchid-data checkout at commit {release} (git clone "
            "https://github.com/orchid-calphad/orchid-data && git -C orchid-data checkout " + release + ")"
        )
    return Path(data_root).expanduser() / rel


def dense_grid() -> Path:
    """The 177-point dense assessed-field dataset of the final fit, from the ORCHID release (sha256 verified).

    Returns:
        Path to the dense grid file, from ORCHID_DENSE_GRID or from the pinned orchid-data commit. Exits the process
        if the file is missing or its sha256 differs from dense_sha256.
    """
    if os.environ.get("ORCHID_DENSE_GRID"):
        return Path(os.environ["ORCHID_DENSE_GRID"])
    for var in ("ORCHID_DENSE_DATA", "ORCHID_DATA"):
        base = os.environ.get(var)
        f = Path(base).expanduser() / dense_rel if base else None
        if f and f.exists():
            h = hashlib.sha256(f.read_bytes()).hexdigest()
            if h != dense_sha256:
                sys.exit(f"{f}: sha256 {h[:12]} differs from the pinned {dense_sha256[:12]} (orchid-data {dense_release})")
            return f
    sys.exit(
        f"the dense assessed-field grid is in orchid-data from commit {dense_release}: set ORCHID_DENSE_DATA (or "
        "ORCHID_DATA) to a checkout of that commit"
    )


pfp_src = Path(os.environ["PHASEFORGEPLUS_SRC"]) if os.environ.get("PHASEFORGEPLUS_SRC") else None
