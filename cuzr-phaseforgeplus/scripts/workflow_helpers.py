"""Small explicit preparation steps used by the reproducible workflow runner."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

expected_compounds = 6  # line compounds with a structure file (Cu5Zr, Cu51Zr14, Cu8Zr3, Cu10Zr7, CuZr, CuZr2)


def prepare_compounds(root) -> None:
    """Copy each compound structure into its run folder and mark it for the workflow's wait step.

    Args:
        root (Path): Workflow run directory holding structures/*.vasp.

    Raises:
        ValueError: If the structures folder does not hold exactly expected_compounds structure files.
    """
    root = Path(root)
    files = sorted((root / "structures").glob("*.vasp"))
    if len(files) != expected_compounds:
        raise ValueError("Expected six compound structures")
    for p in files:
        d = root / "runs/phaseforge/COMPOUNDS" / p.stem
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, d / "POSCAR")
        (d / "wait").write_text("")


def b2_variant(root) -> None:
    """Write the MLIP-only TDB with a linear T term on CUZR_B2 (the archived Amendment 4 variant).

    Args:
        root (Path): Example or run directory holding tdb/CuZr-PhaseForge-MLIP.tdb.

    Raises:
        ValueError: If the MLIP TDB does not have the expected layout (missing or duplicated lines).
    """
    root = Path(root)
    p = root / "tdb/CuZr-PhaseForge-MLIP.tdb"
    s = p.read_text()
    old = "1*GHSERCU+1*GHSERZR+VV0020*1000;"
    if s.count(old) != 1 or "FUNCTION VV0022" in s:
        raise ValueError("Unexpected MLIP TDB layout")
    s = s.replace("FUNCTION VV0021", "FUNCTION VV0022 298.15 0.000000; 6000 N !\nFUNCTION VV0021", 1)
    s = s.replace(old, "1*GHSERCU+1*GHSERZR+VV0020*1000+VV0022*T;")
    header = "$ MLIP-only TDB + linear T term (VV0022, start 0) for CUZR_B2 (Amendment 4 variant)\n"
    (root / "tdb/CuZr-PhaseForge-MLIP-B2b.tdb").write_text(header + s)


def gate_run(root, start) -> None:
    """Run the invariant gate on a workflow run's tuned TDB, writing results/gate_s<start>.

    Args:
        root (Path): Workflow run directory.
        start (int): Start seed of the tuned fit (selects runs/final2/F10b_s<start>/tuned.tdb).
    """
    root = Path(root)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gate  # lazy: gate pulls in the analytic model only when this step runs  # noqa: PLC0415  lazy: gate needs the sys.path insert done first

    gate.tdbs = {
        "SGTE": gate.root / "tdb/CuZr-94Zen-LB.tdb",
        "MLIP-only": root / "tdb/CuZr-PhaseForge-MLIP.tdb",
        "tuned": root / f"runs/final2/F10b_s{start}/tuned.tdb",
    }
    # gate.main takes its printed-data path from gate.root; leave model's unary-table root intact.
    old = gate.root
    gate.root = root
    (root / "inputs").mkdir(exist_ok=True)
    shutil.copy2(root / "results/printed_invariants.json", root / "inputs/printed_invariants.json")
    gate.main(root / f"results/gate_s{start}")
    gate.root = old


def main() -> None:
    """Parse the step name and run that preparation step in the directory named by ORCHID_WORK."""
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare-compounds", "b2-variant", "gate-run"])
    ap.add_argument("--start", type=int)
    a = ap.parse_args()
    root = Path(os.environ["ORCHID_WORK"])
    if a.step == "prepare-compounds":
        prepare_compounds(root)
    elif a.step == "b2-variant":
        b2_variant(root)
    else:
        gate_run(root, a.start)


if __name__ == "__main__":
    main()
