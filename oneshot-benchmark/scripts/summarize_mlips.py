"""Summarise the one-shot Cu-Zr scores of the MLIP-only descriptions next to the references.

Table of results/mlips/cuzr_*.json.
Usage: python summarize_mlips.py   -> results/mlips_summary.md (markdown table, printed too).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from release import example_root

if TYPE_CHECKING:
    from pathlib import Path

model_names = {
    "grace-2l-omat": "GRACE-2L-OMAT",
    "mace-mpa-0": "MACE-MPA-0",
    "mace-omat-0": "MACE-OMAT-0",
    "sevennet-omni": "SevenNet-Omni",
    "orb-v3-omat": "ORB-v3 OMat",
    "orb-v3-mpa": "ORB-v3 MPA",
    "mattersim-5m": "MatterSim 5M",
    "chgnet-0.3.0": "CHGNet 0.3.0",
}


def cell(d: dict, kind: str) -> tuple[str, str, str]:
    """Recall with its interval, precision and F1 of one kind, as table cells.

    Args:
        d (dict): Scored result.
        kind (str): "field" or "boundary".

    Returns:
        tuple[str, str, str]: Recall with its 95% interval, precision and F1, each formatted to two decimals.
    """
    b = d["metrics"]["physical"][kind]["all"]
    w, c = b["weighted"], b["ci95"]["recall"]
    return f"{w['recall']:.2f} [{c[0]:.2f}, {c[1]:.2f}]", f"{w['precision']:.2f}", f"{w['f1']:.2f}"


def main() -> None:
    """Write results/mlips_summary.md, sorted by field recall, with the two references at the bottom."""
    rows = []
    for f in _mlip_files():
        d = json.loads(f.read_text())
        rows.append((d["metrics"]["physical"]["field"]["all"]["weighted"]["recall"], model_names[d["label"][5:]], d))
    rows.sort(key=lambda r: -r[0])
    ref = [
        ("SGTE 94Zen (reference)", "results/cuzr_sgte_94Zen.json"),
        ("PhaseForge+ final (in-sample)", "results/cuzr_phaseforgeplus_final.json"),
    ]
    out = [
        "| model (MLIP-only, no fitting) | field S [95% CI] | P | F1 | boundary S [95% CI] | P | F1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, n, d in rows + [(0, a, json.loads((example_root / p).read_text())) for a, p in ref]:
        out.append("| " + " | ".join([n, *cell(d, "field"), *cell(d, "boundary")]) + " |")
    s = "\n".join(out)
    (example_root / "results" / "mlips_summary.md").write_text(s + "\n")
    print(s)


def _mlip_files() -> list[Path]:
    """Per-MLIP result files, in the order the directory lists them.

    Returns:
        list[Path]: Files results/mlips/cuzr_*.json.
    """
    return list((example_root / "results" / "mlips").glob("cuzr_*.json"))


if __name__ == "__main__":
    main()
