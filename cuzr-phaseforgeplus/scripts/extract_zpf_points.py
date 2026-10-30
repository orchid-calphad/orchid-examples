"""Write inputs/zpf_points.json: the ORCHID phase-equilibrium points drawn on the overlay figure.

They are read from the fit and hold-out datasets that scripts/full/build_data.py assembles from the ORCHID ESPEI bundle
(release pinned in the README): literature ZPF and invariant vertices with a printed composition, and the assessed
phase-assemblage points (ORCHID Dataset 6, in CuZr-94Zen-LB phase names) at their hyperplane composition. Invariant
brackets and liquid-stability points (derived constraints) are not drawn. Only released, derived coordinates are kept.

  python scripts/extract_zpf_points.py DATA_DIR      # DATA_DIR holds fit/ and holdout/ (output of build_data.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def main(data) -> None:
    """Collect the phase-equilibrium points of the fit and hold-out sets and write inputs/zpf_points.json.

    Args:
        data (str | Path): Directory holding the fit/ and holdout/ subfolders written by build_data.py.
    """
    pts = []
    for split in ("fit", "holdout"):
        files = [p for p in Path(data, split).glob("*.json") if not p.name.startswith(".")]  # glob.glob skips dotfiles
        for path in sorted(files, key=str):
            f = str(path)
            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("output") != "ZPF" or "liquid-stability" in f:
                continue
            kind = "assessed" if "assessed__" in f else ("bracket" if "brackets" in f else "exp")
            if kind == "bracket":
                continue
            temps = d["conditions"]["T"]
            temps = temps if isinstance(temps, list) else [temps] * len(d["values"])
            # Lengths are not assumed equal here: a list of temperatures may be shorter than the value rows.
            for temp, row in zip(temps, d["values"], strict=False):
                hp = [v for v in row if v[0] == "__HYPERPLANE__"]
                for ph, _comps, xs in row:
                    if ph == "__HYPERPLANE__":
                        continue
                    if xs and xs[0] is not None:
                        pts.append({"x_Zr": xs[0], "T_K": temp, "split": split, "kind": kind, "file": path.name})
                    elif kind == "assessed" and hp and hp[0][2] and hp[0][2][0] is not None:
                        pts.append({"x_Zr": hp[0][2][0], "T_K": temp, "split": split, "kind": kind, "file": path.name})
                        break
    out = {
        "note": "ORCHID phase-equilibrium points (fit / held out) and assessed phase-assemblage points (Dataset 6), "
        "as read from the PhaseForge+ fit data; derived coordinates only",
        "points": pts,
    }
    (root / "inputs/zpf_points.json").write_text(json.dumps(out, indent=0), encoding="utf-8")
    print(len(pts), "points ->", root / "inputs/zpf_points.json")


if __name__ == "__main__":
    main(sys.argv[1])
