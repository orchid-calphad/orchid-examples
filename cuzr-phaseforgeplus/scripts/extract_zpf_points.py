"""Write inputs/zpf_points.json: the ORCHID phase-equilibrium points drawn on the overlay figure.

They are read from the fit and hold-out datasets that scripts/full/build_data.py assembles from the ORCHID ESPEI bundle
(release pinned in the README): literature ZPF and invariant vertices with a printed composition, and the assessed
phase-assemblage points (ORCHID Dataset 6, in CuZr-94Zen-LB phase names) at their hyperplane composition. Invariant
brackets and liquid-stability points (derived constraints) are not drawn. Only released, derived coordinates are kept.

  python scripts/extract_zpf_points.py DATA_DIR      # DATA_DIR holds fit/ and holdout/ (output of build_data.py)
"""
import glob, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(data):
    pts = []
    for split in ("fit", "holdout"):
        for f in sorted(glob.glob(str(Path(data) / split / "*.json"))):
            d = json.load(open(f))
            if d.get("output") != "ZPF" or "liquid-stability" in f:
                continue
            kind = "assessed" if "assessed__" in f else ("bracket" if "brackets" in f else "exp")
            if kind == "bracket":
                continue
            Ts = d["conditions"]["T"]; Ts = Ts if isinstance(Ts, list) else [Ts] * len(d["values"])
            for T, row in zip(Ts, d["values"]):
                hp = [v for v in row if v[0] == "__HYPERPLANE__"]
                for ph, comps, xs in row:
                    if ph == "__HYPERPLANE__":
                        continue
                    if xs and xs[0] is not None:
                        pts.append({"x_Zr": xs[0], "T_K": T, "split": split, "kind": kind, "file": Path(f).name})
                    elif kind == "assessed" and hp and hp[0][2] and hp[0][2][0] is not None:
                        pts.append({"x_Zr": hp[0][2][0], "T_K": T, "split": split, "kind": kind, "file": Path(f).name})
                        break
    out = {"note": "ORCHID phase-equilibrium points (fit / held out) and assessed phase-assemblage points (Dataset 6), "
                   "as read from the PhaseForge+ fit data; derived coordinates only", "points": pts}
    json.dump(out, open(ROOT / "inputs/zpf_points.json", "w"), indent=0)
    print(len(pts), "points ->", ROOT / "inputs/zpf_points.json")


if __name__ == "__main__":
    main(sys.argv[1])
