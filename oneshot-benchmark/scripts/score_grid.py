"""Score a Cu-Zr phase diagram given as an equilibrium GRID (no TDB needed).

This is the "score your own model" path for models that are not CALPHAD databases (e.g. direct MLIP free-energy calculations, ML phase classifiers, digitised diagrams).

  python score_grid.py GRID.csv [--label NAME] [--names-as FILE.tdb] [--evidence inputs/evidence.json] [-o out.json]

GRID.csv columns: T_K, x_Zr, phases. `phases` lists the stable phases at that node separated by ';' (or '+'), given
either as benchmark classes (LIQUID, FCC_A1, BCC_A2, HCP_A3, C15B_CU5ZR, CU51ZR14, CU8ZR3, CU10ZR7, CUZR_B2, CUZR2,
CU24ZR13, CU2ZR, CU5ZR8, CU5ZR7, ...; physical mode of phase_identity.py) or, with --names-as, as phase names of a TDB
that the ORCHID phase registry knows (e.g. --names-as CuZr-94Zen-LB.tdb maps CU5ZR_C15B -> C15B_CU5ZR). Unknown names
count as their own class (false positives only).

Rule (the grid analogue of score.py): a FIELD point takes the assemblage of the nearest grid node (T scaled by 1/1000 K
and x by 1 in the distance); a BOUNDARY point takes the union of the assemblages of all nodes within T +- 10 K and
x +- 0.01, or of the nearest node when the band holds none. Use a grid of at most 5 K x 0.005 over 700-1800 K so that
every band holds nodes; the script reports how many bands were empty. Metrics, bootstrap and cluster CIs are those of
score.py (physical mode). Grid resolution changes boundary results slightly: scoring the GRACE TDB's 4 K x 0.004 map
this way gives recalls within about 0.02 of score.py on the TDB (see README).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parent))
import phase_identity as pid  # noqa: E402
from release import example_root, file_sha256, rel, release_commit  # noqa: E402
from score_metrics import cluster_ci, metric_block, seed, source_key  # noqa: E402

grid_dt = 10.0  # K, half-width of the boundary tolerance band in T
grid_dx = 0.01  # mole fraction, half-width of the boundary tolerance band in x


def load_grid(path: str | Path, names_as: str | None) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    """Read the grid CSV into node temperatures, compositions and assemblages.

    Args:
        path (str | Path): Grid CSV with columns T_K, x_Zr and phases.
        names_as (str | None): Registry TDB whose phase names the grid uses; None keeps the names as given.

    Returns:
        tuple[np.ndarray, np.ndarray, list[list[str]]]: Node temperatures in K, node Zr mole fractions and the sorted
        classes of each node.
    """
    reg = pid.Registry() if names_as else None
    temps, xs, assemblages = [], [], []
    with Path(path).open() as f:
        for r in csv.DictReader(f):
            names = [n.strip() for n in r["phases"].replace("+", ";").split(";") if n.strip()]
            cls = set()
            for n in names:
                if reg is not None:
                    rid = reg.tdb_name(names_as, n)
                    cls.add(pid.bench_class(rid) if rid else f"model:{n}")
                else:
                    cls.add(n)
            temps.append(float(r["T_K"]))
            xs.append(float(r["x_Zr"]))
            assemblages.append(sorted(cls))
    return np.array(temps), np.array(xs), assemblages


def predict(temps: np.ndarray, xs: np.ndarray, assemblages: list[list[str]], points: list[dict]) -> tuple[list[list[str]], int]:
    """Assemblage predicted for each evidence point from the grid.

    Args:
        temps (np.ndarray): Node temperatures in K.
        xs (np.ndarray): Node mole fractions of Zr.
        assemblages (list[list[str]]): Classes of each node.
        points (list[dict]): Evidence points with "kind", "T_K" (K) and "x_Zr".

    Returns:
        tuple[list[list[str]], int]: Predicted classes per point, and the number of boundary bands that held no node.
    """
    preds, empty = [], 0
    for p in points:
        d = ((temps - p["T_K"]) / 1000.0) ** 2 + (xs - p["x_Zr"]) ** 2
        if p["kind"] == "field":
            preds.append(assemblages[int(np.argmin(d))])
            continue
        sel = np.where((np.abs(temps - p["T_K"]) <= grid_dt + 1e-9) & (np.abs(xs - p["x_Zr"]) <= grid_dx + 1e-9))[0]
        if not len(sel):
            empty += 1
            sel = [int(np.argmin(d))]
        s = set()
        for i in sel:
            s.update(assemblages[i])
        preds.append(sorted(s))
    return preds, empty


def main() -> None:
    """Score a grid CSV against the evidence and write its result JSON."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid")
    ap.add_argument("--label", default=None)
    ap.add_argument("--names-as", default=None)
    ap.add_argument("--evidence", default=str(example_root / "inputs" / "evidence.json"))
    ap.add_argument("-o", default=None)
    a = ap.parse_args()
    label = a.label or Path(a.grid).stem
    ev = json.loads(Path(a.evidence).read_text())
    pts = ev["points"]
    temps, xs, assemblages = load_grid(a.grid, a.names_as)
    preds, empty = predict(temps, xs, assemblages, pts)
    model_cls = set().union(*assemblages)
    rng = np.random.default_rng(seed)
    metrics = {}
    for kind in ("field", "boundary"):
        for subset in ("all", "representable"):
            sel = [
                i for i, p in enumerate(pts) if p["kind"] == kind and (subset == "all" or set(p["truth_physical"]) <= model_cls)
            ]
            truth = [pts[i]["truth_physical"] for i in sel]
            pred = [preds[i] for i in sel]
            mb = metric_block(truth, pred, rng)
            if mb and kind == "boundary":
                mb["ci95_cluster_equilibrium"] = cluster_ci(
                    truth, pred, [pts[i]["provenance"]["observation_id"] for i in sel], rng
                )
                mb["ci95_cluster_source"] = cluster_ci(truth, pred, [source_key(pts[i]) for i in sel], rng)
            metrics.setdefault(kind, {})[subset] = mb
    out = a.o or str(example_root / "results" / f"{label}.json")
    res = {
        "label": label,
        "grid": Path(a.grid).name,
        "grid_sha256": file_sha256(a.grid),
        "names_as": a.names_as,
        "evidence": {"orchid_data_release": release_commit, "path": rel(a.evidence), "sha256": file_sha256(a.evidence)},
        "grid_nodes": int(len(temps)),
        "boundary_bands_without_nodes": empty,
        "model_classes": sorted(model_cls),
        "rule": "field: nearest node; boundary: union over nodes within T +- 10 K, x +- 0.01 (nearest node if none)",
        "metrics": {"physical": metrics},
        "points": [
            {
                "id": p["id"],
                "kind": p["kind"],
                "T_K": p["T_K"],
                "x_Zr": p["x_Zr"],
                "truth_physical": p["truth_physical"],
                "pred_physical": q,
                "recall_point": len(set(p["truth_physical"]) & set(q)) / len(p["truth_physical"]),
            }
            for p, q in zip(pts, preds, strict=True)
        ],
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open("w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    for kind in ("field", "boundary"):
        b = metrics[kind]["all"]
        w, c = b["weighted"], b["ci95"]
        print(
            f"{label:22s} {kind:8s} n={b['n_points']:3d} recall {w['recall']:.3f} [{c['recall'][0]:.3f},{c['recall'][1]:.3f}] "
            f"prec {w['precision']:.3f} F1 {w['f1']:.3f} exact {b['exact_assemblage_accuracy']:.3f}"
        )
    print(f"  -> {out}  (grid nodes {len(temps)}, empty boundary bands {empty})")


if __name__ == "__main__":
    main()
