# Copied from orchid-bench-pd scripts/score.py; evidence and provenance paths made repository-relative.
"""Score a Cu-Zr TDB on the ORCHID-bench one-shot phase-diagram track (ZPF as classification).

  python score.py MODEL.tdb [--evidence evidence.json] [--label NAME] [--names-as CuZr-94Zen-LB.tdb]
                  [--in-sample-note TEXT] [-o results/<label>.json]

For every evidence point the model's predicted assemblage is the set of phases with phase fraction NP > NP_MIN in a
pycalphad equilibrium at (T, x_Zr, P = 101325 Pa, N = 1):
  field points    : one equilibrium at the point;
  boundary points : the union of the assemblages on a 5 x 5 grid over T +- 10 K, x +- 0.01 (the tolerance band);
                    a model scores a listed phase there when the phase is present somewhere within the band.
Model phase names -> ORCHID registry id ('tdb:<file>:<NAME>' names of the registry; `--names-as` for a TDB that is not
in the registry but uses another file's phase names and sublattice models, e.g. the PhaseForge TDBs = CuZr-94Zen-LB)
-> benchmark class (phase_identity.py; modes `physical` (primary) and `strict`). A phase name the registry does not
know becomes its own class `model:<NAME>` (it can only add false positives).

Metrics: phase_presence_metrics(truth, predicted) (PI's function, copied verbatim in phase_metrics.py); primary metric
= prevalence-weighted recall. Also weighted precision / F1 / accuracy, macro, per-phase table, exact-assemblage accuracy
(predicted set == reference set). Bootstrap 95% CIs: points resampled with replacement (B = 2000, seed 0), percentile.
Reported for field and boundary points separately, each for `all` points and for `representable` points (every
reference class exists in the model's phase list).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parent))
import phase_identity as pid  # noqa: E402
from release import example_root, file_sha256, rel, release_commit  # noqa: E402
from score_metrics import cluster_ci, metric_block, nboot, seed, source_key  # noqa: E402

warnings.filterwarnings("ignore")

np_min = 1e-6  # phase fraction above which a phase counts as present
pdens = 500  # pycalphad point density of the equilibrium calculation
x_clip = 1e-6  # mole fraction at which the composition is clipped away from the pure end members
pressure_pa = 101325  # Pa, 1 atm
comps = ["CU", "ZR", "VA"]
box_dt = (-10.0, -5.0, 0.0, 5.0, 10.0)  # K, offsets of the boundary tolerance band in T
box_dx = (-0.01, -0.005, 0.0, 0.005, 0.01)  # mole fraction, offsets of the boundary tolerance band in x
retry_dx = (1e-4, -1e-4, 1e-3, -1e-3)  # mole-fraction nudges tried for a cell the solver fails on


class Model:
    """Equilibrium model of one Cu-Zr TDB, with its phases mapped to benchmark classes.

    Args:
        tdb (str): Path of the TDB file.
        names_as (str | None): Registry TDB whose phase names this TDB uses. Defaults to the TDB's own file name.
    """

    def __init__(self, tdb: str, names_as: str | None = None) -> None:
        """Parse the TDB and index its order/disorder phases and registry ids."""
        from pycalphad import Database  # noqa: PLC0415 (lazy: the cached quick path does not need pycalphad)

        self.tdb = tdb
        self.db = Database(tdb)
        hints = {n: p.model_hints for n, p in self.db.phases.items()}
        self.dis_parts = {
            h["disordered_phase"]
            for n, h in hints.items()
            if h.get("ordered_phase") == n and h.get("disordered_phase") and h["disordered_phase"] != n
        }
        self.ordered = {
            n
            for n, h in hints.items()
            if h.get("ordered_phase") == n and h.get("disordered_phase") and h["disordered_phase"] != n
        }
        self.phases = sorted(p for p in self.db.phases if p not in self.dis_parts and p != "GAS")
        self.reg = pid.Registry()
        self.names_as = names_as or Path(tdb).name
        self.ordered_dis_reg = {}
        for n in self.ordered:
            dis = hints[n]["disordered_phase"]
            self.ordered_dis_reg[n] = self.reg.tdb_name(self.names_as, dis) or "phase:cu-zr:bcc-a2"
        # substitutional sublattice indices of ordered phases (for the order check)
        self.ord_info = {}
        for n in self.ordered:
            ph = self.db.phases[n]
            subl = [sorted(str(c) for c in cons) for cons in ph.constituents]
            self.ord_info[n] = [i for i, s in enumerate(subl) if s != ["VA"]]

    def reg_id(self, name: str, disordered: bool = False) -> str:
        """Registry id of a model phase.

        Args:
            name (str): Phase name in the TDB.
            disordered (bool): True for the disordered part of an order/disorder phase.

        Returns:
            str: Registry id, or "model:<name>" when the registry does not know the phase.
        """
        if disordered:
            return self.ordered_dis_reg[name]
        return self.reg.tdb_name(self.names_as, name) or f"model:{name}"

    def mapping_table(self) -> list[dict]:
        """Phase-name mapping of every model phase, for the result file.

        Returns:
            list[dict]: One row per model phase with its registry id and benchmark classes in both modes, plus the
            order/disorder details of ordered phases and rows for the excluded disordered parts.
        """
        rows = []
        for n in self.phases:
            r = {
                "model_phase": n,
                "registry_id": self.reg_id(n),
                "class_physical": pid.bench_class(self.reg_id(n)),
                "class_strict": pid.bench_class(self.reg_id(n), "strict"),
                "names_as": self.names_as,
            }
            if n in self.ordered:
                r["order_disorder"] = {
                    "disordered_part": self.db.phases[n].model_hints["disordered_phase"],
                    "if_disordered_registry_id": self.ordered_dis_reg[n],
                    "order_tol": pid.order_tol,
                }
            rows.append(r)
        for n in sorted(self.dis_parts):
            rows.append({"model_phase": n, "excluded": "disordered part of an order/disorder model (not a separate phase)"})
        return rows

    def equilibrium_drop_one(self, temp: float, x: float) -> list:
        """Assemblage of one cell by solving with each phase left out in turn.

        Fallback for a cell where the full solve returns NaN. Exact when the stable assemblage has fewer phases than
        the model (always, in a binary interior cell): any subset that keeps the stable phases has the same minimum.

        Args:
            temp (float): Temperature in K.
            x (float): Mole fraction of Zr.

        Returns:
            list: Assemblage as a sorted list of (phase, disordered_flag) pairs; empty when every solve fails.
        """
        best, bestg = [], np.inf
        for drop in self.phases:
            a, g = self.equilibrium([temp], [x], phases=[p for p in self.phases if p != drop], return_gm=True)
            if a[0, 0] and np.isfinite(g[0, 0]) and g[0, 0] < bestg - 1e-6:
                best, bestg = a[0, 0], g[0, 0]
        return best

    def equilibrium(
        self, temps: list[float], xs: list[float], phases: list[str] | None = None, return_gm: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Assemblages on the grid of temperatures and compositions.

        Args:
            temps (list[float]): Temperatures in K.
            xs (list[float]): Mole fractions of Zr, clipped to [x_clip, 1 - x_clip].
            phases (list[str] | None): Phases allowed in the equilibrium; defaults to all model phases.
            return_gm (bool): Also return the molar Gibbs energy grid, in J/mol of atoms.

        Returns:
            np.ndarray | tuple[np.ndarray, np.ndarray]: Object array (temperature, composition) of sorted lists of
            (phase, disordered_flag) pairs; with return_gm, also the Gibbs energy grid.
        """
        from pycalphad import equilibrium  # noqa: PLC0415 (lazy, see Model.__init__)
        from pycalphad import variables as v  # noqa: PLC0415 (lazy, see Model.__init__)

        xs = np.clip(np.asarray(xs, float), x_clip, 1 - x_clip)
        r = equilibrium(
            self.db,
            comps,
            phases or self.phases,
            {v.T: list(map(float, temps)), v.P: pressure_pa, v.N: 1, v.X("ZR"): list(xs)},
            calc_opts={"pdens": pdens},
        )
        gm = r.GM.values.squeeze(axis=(0, 1))
        phase_arr = r.Phase.values.squeeze(axis=(0, 1))  # N, P squeezed -> (T, X, vertex)
        np_arr = r.NP.values.squeeze(axis=(0, 1))
        site_frac = r.Y.values.squeeze(axis=(0, 1))
        out = np.empty(phase_arr.shape[:2], dtype=object)
        for i in range(phase_arr.shape[0]):
            for j in range(phase_arr.shape[1]):
                s = set()
                for k in range(phase_arr.shape[2]):
                    name = str(phase_arr[i, j, k])
                    if not name or not np.isfinite(np_arr[i, j, k]) or np_arr[i, j, k] <= np_min:
                        continue
                    dis = False
                    if name in self.ordered:
                        y = site_frac[i, j, k]
                        # site fractions laid out by sublattice in the phase's constituent order
                        ph = self.db.phases[name]
                        off, ys = 0, []
                        for cons in ph.constituents:
                            cs = sorted(str(c) for c in cons)
                            ys.append(dict(zip(cs, y[off : off + len(cs)], strict=True)))
                            off += len(cs)
                        a, b = self.ord_info[name][:2]
                        dis = abs(ys[a].get("ZR", 0) - ys[b].get("ZR", 0)) < pid.order_tol
                    s.add((name, dis))
                out[i, j] = sorted(s)
        return (out, gm) if return_gm else out


def predict(model: Model, points: list[dict]) -> tuple[list[tuple[list, int]], int]:
    """Raw model assemblages per evidence point.

    A field point takes one equilibrium at the point. A boundary point takes the union of the assemblages over the
    tolerance box. Cells where the solver returns no phase are retried in three steps, and the retry counts are
    recorded on the model.

    Args:
        model (Model): Model to evaluate.
        points (list[dict]): Evidence points with "kind", "T_K" (K) and "x_Zr".

    Returns:
        tuple[list[tuple[list, int]], int]: Per point, the sorted (phase, disordered_flag) assemblage and the number of
        failed cells in its box; and the total number of failed cells after retry.
    """
    # gather all (T, x) requests, grouped by T
    req = {}
    for p in points:
        temps = [p["T_K"]] if p["kind"] == "field" else [p["T_K"] + d for d in box_dt]
        xs = [p["x_Zr"]] if p["kind"] == "field" else [p["x_Zr"] + d for d in box_dx]
        for temp in temps:
            req.setdefault(round(temp, 6), set()).update(round(float(np.clip(x, x_clip, 1 - x_clip)), 9) for x in xs)
    res = {}
    for temp, xs_set in req.items():
        xs = sorted(xs_set)
        a = model.equilibrium([temp], xs)
        for j, x in enumerate(xs):
            res[(temp, x)] = a[0, j]
    _refill_failed_cells(model, res)
    preds, fails = [], 0
    for p in points:
        temps = [p["T_K"]] if p["kind"] == "field" else [p["T_K"] + d for d in box_dt]
        xs = [p["x_Zr"]] if p["kind"] == "field" else [p["x_Zr"] + d for d in box_dx]
        s, nf = set(), 0
        for temp in temps:
            for x in xs:
                cell = res[(round(temp, 6), round(float(np.clip(x, x_clip, 1 - x_clip)), 9))]
                if not cell:
                    nf += 1
                s.update(cell)
        fails += nf
        preds.append((sorted(s), nf))
    return preds, fails


def _refill_failed_cells(model: Model, res: dict) -> None:
    """Re-solve the cells the first pass left without a phase, in three steps, recording each outcome.

    Steps: the cell alone; then with each phase left out in turn, keeping the lowest-GM solution; then with x nudged.

    Args:
        model (Model): Model that solves the cells; its retry counters are set on it here.
        res (dict): (T, x) -> assemblage of every cell, updated in place.
    """
    model.retries = {
        "cells_failed_first_pass": 0,
        "recovered_alone": 0,
        "recovered_drop_one_min_gm": 0,
        "recovered_nudged": 0,
        "still_failed": 0,
    }
    for (temp, x), first in list(res.items()):
        if first:
            continue
        model.retries["cells_failed_first_pass"] += 1
        filled = model.equilibrium([temp], [x])[0, 0]
        if filled:
            model.retries["recovered_alone"] += 1
        else:
            filled = model.equilibrium_drop_one(temp, x)
            if filled:
                model.retries["recovered_drop_one_min_gm"] += 1
        if not filled:
            for dx in retry_dx:
                filled = model.equilibrium([temp], [x + dx])[0, 0]
                if filled:
                    model.retries["recovered_nudged"] += 1
                    break
            else:
                model.retries["still_failed"] += 1
        res[(temp, x)] = filled


def classes(model: Model, raw: list, mode: str) -> list[str]:
    """Benchmark classes of a raw assemblage.

    Args:
        model (Model): Model that produced the assemblage.
        raw (list): (phase, disordered_flag) pairs.
        mode (str): "physical" or "strict" class mode.

    Returns:
        list[str]: Sorted, de-duplicated benchmark classes.
    """
    return sorted({pid.bench_class(model.reg_id(n, d), mode) for n, d in raw})


def _model_classes(model: Model) -> dict[str, set[str]]:
    """Benchmark classes of the model's phases in both modes.

    Args:
        model (Model): Scored model.

    Returns:
        dict[str, set[str]]: Classes per mode ("physical", "strict"), including the disordered parts of ordered phases.
    """
    return {
        m: {pid.bench_class(model.reg_id(n), m) for n in model.phases}
        | {pid.bench_class(model.ordered_dis_reg[n], m) for n in model.ordered}
        for m in ("physical", "strict")
    }


def _metrics(model: Model, pts: list[dict], raw: list, model_cls: dict[str, set[str]]) -> dict:
    """Metric blocks for every mode, kind and subset, plus the split by the final TDB's fit set.

    The bootstrap generator is re-seeded per mode and the blocks are computed in a fixed order, so the intervals are
    reproducible.

    Args:
        model (Model): Scored model.
        pts (list[dict]): Evidence points.
        raw (list): Per-point predictions from predict().
        model_cls (dict[str, set[str]]): Model classes per mode, from _model_classes().

    Returns:
        dict: Metric blocks keyed by mode, then kind ("field", "boundary"), then subset ("all", "representable"), and
        "by_final_fit_split" blocks per mode.
    """
    metrics = {}
    for mode in ("physical", "strict"):
        rng = np.random.default_rng(seed)
        for kind in ("field", "boundary"):
            for subset in ("all", "representable"):
                sel = [
                    i
                    for i, p in enumerate(pts)
                    if p["kind"] == kind and (subset == "all" or set(p[f"truth_{mode}"]) <= model_cls[mode])
                ]
                truth = [pts[i][f"truth_{mode}"] for i in sel]
                pred = [classes(model, raw[i][0], mode) for i in sel]
                mb = metric_block(truth, pred, rng)
                if mb and kind == "boundary":
                    mb["ci95_cluster_equilibrium"] = cluster_ci(
                        truth, pred, [pts[i]["provenance"]["observation_id"] for i in sel], rng
                    )
                    mb["ci95_cluster_source"] = cluster_ci(truth, pred, [source_key(pts[i]) for i in sel], rng)
                metrics.setdefault(mode, {}).setdefault(kind, {})[subset] = mb
        # in/out of the final TDB's fit set (relevant for the tuned model only; reported for every model)
        for split in ("fit", "out_of_fit"):
            for kind in ("field", "boundary"):
                sel = [
                    i
                    for i, p in enumerate(pts)
                    if p["kind"] == kind and (p["final_tdb_split"].startswith("fit") == (split == "fit"))
                ]
                metrics[mode].setdefault("by_final_fit_split", {}).setdefault(split, {})[kind] = metric_block(
                    [pts[i][f"truth_{mode}"] for i in sel], [classes(model, raw[i][0], mode) for i in sel], rng
                )
    return metrics


def _per_point(model: Model, pts: list[dict], raw: list) -> list[dict]:
    """Per-point records of the result file.

    Args:
        model (Model): Scored model.
        pts (list[dict]): Evidence points.
        raw (list): Per-point predictions from predict().

    Returns:
        list[dict]: One record per point with the truth and the prediction in both modes, the per-point recall and
        the number of failed cells.
    """
    per_point = []
    for p, (r, nf) in zip(pts, raw, strict=True):
        tp, pp = set(p["truth_physical"]), set(classes(model, r, "physical"))
        per_point.append(
            {
                "id": p["id"],
                "kind": p["kind"],
                "T_K": p["T_K"],
                "x_Zr": p["x_Zr"],
                "truth_physical": sorted(tp),
                "pred_physical": sorted(pp),
                "truth_strict": p["truth_strict"],
                "pred_strict": classes(model, r, "strict"),
                "pred_model_phases": [n + ("(disordered)" if d else "") for n, d in r],
                "recall_point": len(tp & pp) / len(tp),
                "exact": tp == pp,
                "failed_cells": nf,
                "final_tdb_split": p["final_tdb_split"],
            }
        )
    return per_point


def _print_summary(label: str, metrics: dict, out: str, *, wall: float, retries: dict, fails: int) -> None:
    """Print the physical-mode summary line of each kind.

    Args:
        label (str): Model label.
        metrics (dict): Metric blocks from _metrics().
        out (str): Path of the written result file.
        wall (float): Wall time of the prediction in seconds.
        retries (dict): Retry counters of the prediction.
        fails (int): Failed cells after retry.
    """
    m = metrics["physical"]
    for kind in ("field", "boundary"):
        b = m[kind]["all"]
        w, c = b["weighted"], b["ci95"]
        cl = ""
        if kind == "boundary":
            eq, src = b["ci95_cluster_equilibrium"], b["ci95_cluster_source"]
            cl = (
                f" | eq-cluster [{eq['recall'][0]:.3f},{eq['recall'][1]:.3f}]"
                f" source-cluster(n={src['n_clusters']}) [{src['recall'][0]:.3f},{src['recall'][1]:.3f}]"
            )
        print(
            f"{label:22s} {kind:8s} n={b['n_points']:3d} recall {w['recall']:.3f} [{c['recall'][0]:.3f},{c['recall'][1]:.3f}]{cl} "
            f"prec {w['precision']:.3f} F1 {w['f1']:.3f} acc {w['accuracy']:.3f} exact {b['exact_assemblage_accuracy']:.3f}"
        )
    print(f"  -> {out}  ({wall:.0f} s, retries {retries}, failed cells {fails})")


def main() -> None:
    """Score one TDB against the evidence file and write its result JSON."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tdb")
    ap.add_argument("--evidence", default=str(example_root / "inputs" / "evidence.json"))
    ap.add_argument("--label", default=None)
    ap.add_argument("--names-as", default=None, help="registry TDB file whose phase names this TDB uses")
    ap.add_argument("--in-sample-note", default=None)
    ap.add_argument("-o", default=None)
    a = ap.parse_args()
    label = a.label or Path(a.tdb).stem
    out = a.o or str(example_root / "results" / f"{label}.json")
    import pycalphad  # noqa: PLC0415 (lazy, see Model.__init__)

    ev = json.loads(Path(a.evidence).read_text(encoding="utf-8"))
    pts = ev["points"]
    t0 = time.time()
    model = Model(a.tdb, a.names_as)
    raw, fails = predict(model, pts)
    wall = time.time() - t0
    model_cls = _model_classes(model)
    metrics = _metrics(model, pts, raw, model_cls)
    per_point = _per_point(model, pts, raw)
    res = {
        "label": label,
        "tdb": rel(a.tdb),
        "tdb_sha256": file_sha256(a.tdb),
        "names_as": model.names_as,
        "in_sample_note": a.in_sample_note,
        "evidence": {
            "orchid_data_release": release_commit,
            "path": rel(a.evidence),
            "sha256": file_sha256(a.evidence),
            "counts": ev["counts"],
        },
        "settings": {
            "software": {"pycalphad": pycalphad.__version__, "numpy": np.__version__, "python": sys.version.split()[0]},
            "solver": "pycalphad.equilibrium default Solver",
            "calc_opts": {"pdens": pdens},
            "components": comps,
            "phases": model.phases,
            "P_Pa": pressure_pa,
            "N": 1,
            "np_min": np_min,
            "x_clip": x_clip,
            "boundary_box": {"dT_K": box_dt, "dx": box_dx, "rule": "union of assemblages"},
            "bootstrap": {
                "B": nboot,
                "seed": seed,
                "unit": "point",
                "interval": "percentile 2.5-97.5",
                "boundary_cluster_bootstraps": "ci95_cluster_equilibrium (whole tie-line/invariant, ORCHID observation id) and ci95_cluster_source (original measuring study, source_key())",
            },
            "metric_function": "phase_metrics.phase_presence_metrics (PI, verbatim copy)",
            "primary_metric": "prevalence-weighted recall (weighted.recall)",
        },
        "phase_mapping_model": model.mapping_table(),
        "model_classes": {k: sorted(v) for k, v in model_cls.items()},
        "equilibrium_failed_cells_after_retry": fails,
        "equilibrium_retries": model.retries,
        "retry_rule": f"cells with no phase are re-solved alone; then with each phase left out in turn, keeping the lowest-GM solution (exact for <= n-1 stable phases); then at x + {retry_dx}",
        "wall_s": wall,
        "metrics": metrics,
        "points": per_point,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with Path(out).open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    _print_summary(label, metrics, out, wall=wall, retries=model.retries, fails=fails)


if __name__ == "__main__":
    main()
