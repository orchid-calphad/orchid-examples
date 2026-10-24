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
import argparse, hashlib, json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_identity as pid
from release import RELEASE, rel  # noqa: E402
from phase_metrics import phase_presence_metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NP_MIN = 1e-6
PDENS = 500
X_CLIP = 1e-6
BOX_DT = (-10.0, -5.0, 0.0, 5.0, 10.0)
BOX_DX = (-0.01, -0.005, 0.0, 0.005, 0.01)
NBOOT, SEED = 2000, 0
RETRY_DX = (1e-4, -1e-4, 1e-3, -1e-3)
COMPS = ["CU", "ZR", "VA"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class Model:
    def __init__(self, tdb, names_as=None):
        from pycalphad import Database
        self.tdb, self.db = tdb, Database(tdb)
        hints = {n: p.model_hints for n, p in self.db.phases.items()}
        self.dis_parts = {h["disordered_phase"] for n, h in hints.items() if h.get("ordered_phase") == n and h.get("disordered_phase") and h["disordered_phase"] != n}
        self.ordered = {n for n, h in hints.items() if h.get("ordered_phase") == n and h.get("disordered_phase") and h["disordered_phase"] != n}
        self.phases = sorted(p for p in self.db.phases if p not in self.dis_parts and p != "GAS")
        self.reg = pid.Registry()
        self.names_as = names_as or os.path.basename(tdb)
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

    def reg_id(self, name, disordered=False):
        if disordered:
            return self.ordered_dis_reg[name]
        return self.reg.tdb_name(self.names_as, name) or f"model:{name}"

    def mapping_table(self):
        rows = []
        for n in self.phases:
            r = {"model_phase": n, "registry_id": self.reg_id(n), "class_physical": pid.bench_class(self.reg_id(n)),
                 "class_strict": pid.bench_class(self.reg_id(n), "strict"), "names_as": self.names_as}
            if n in self.ordered:
                r["order_disorder"] = {"disordered_part": self.db.phases[n].model_hints["disordered_phase"],
                                       "if_disordered_registry_id": self.ordered_dis_reg[n], "order_tol": pid.ORDER_TOL}
            rows.append(r)
        for n in sorted(self.dis_parts):
            rows.append({"model_phase": n, "excluded": "disordered part of an order/disorder model (not a separate phase)"})
        return rows

    def equilibrium_drop_one(self, T, x):
        """Fallback for a cell where the full solve returns NaN: solve with each phase left out in turn and keep the
        lowest-GM successful solution. Exact when the stable assemblage has fewer phases than the model (always, in a
        binary interior cell): any subset that keeps the stable phases has the same minimum."""
        best, bestg = [], np.inf
        for drop in self.phases:
            a, g = self.equilibrium([T], [x], phases=[p for p in self.phases if p != drop], return_gm=True)
            if a[0, 0] and np.isfinite(g[0, 0]) and g[0, 0] < bestg - 1e-6:
                best, bestg = a[0, 0], g[0, 0]
        return best

    def equilibrium(self, Ts, xs, phases=None, return_gm=False):
        """Assemblages (lists of (phase, disordered_flag)) on the grid Ts x xs."""
        from pycalphad import equilibrium, variables as v
        xs = np.clip(np.asarray(xs, float), X_CLIP, 1 - X_CLIP)
        r = equilibrium(self.db, COMPS, phases or self.phases, {v.T: list(map(float, Ts)), v.P: 101325, v.N: 1, v.X("ZR"): list(xs)},
                        calc_opts={"pdens": PDENS})
        GM = r.GM.values.squeeze(axis=(0, 1))
        Ph = r.Phase.values.squeeze(axis=(0, 1))   # N, P squeezed -> (T, X, vertex)
        NP = r.NP.values.squeeze(axis=(0, 1))
        Y = r.Y.values.squeeze(axis=(0, 1))
        out = np.empty(Ph.shape[:2], dtype=object)
        for i in range(Ph.shape[0]):
            for j in range(Ph.shape[1]):
                s = set()
                for k in range(Ph.shape[2]):
                    name = str(Ph[i, j, k])
                    if not name or not np.isfinite(NP[i, j, k]) or NP[i, j, k] <= NP_MIN:
                        continue
                    dis = False
                    if name in self.ordered:
                        y = Y[i, j, k]
                        # site fractions laid out by sublattice in the phase's constituent order
                        ph = self.db.phases[name]
                        off, ys = 0, []
                        for cons in ph.constituents:
                            cs = sorted(str(c) for c in cons)
                            ys.append(dict(zip(cs, y[off:off + len(cs)]))); off += len(cs)
                        a, b = self.ord_info[name][:2]
                        dis = abs(ys[a].get("ZR", 0) - ys[b].get("ZR", 0)) < pid.ORDER_TOL
                    s.add((name, dis))
                out[i, j] = sorted(s)
        return (out, GM) if return_gm else out


def predict(model, points):
    """Raw model assemblages per point: field -> one equilibrium; boundary -> union over the tolerance box."""
    # gather all (T, x) requests, grouped by T
    req = {}
    for n, p in enumerate(points):
        Ts = [p["T_K"]] if p["kind"] == "field" else [p["T_K"] + d for d in BOX_DT]
        xs = [p["x_Zr"]] if p["kind"] == "field" else [p["x_Zr"] + d for d in BOX_DX]
        for T in Ts:
            req.setdefault(round(T, 6), set()).update(round(float(np.clip(x, X_CLIP, 1 - X_CLIP)), 9) for x in xs)
    res = {}
    for T, xs in req.items():
        xs = sorted(xs)
        a = model.equilibrium([T], xs)
        for j, x in enumerate(xs):
            res[(T, x)] = a[0, j]
    # retry cells where the solver returned no phase (NaN): alone; then drop-one-phase minimum; then x nudged (recorded)
    model.retries = {"cells_failed_first_pass": 0, "recovered_alone": 0, "recovered_drop_one_min_gm": 0,
                     "recovered_nudged": 0, "still_failed": 0}
    for (T, x), c in list(res.items()):
        if c:
            continue
        model.retries["cells_failed_first_pass"] += 1
        c = model.equilibrium([T], [x])[0, 0]
        if c:
            model.retries["recovered_alone"] += 1
        else:
            c = model.equilibrium_drop_one(T, x)
            if c:
                model.retries["recovered_drop_one_min_gm"] += 1
        if not c:
            for dx in RETRY_DX:
                c = model.equilibrium([T], [x + dx])[0, 0]
                if c:
                    model.retries["recovered_nudged"] += 1
                    break
            else:
                model.retries["still_failed"] += 1
        res[(T, x)] = c
    preds, fails = [], 0
    for p in points:
        Ts = [p["T_K"]] if p["kind"] == "field" else [p["T_K"] + d for d in BOX_DT]
        xs = [p["x_Zr"]] if p["kind"] == "field" else [p["x_Zr"] + d for d in BOX_DX]
        s, nf = set(), 0
        for T in Ts:
            for x in xs:
                c = res[(round(T, 6), round(float(np.clip(x, X_CLIP, 1 - X_CLIP)), 9))]
                if not c:
                    nf += 1
                s.update(c)
        fails += nf
        preds.append((sorted(s), nf))
    return preds, fails


def classes(model, raw, mode):
    return sorted({pid.bench_class(model.reg_id(n, d), mode) for n, d in raw})


def metric_block(truth, pred, rng):
    if not truth:
        return None
    m = phase_presence_metrics(truth, pred)
    exact = np.array([set(t) == set(q) for t, q in zip(truth, pred)], float)
    n = len(truth)
    boots = {k: [] for k in ("recall", "precision", "f1", "accuracy", "exact")}
    for _ in range(NBOOT):
        idx = rng.integers(0, n, n)
        mb = phase_presence_metrics([truth[i] for i in idx], [pred[i] for i in idx])
        for k in ("recall", "precision", "f1", "accuracy"):
            boots[k].append(mb["weighted"][k])
        boots["exact"].append(exact[idx].mean())
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boots.items()}
    return {"n_points": n, "weighted": m["weighted"], "macro_ground_truth_phases": m["macro_ground_truth_phases"],
            "exact_assemblage_accuracy": float(exact.mean()), "ci95": ci, "per_phase": m["per_phase"]}


def source_key(p):
    """Cluster key of a literature point's original source, from the reference string: '<surname> <year>' (so that
    '1953, C.E. Lundin, ...' and '1953, Lundin CE, ...' agree, and 'Liu 2017, doi:...' rows with or without a
    measured_in field agree); a reference that is only a DOI keys by that DOI. Years are taken as printed."""
    import re
    ref = (p.get("provenance", {}).get("reference") or "").strip()
    if ref.lower().startswith("doi:"):
        return ref.split()[0].lower().rstrip(",")
    yr = re.search(r"(18|19|20)\d\d", ref)
    names = re.findall(r"[A-Z][a-z]{2,}", ref)
    return f"{names[0].lower() if names else ref[:20]} {yr.group(0) if yr else ''}".strip()


def cluster_ci(truth, pred, clusters, rng, nboot=NBOOT):
    """Percentile CI of weighted recall and exact accuracy with whole clusters resampled (cluster bootstrap)."""
    keys = sorted(set(clusters)); idx_of = {k: [i for i, c in enumerate(clusters) if c == k] for k in keys}
    rec, ex = [], []
    for _ in range(nboot):
        pick = rng.integers(0, len(keys), len(keys))
        idx = [i for k in pick for i in idx_of[keys[k]]]
        rec.append(phase_presence_metrics([truth[i] for i in idx], [pred[i] for i in idx])["weighted"]["recall"])
        ex.append(np.mean([set(truth[i]) == set(pred[i]) for i in idx]))
    return {"n_clusters": len(keys), "recall": [float(np.percentile(rec, 2.5)), float(np.percentile(rec, 97.5))],
            "exact": [float(np.percentile(ex, 2.5)), float(np.percentile(ex, 97.5))]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tdb")
    ap.add_argument("--evidence", default=os.path.join(ROOT, "inputs", "evidence.json"))
    ap.add_argument("--label", default=None)
    ap.add_argument("--names-as", default=None, help="registry TDB file whose phase names this TDB uses")
    ap.add_argument("--in-sample-note", default=None)
    ap.add_argument("-o", default=None)
    a = ap.parse_args()
    label = a.label or os.path.splitext(os.path.basename(a.tdb))[0]
    out = a.o or os.path.join(ROOT, "results", f"{label}.json")
    import pycalphad
    ev = json.load(open(a.evidence))
    pts = ev["points"]
    t0 = time.time()
    model = Model(a.tdb, a.names_as)
    raw, fails = predict(model, pts)
    wall = time.time() - t0
    model_cls = {m: {pid.bench_class(model.reg_id(n), m) for n in model.phases} |
                 ({pid.bench_class(model.ordered_dis_reg[n], m) for n in model.ordered}) for m in ("physical", "strict")}
    per_point, metrics = [], {}
    for mode in ("physical", "strict"):
        rng = np.random.default_rng(SEED)
        for kind in ("field", "boundary"):
            for subset in ("all", "representable"):
                sel = [i for i, p in enumerate(pts) if p["kind"] == kind and
                       (subset == "all" or set(p[f"truth_{mode}"]) <= model_cls[mode])]
                truth = [pts[i][f"truth_{mode}"] for i in sel]
                pred = [classes(model, raw[i][0], mode) for i in sel]
                mb = metric_block(truth, pred, rng)
                if mb and kind == "boundary":
                    mb["ci95_cluster_equilibrium"] = cluster_ci(truth, pred, [pts[i]["provenance"]["observation_id"] for i in sel], rng)
                    mb["ci95_cluster_source"] = cluster_ci(truth, pred, [source_key(pts[i]) for i in sel], rng)
                metrics.setdefault(mode, {}).setdefault(kind, {})[subset] = mb
        # in/out of the final TDB's fit set (relevant for the tuned model only; reported for every model)
        for split in ("fit", "out_of_fit"):
            for kind in ("field", "boundary"):
                sel = [i for i, p in enumerate(pts) if p["kind"] == kind and
                       (p["final_tdb_split"].startswith("fit") == (split == "fit"))]
                metrics[mode].setdefault("by_final_fit_split", {}).setdefault(split, {})[kind] = metric_block(
                    [pts[i][f"truth_{mode}"] for i in sel], [classes(model, raw[i][0], mode) for i in sel], rng)
    for p, (r, nf) in zip(pts, raw):
        tp, pp = set(p["truth_physical"]), set(classes(model, r, "physical"))
        per_point.append({"id": p["id"], "kind": p["kind"], "T_K": p["T_K"], "x_Zr": p["x_Zr"],
                          "truth_physical": sorted(tp), "pred_physical": sorted(pp),
                          "truth_strict": p["truth_strict"], "pred_strict": classes(model, r, "strict"),
                          "pred_model_phases": [n + ("(disordered)" if d else "") for n, d in r],
                          "recall_point": len(tp & pp) / len(tp), "exact": tp == pp, "failed_cells": nf,
                          "final_tdb_split": p["final_tdb_split"]})
    res = {
        "label": label, "tdb": rel(a.tdb), "tdb_sha256": sha(a.tdb), "names_as": model.names_as,
        "in_sample_note": a.in_sample_note,
        "evidence": {"orchid_data_release": RELEASE, "path": rel(a.evidence), "sha256": sha(a.evidence), "counts": ev["counts"]},
        "settings": {"software": {"pycalphad": pycalphad.__version__, "numpy": np.__version__, "python": sys.version.split()[0]},
                     "solver": "pycalphad.equilibrium default Solver", "calc_opts": {"pdens": PDENS},
                     "components": COMPS, "phases": model.phases, "P_Pa": 101325, "N": 1, "np_min": NP_MIN,
                     "x_clip": X_CLIP, "boundary_box": {"dT_K": BOX_DT, "dx": BOX_DX, "rule": "union of assemblages"},
                     "bootstrap": {"B": NBOOT, "seed": SEED, "unit": "point", "interval": "percentile 2.5-97.5",
                                   "boundary_cluster_bootstraps": "ci95_cluster_equilibrium (whole tie-line/invariant, ORCHID observation id) and ci95_cluster_source (original measuring study, source_key())"},
                     "metric_function": "phase_metrics.phase_presence_metrics (PI, verbatim copy)",
                     "primary_metric": "prevalence-weighted recall (weighted.recall)"},
        "phase_mapping_model": model.mapping_table(),
        "model_classes": {k: sorted(v) for k, v in model_cls.items()},
        "equilibrium_failed_cells_after_retry": fails, "equilibrium_retries": model.retries,
        "retry_rule": f"cells with no phase are re-solved alone; then with each phase left out in turn, keeping the lowest-GM solution (exact for <= n-1 stable phases); then at x + {RETRY_DX}", "wall_s": wall,
        "metrics": metrics, "points": per_point,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    m = metrics["physical"]
    for kind in ("field", "boundary"):
        b = m[kind]["all"]
        w, c = b["weighted"], b["ci95"]
        cl = "" if kind == "field" else (f" | eq-cluster [{b['ci95_cluster_equilibrium']['recall'][0]:.3f},{b['ci95_cluster_equilibrium']['recall'][1]:.3f}]"
                                         f" source-cluster(n={b['ci95_cluster_source']['n_clusters']}) [{b['ci95_cluster_source']['recall'][0]:.3f},{b['ci95_cluster_source']['recall'][1]:.3f}]")
        print(f"{label:22s} {kind:8s} n={b['n_points']:3d} recall {w['recall']:.3f} [{c['recall'][0]:.3f},{c['recall'][1]:.3f}]{cl} "
              f"prec {w['precision']:.3f} F1 {w['f1']:.3f} acc {w['accuracy']:.3f} exact {b['exact_assemblage_accuracy']:.3f}")
    print(f"  -> {out}  ({wall:.0f} s, retries {model.retries}, failed cells {fails})")


if __name__ == "__main__":
    main()
