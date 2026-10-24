"""Paired bootstrap of metric differences between two scored models (same evidence points), plus the markdown tables
used in RESULTS.md. Usage: python compare.py [A.json B.json ...]  (default: GRACE vs SGTE and final vs SGTE)
Writes results/comparison.json."""
import json, os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase_metrics import phase_presence_metrics
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda l: json.load(open(os.path.join(ROOT, "results", l + ".json")))
NB, SEED = 2000, 0


def paired(a, b, kind, mode="physical"):
    pa = [p for p in a["points"] if p["kind"] == kind]; pb = {p["id"]: p for p in b["points"] if p["kind"] == kind}
    truth = [p[f"truth_{mode}"] for p in pa]
    A = [p[f"pred_{mode}"] for p in pa]; B = [pb[p["id"]][f"pred_{mode}"] for p in pa]
    rng = np.random.default_rng(SEED); n = len(pa); d = []
    f = lambda t, q: phase_presence_metrics(t, q)["weighted"]["recall"]
    base = f(truth, A) - f(truth, B)
    for _ in range(NB):
        i = rng.integers(0, n, n)
        d.append(f([truth[k] for k in i], [A[k] for k in i]) - f([truth[k] for k in i], [B[k] for k in i]))
    return {"n": n, "delta_weighted_recall": base, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_boot_le0": float(np.mean(np.array(d) >= 0)) if base < 0 else float(np.mean(np.array(d) <= 0))}


out = {}
for a, b in (("cuzr_grace", "cuzr_sgte_94Zen"), ("cuzr_phaseforgeplus_final", "cuzr_sgte_94Zen"), ("cuzr_phaseforgeplus_final", "cuzr_grace")):
    A, B = R(a), R(b)
    out[f"{a}-{b}"] = {k: paired(A, B, k) for k in ("field", "boundary")}
json.dump(out, open(os.path.join(ROOT, "results/comparison.json"), "w"), indent=1)
for k, v in out.items():
    print(k, {kk: (round(vv["delta_weighted_recall"], 3), [round(x, 3) for x in vv["ci95"]]) for kk, vv in v.items()})


def row(lab, d, kind, mode="physical", sub="all"):
    b = d["metrics"][mode][kind][sub]; w, c = b["weighted"], b["ci95"]
    f = lambda k: f"{w[k]:.3f} [{c[k][0]:.2f}, {c[k][1]:.2f}]"
    return f"| {lab} | {b['n_points']} | **{f('recall')}** | {f('precision')} | {f('f1')} | {f('accuracy')} | {b['exact_assemblage_accuracy']:.3f} [{c['exact'][0]:.2f}, {c['exact'][1]:.2f}] |"


print("\n## tables")
for mode in ("physical", "strict"):
    for sub in ("all", "representable"):
        for kind in ("field", "boundary"):
            print(f"\n{mode} / {sub} / {kind}\n| model | n | weighted recall | weighted precision | weighted F1 | weighted accuracy | exact assemblage |\n|---|---|---|---|---|---|---|")
            for lab, name in (("cuzr_grace", "GRACE-2L-OMAT MLIP-only"), ("cuzr_sgte_94Zen", "SGTE CuZr-94Zen-LB"),
                              ("cuzr_phaseforgeplus_final", "PhaseForge+ final (in-sample)")):
                print(row(name, R(lab), kind, mode, sub))
print("\npanel")
print("| TDB | field recall | field exact | boundary recall | boundary exact |\n|---|---|---|---|---|")
for f in sorted(glob.glob(os.path.join(ROOT, "results/reference_panel/*.json"))) + [os.path.join(ROOT, "results/cuzr_sgte_94Zen.json")]:
    d = json.load(open(f)); m = d["metrics"]["physical"]
    g = lambda k: f"{m[k]['all']['weighted']['recall']:.3f} [{m[k]['all']['ci95']['recall'][0]:.2f}, {m[k]['all']['ci95']['recall'][1]:.2f}]"
    print(f"| {os.path.basename(d['tdb'])} | {g('field')} | {m['field']['all']['exact_assemblage_accuracy']:.3f} | {g('boundary')} | {m['boundary']['all']['exact_assemblage_accuracy']:.3f} |")
