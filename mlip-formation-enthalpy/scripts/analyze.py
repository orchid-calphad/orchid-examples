"""Saturation benchmark of MLIP formation enthalpies against ORCHID experiments, from the cached predictions.

Input: inputs/pairs_identical_set.csv, one row per experimental record (ORCHID Dataset 4 default subset) that has a
default Materials Project pair, a default OQMD pair and a valid value from all eight MLIPs (1,138 records, 455
compounds): experimental formation enthalpy (exp), MP and OQMD values, and each MLIP's value (relaxed from the MP
structure, MP2020 correction added except for CHGNet), kJ/mol-atom.

Output (results/summary.json, and a printed leaderboard): for each source, the robust sigma of (computed - experiment)
with its 95 % system-cluster bootstrap interval (1,000 resamples of chemical systems), by chemistry and DFT match
stratum; sigma_MLIP - sigma_MP with a paired system-cluster bootstrap; the correlation of each MLIP's residuals with
MP's residuals; MLIP vs MP / OQMD (one pair per compound). Statistics and seeds are those of ORCHID's DFT calibration
(estimators.py), so the numbers equal the source benchmark's (checked against cached/CALIBRATION_MLIP_reference.json).

  python scripts/analyze.py [OUT_DIR]       # default results/
"""
import csv, hashlib, json, math, os, random, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import estimators as B

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["chgnet-0.3.0", "grace-2l-omat", "mace-mpa-0", "mace-omat-0", "mattersim-5m", "orb-v3-mpa", "orb-v3-omat", "sevennet-omni"]
LABEL = {"grace-2l-omat": "GRACE-2L-OMAT", "mace-mpa-0": "MACE-MPA-0", "mace-omat-0": "MACE-OMAT-0",
         "sevennet-omni": "SevenNet-Omni (mpa head)", "orb-v3-mpa": "ORB-v3 cons.-inf. MPA", "orb-v3-omat": "ORB-v3 cons.-inf. OMat",
         "mattersim-5m": "MatterSim v1.0.0-5M", "chgnet-0.3.0": "CHGNet 0.3.0", "mp": "DFT: Materials Project", "oqmd": "DFT: OQMD"}
CHEM = ("magnetic_3d", "metal", "pblock_anion")
MATCH = ("on_hull_single", "polymorph_ambiguous", "above_hull")


def block(ps, label):
    res = [p["res"] for p in ps]
    s = B.stats_of(res)
    s = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}
    s["mae"] = round(sum(abs(x) for x in res) / len(res), 3)
    if len({p["system"] for p in ps}) > 1:
        s["ci95_cluster_system"] = B.boot_ci_cluster(ps, label)
    return s


def corr(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y)); sxx = sum((a - mx) ** 2 for a in x); syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy)


def main(out):
    rows = list(csv.DictReader(open(os.path.join(ROOT, "inputs", "pairs_identical_set.csv"))))
    for r in rows:
        for k in ["exp", "mp", "oqmd"] + MODELS:
            r[k] = float(r[k])
    rows.sort(key=lambda r: r["rid"])
    same = {}
    for key in MODELS + ["mp", "oqmd"]:
        P = [{"res": round(r[key] - r["exp"], 3), "system": r["system"], "chem": r["chem"],
              "match": r["oqmd_match"] if key == "oqmd" else r["mp_match"]} for r in rows]
        st = {"all": block(P, "same|%s|all" % key), "strata": {}}
        for c in CHEM:
            q = [p for p in P if p["chem"] == c]
            st["strata"]["chem:" + c] = block(q, "same|%s|chem:%s" % (key, c)) if q else None
        for c in MATCH:
            q = [p for p in P if p["match"] == c]
            st["strata"]["match:" + c] = block(q, "same|%s|match:%s" % (key, c)) if q else None
        same[key] = st
    comps = {}
    for r in rows:
        comps.setdefault((r["formula"], r["mp_mid"]), r)
    vs_dft = {m: {s: block([{"res": round(c[m] - c[s], 3), "system": c["system"]} for c in comps.values()], "vsdft|%s|%s" % (m, s))
                  for s in ("mp", "oqmd")} for m in MODELS}
    bysys = defaultdict(list)
    for r in rows:
        bysys[r["system"]].append(r)
    systems = sorted(bysys)
    sat = {}
    for m in MODELS:
        rng = random.Random(int(hashlib.sha256(("sat|" + m).encode()).hexdigest()[:8], 16))
        d = []
        for _ in range(B.N_BOOT):
            rm, rd = [], []
            for _ in range(len(systems)):
                for r in bysys[systems[rng.randrange(len(systems))]]:
                    rm.append(r[m] - r["exp"]); rd.append(r["mp"] - r["exp"])
            d.append(B.stats_of(rm)["sigma_robust"] - B.stats_of(rd)["sigma_robust"])
        sat[m] = {"delta_sigma_vs_mp": round(same[m]["all"]["sigma_robust"] - same["mp"]["all"]["sigma_robust"], 3),
                  "delta_sigma_vs_mp_ci95_cluster_system": [round(B.quantile(d, 0.025), 3), round(B.quantile(d, 0.975), 3)],
                  "corr_residual_with_mp": round(corr([r[m] - r["exp"] for r in rows], [r["mp"] - r["exp"] for r in rows]), 3)}
    res = {"n_records": len(rows), "n_compounds": len(comps), "n_systems": len(systems), "units": "kJ/mol-atom",
           "residual": "computed - experiment", "same_set": same, "mlip_vs_dft": vs_dft, "saturation": sat}
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(f"identical set: {len(rows)} records, {len(comps)} compounds, {len(systems)} systems; kJ/mol-atom; 95% system-cluster CIs")
    print(f"{'source':28s} {'MAE':>5s} {'bias':>6s} {'robust sigma [95% CI]':>24s} {'d_sigma vs MP [95% CI]':>24s} {'r(res, MP res)':>14s}")
    for key in sorted(MODELS + ["mp", "oqmd"], key=lambda k: same[k]["all"]["sigma_robust"]):
        a = same[key]["all"]; ci = a["ci95_cluster_system"]["sigma_robust"]
        s = sat.get(key)
        ds = f"{s['delta_sigma_vs_mp']:+.2f} [{s['delta_sigma_vs_mp_ci95_cluster_system'][0]:+.2f}, {s['delta_sigma_vs_mp_ci95_cluster_system'][1]:+.2f}]" if s else "-"
        print(f"{LABEL[key]:28s} {a['mae']:5.1f} {a['bias_median']:+6.1f} {a['sigma_robust']:8.1f} [{ci[0]:4.1f}, {ci[1]:4.1f}]      {ds:>24s} {s['corr_residual_with_mp'] if s else '-':>14}")
    # check against the source benchmark's numbers
    ref = json.load(open(os.path.join(ROOT, "cached", "CALIBRATION_MLIP_reference.json")))
    worst = 0.0
    for key in same:
        r_, n_ = ref["same_set"]["stats"][key]["all"], same[key]["all"]
        worst = max(worst, abs(r_["sigma_robust"] - n_["sigma_robust"]), abs(r_["mae"] - n_["mae"]),
                    *[abs(a - b) for a, b in zip(r_["ci95_cluster_system"]["sigma_robust"], n_["ci95_cluster_system"]["sigma_robust"])])
        for st, v in same[key]["strata"].items():
            rv = ref["same_set"]["stats"][key]["strata"].get(st)
            if v and rv:
                worst = max(worst, abs(rv["sigma_robust"] - v["sigma_robust"]),
                            *[abs(a - b) for a, b in zip(rv["ci95_cluster_system"]["sigma_robust"], v["ci95_cluster_system"]["sigma_robust"])])
    for m in MODELS:
        worst = max(worst, abs(ref["saturation"][m]["corr_residual_with_mp"] - sat[m]["corr_residual_with_mp"]),
                    *[abs(a - b) for a, b in zip(ref["saturation"][m]["delta_sigma_vs_mp_ci95_cluster_system"], sat[m]["delta_sigma_vs_mp_ci95_cluster_system"])])
        for s in ("mp", "oqmd"):
            worst = max(worst, abs(ref["mlip_vs_dft"][m][s]["sigma_robust"] - vs_dft[m][s]["sigma_robust"]))
    print(f"max |difference| from the source benchmark (sigma, MAE, cluster CIs, strata, saturation, MLIP vs DFT): {worst:.3g} kJ/mol-atom")
    return worst


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results"))
