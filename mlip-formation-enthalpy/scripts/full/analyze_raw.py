# Full-path analysis of the MLIP benchmark, copied from orchid-mlip-benchmark scripts/analyze.py; only the two repository
# paths changed (ORCHID_SITE_REPO, ORCHID_DATA_REPO). Run with --site-rev=181ebd6001a41f93de539a288fbf96cb1f5bdab2
# --data-rev=1e5f4ce7d189da3395798135f7726f79af67703f to reproduce the cached pairs.
#!/usr/bin/env python3
"""MLIP formation enthalpies vs experiment (and vs MP / OQMD) with ORCHID's DFT-benchmark rules.

Mirrors ORCHID Dataset 4 (DFT vs experiment) exactly by IMPORTING its code, not reimplementing it:
  orchid-project.org/tools/build_espei_bundle.py : stats_of (bias, robust sigma = 1.4826 MAD, q68/q95, RMSE),
      boot_ci (1000 percentile bootstrap resamples, label-seeded), calibrate (strata, min_stratum 30 fallback,
      leave-system-out sigma), stratum_key, chem_class (metal / pblock_anion / magnetic_3d), median
  orchid-project.org/data/literature/dft_parity.json : records, default subset, matched MP/OQMD entries
      (lowest-energy entry per composition, npoly, ehull)

Pairs: record by record, exactly as calibration_pairs(): one pair per (default-subset experimental record x
matched entry). MLIP value of a record = MLIP formation energy of the MP entry matched to it (DFT-relaxed MP
structure, relaxed again with the MLIP; elemental references = MP elemental ground-state structures relaxed with
the same MLIP). Match-quality stratum = that of the MP entry (on_hull_single / polymorph_ambiguous / above_hull,
hull tolerance parameterised). Per-compound-median pairing (literature-DFT style) is reported as well.

  python scripts/analyze.py --results results/pilot --out results/pilot_analysis [--hull-tol-mev auto|0|1]
"""
import argparse, glob, hashlib, json, math, os, subprocess, sys
from collections import Counter, defaultdict

sys.dont_write_bytecode = True
SITE_REPO = os.path.expanduser(os.environ.get("ORCHID_SITE_REPO", "orchid-project.org"))   # git checkout of orchid-calphad/orchid-project.org
DATA = os.path.expanduser(os.environ.get("ORCHID_DATA_REPO", "orchid-data"))               # git checkout of orchid-calphad/orchid-data
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # example root


def _git_sha(repo, rev):
    return subprocess.run(["git", "-C", repo, "rev-parse", rev], capture_output=True, text=True, check=True).stdout.strip()


def vendor_site(site_rev, data_rev):
    """Committed state only (other agents' uncommitted edits are never used): orchid-project.org tools/ at site_rev
    (the DFT calibration code) + orchid-data literature/{dft_parity.json, orchid-literature.jsonl.gz, papers.json}
    at data_rev (the released experimental reference data and DFT pairs), laid out as <dir>/tools, <dir>/data/literature."""
    s1, s2 = _git_sha(SITE_REPO, site_rev), _git_sha(DATA, data_rev)
    d = os.path.join(ROOT, ".cache", "orchid-%s-%s" % (s1[:10], s2[:10]))
    if not os.path.exists(os.path.join(d, "ok")):
        os.makedirs(os.path.join(d, "data"), exist_ok=True)
        ar = subprocess.run(["git", "-C", SITE_REPO, "archive", s1, "tools"], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", d], input=ar, check=True)
        ar = subprocess.run(["git", "-C", DATA, "archive", s2, "literature/dft_parity.json",
                             "literature/orchid-literature.jsonl.gz", "literature/papers.json", "literature/index.json"], capture_output=True,
                            check=True).stdout
        subprocess.run(["tar", "-x", "-C", os.path.join(d, "data")], input=ar, check=True)
        open(os.path.join(d, "ok"), "w").write(s1 + " " + s2)
    return d, s1, s2


def _opt(name, default):
    return next((x.split("=", 1)[1] for x in sys.argv if x.startswith(name + "=")), default)


SITE, SITE_SHA, DATA_SHA = vendor_site(_opt("--site-rev", "HEAD"), _opt("--data-rev", "HEAD"))
sys.path.insert(0, os.path.join(SITE, "tools"))
import build_espei_bundle as B  # noqa: E402
KJ = 96.485  # kJ/mol per eV/atom, as build_dft_parity.KJ_PER_EV
CUZR_INDIRECT = 9.7  # kJ/mol-atom, sigma_DFT (+) MLIP->DFT, Cu-Zr (GRACE-2L-OMAT)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def git(repo, *a):
    try:
        return subprocess.run(["git", "-C", repo] + list(a), capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def committed_hull_tol():
    txt = git(DATA, "show", "HEAD:espei/dft/CALIBRATION.json")
    try:
        m = json.loads(txt)["method"]
        return float(m.get("hull_tolerance_meV_per_atom", 0.0))
    except Exception:
        return 0.0


def default(r):  # build_dft_parity / calibration_pairs default subset
    return (r["ev"] == "experimental" and r["acc"] and not r["unsigned"] and not r["sign_lost"]
            and "magnitude_suspect" not in r["suspect"])


def load_results(dirs):
    E = defaultdict(dict)  # model -> id -> rec
    meta = {}
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            for line in open(f):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("meta"):
                    meta.setdefault(r["model"], r)
                    continue
                E[r["model"]][r["id"]] = r
    return E, meta


MODELS_SPEC = json.load(open(os.path.join(ROOT, "scripts/full/models.json")))


MOLECULAR = {"H", "N", "O", "F", "Cl", "Br", "I"}  # molecular-crystal elements: no-vdW MLIPs expand them freely


def ref_bad(q, e):
    """Elemental reference relaxation unusable: not converged, or exploded. For molecular-crystal elements the
    volume criterion is dropped (cell expansion without dispersion is the model's physics); energy criteria stay."""
    if q.get("status") != "ok" or not q.get("converged"):
        return True
    if e and e.get("element") in MOLECULAR:
        e1 = q["e_relax_eV_per_atom"]
        return not (-15 < e1 < 5) or e1 < q["e_sp_eV_per_atom"] - 1.0
    return bool(q.get("exploded"))


def mlip_formation(E, items):
    """{model: {compound id: {...variants...}}}; eV/atom. corr = raw + MP correction, unless the model's training
    targets already include the MP2020 corrections (models.json predicts_mp2020_corrected), then corr = raw."""
    comp = {i["id"]: i for i in items if i["kind"] == "compound"}
    eref = {(i["src"], i["element"]): i for i in items if i["kind"] == "element"}
    out = {}
    for model, R in E.items():
        res = {}
        pre = bool((MODELS_SPEC.get(model) or {}).get("predicts_mp2020_corrected"))
        for cid, c in comp.items():
            r = R.get(cid)
            if not r:
                continue
            row = {"status": r["status"], "converged": r.get("converged"), "exploded": r.get("exploded"),
                   "nsteps": r.get("nsteps"), "vol_ratio": r.get("vol_ratio"), "sg_in": r.get("sg_in"),
                   "sg_out": r.get("sg_out"), "t_s": (r.get("t_relax_s") or 0) + (r.get("t_sp_s") or 0)}
            refs = [(eref.get((c["src"], el)), x) for el, x in c["comp"].items()]
            rr = [(R.get(e["id"]) if e else None, x, e) for e, x in refs]
            ok = r["status"] == "ok" and all(q and q["status"] == "ok" for q, _, _ in rr)
            row["refs_ok"] = all(q and q["status"] == "ok" for q, _, _ in rr)
            row["refs_flag"] = any(q and ref_bad(q, e) for q, _, e in rr)
            if ok:
                corr = c.get("mp_correction_eV_per_atom", 0.0) - sum(x * e.get("mp_correction_eV_per_atom", 0.0)
                                                                     for _, x, e in rr)
                for var, key in (("relax", "e_relax_eV_per_atom"), ("sp", "e_sp_eV_per_atom")):
                    raw = r[key] - sum(x * q[key] for q, x, _ in rr)
                    row["ef_%s_raw" % var] = raw
                    row["ef_%s_corr" % var] = raw + (0.0 if pre else corr)
                row["mp_corr"] = corr
            res[cid] = row
        out[model] = res
    return out


def stats_block(res, label):
    s = B.stats_of(res)
    if s is None:
        return None
    s = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}
    s["mae"] = round(sum(abs(x) for x in res) / len(res), 3)
    s["ci95"] = B.boot_ci(res, label)
    s["rmse_over_sigma_robust"] = round(s["rmse"] / s["sigma_robust"], 2) if s["sigma_robust"] else None
    return s


def with_cluster(ps, label):
    s = stats_block([p["res"] for p in ps], label)
    if s and len({p["system"] for p in ps}) > 1:
        s["ci95_cluster_system"] = B.boot_ci_cluster(ps, label)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--inputs", nargs="+", default=[os.path.join(ROOT, "inputs/structures.json")])
    ap.add_argument("--out", required=True)
    ap.add_argument("--site-rev", default="HEAD", help="orchid-project.org revision of the calibration code (use --site-rev=X)")
    ap.add_argument("--data-rev", default="HEAD", help="orchid-data revision of the reference data (use --data-rev=X)")
    ap.add_argument("--variant", default="relax_corr", help="primary MLIP value: relax|sp x raw|corr")
    a, _ = ap.parse_known_args()
    os.makedirs(a.out, exist_ok=True)
    dpp = os.path.join(SITE, "data/literature/dft_parity.json")
    pub = B.published(os.path.join(SITE, "data", "literature"))[0]
    # DFT pairs: exactly the DFT calibration's (default subset per point, strata, systems, sigma_exp)
    pairs_dft, dp = B.calibration_pairs(os.path.join(SITE, "data"), pub)
    R = dp["records"]
    items = [it for f in a.inputs for it in json.load(open(f))["items"]]
    itm = {i["id"]: i for i in items}
    E, meta = load_results(a.results)
    F = mlip_formation(E, items)
    models = sorted(F)

    # the matched DFT entry of every default point (calibration_pairs keeps rid, not mid)
    def point_default(p):
        return p["default"] if "default" in p else default(R[p["rid"]])
    pt = {(p["src"], p["rid"]): p for p in dp["points"] if point_default(p)}

    def mval(m, rid, var):
        p = pt.get(("mp", rid))
        f = F[m].get(p["mid"]) if p else None
        if not f or ("ef_" + var) not in f:
            return None, "no MLIP value (MP entry %s not in the inputs)" % (p and p["mid"]) if p and p["mid"] not in itm else "relaxation failed"
        if f.get("exploded") or not f.get("converged") or f.get("refs_flag"):
            return None, "flagged (not converged / exploded / reference)"
        return round(f["ef_" + var] * KJ, 3), None

    pairs, dropped = {"mp": pairs_dft["mp"], "oqmd": pairs_dft["oqmd"]}, {}
    for m in models:
        pairs[m], dropped[m] = [], Counter()
        for q in pairs_dft["mp"]:
            v, why = mval(m, q["rid"], a.variant)
            if v is None:
                dropped[m][why] += 1
                continue
            pairs[m].append(dict(q, res=round(v - R[q["rid"]]["lit"], 3), value=v))
    cal = B.calibrate(pairs, {(s_, B.stratum_key(p), p["system"]) for s_, P in pairs.items() for p in P})

    # identical records: default MP pair, default OQMD pair and a valid value from every model
    rid_sets = [{p["rid"] for p in pairs[k]} for k in pairs]
    same_rids = set.intersection(*rid_sets)
    same = {}
    for key in models + ["mp", "oqmd"]:
        P = [p for p in pairs[key] if p["rid"] in same_rids]
        st = {"all": with_cluster(P, "same|%s|all" % key), "strata": {}}
        for k in sorted({B.stratum_key(p) for p in P}):
            st["strata"][k] = with_cluster([p for p in P if B.stratum_key(p) == k], "same|%s|%s" % (key, k))
        for c in sorted(B.CHEM_TEXT):
            q = [p for p in P if p["chem"] == c]
            st["strata"]["chem:" + c] = with_cluster(q, "same|%s|chem:%s" % (key, c)) if q else None
        for c in ("on_hull_single", "polymorph_ambiguous", "above_hull"):
            q = [p for p in P if p["match"] == c]
            st["strata"]["match:" + c] = with_cluster(q, "same|%s|match:%s" % (key, c)) if q else None
        systems = sorted({p["system"] for p in P})
        lso = [B.stats_of([p["res"] for p in P if p["system"] != s_])["sigma_robust"] for s_ in systems]
        st["lso_sigma_robust"] = {"median": round(B.median(lso), 3), "min": round(min(lso), 3), "max": round(max(lso), 3),
                                  "n_systems": len(systems)} if len(lso) > 1 else None
        same[key] = st

    # rows per identical record (values), for MLIP-vs-DFT, per-compound medians, variants and the CSV
    rows = []
    for rid in sorted(same_rids):
        r = R[rid]
        row = {"rid": rid, "formula": r["formula"], "system": "-".join(sorted(e.upper() for e in r["comp"])),
               "chem": B.chem_class([e.upper() for e in r["comp"]]), "exp": r["lit"], "paper": r["paper"],
               "method": r.get("method")}
        for s_ in ("mp", "oqmd"):
            p = pt[(s_, rid)]
            row.update({s_: p["dft"], s_ + "_mid": p["mid"], s_ + "_smatch": p.get("smatch"), s_ + "_ehull": p.get("ehull"),
                        s_ + "_npoly": p["npoly"]})
        row["mp_match"] = next(q["match"] for q in pairs["mp"] if q["rid"] == rid)
        row["oqmd_match"] = next(q["match"] for q in pairs["oqmd"] if q["rid"] == rid)
        for m in models:
            row[m] = mval(m, rid, a.variant)[0]
            for v in ("relax_raw", "relax_corr", "sp_raw", "sp_corr"):
                row["%s|%s" % (m, v)] = mval(m, rid, v)[0]
        rows.append(row)
    comps = {}
    for r in rows:
        comps.setdefault((r["formula"], r["mp_mid"]), r)
    vs_dft = {}
    for m in models:
        vs_dft[m] = {s_: with_cluster([{"res": round(c[m] - c[s_], 3), "system": c["system"]} for c in comps.values()],
                                      "vsdft|%s|%s" % (m, s_)) for s_ in ("mp", "oqmd")}
    vs_dft["oqmd_vs_mp"] = {"mp": with_cluster([{"res": round(c["oqmd"] - c["mp"], 3), "system": c["system"]}
                                                for c in comps.values()], "vsdft|oqmd|mp")}
    g = defaultdict(list)
    for r in rows:
        g[(r["formula"], r["mp_mid"])].append(r)
    med = {key: with_cluster([{"res": round(v[0][key] - B.median([x["exp"] for x in v]), 3), "system": v[0]["system"]}
                              for v in g.values()], "median|%s" % key) for key in models + ["mp", "oqmd"]}
    variants = {m: {v: with_cluster([{"res": round(r["%s|%s" % (m, v)] - r["exp"], 3), "system": r["system"]}
                                     for r in rows if r.get("%s|%s" % (m, v)) is not None], "var|%s|%s" % (m, v))
                    for v in ("relax_corr", "relax_raw", "sp_corr", "sp_raw")} for m in models}
    # saturation: per-record difference |MLIP-exp| vs |MP-exp| (paired), and MLIP sigma minus MP sigma (paired bootstrap)
    sat = {}
    for m in models:
        P = [r for r in rows]
        d_sig = []
        import random as _r
        bysys = defaultdict(list)
        for r in P:
            bysys[r["system"]].append(r)
        systems = sorted(bysys)
        rng = _r.Random(int(hashlib.sha256(("sat|" + m).encode()).hexdigest()[:8], 16))
        for _ in range(B.N_BOOT):
            res_m, res_d = [], []
            for _ in range(len(systems)):
                for r in bysys[systems[rng.randrange(len(systems))]]:
                    res_m.append(r[m] - r["exp"]); res_d.append(r["mp"] - r["exp"])
            d_sig.append(B.stats_of(res_m)["sigma_robust"] - B.stats_of(res_d)["sigma_robust"])
        sat[m] = {"delta_sigma_vs_mp": round(same[m]["all"]["sigma_robust"] - same["mp"]["all"]["sigma_robust"], 3),
                  "delta_sigma_vs_mp_ci95_cluster_system": [round(B.quantile(d_sig, 0.025), 3), round(B.quantile(d_sig, 0.975), 3)],
                  "corr_residual_with_mp": round(_corr([r[m] - r["exp"] for r in P], [r["mp"] - r["exp"] for r in P]), 3)}
    health = {}
    for m in models:
        f = list(F[m].values())
        el = [E[m][i["id"]] for i in items if i["kind"] == "element" and i["id"] in E[m]]
        health[m] = {"compounds_run": len(f), "errors": sum(1 for x in f if x["status"] != "ok"),
                     "not_converged": sum(1 for x in f if x["status"] == "ok" and not x["converged"]),
                     "exploded": sum(1 for x in f if x.get("exploded")),
                     "symmetry_changed": sum(1 for x in f if x["status"] == "ok" and x["sg_in"] != x["sg_out"]),
                     "elements_run": len(el), "elements_not_ok": sum(1 for x in el if ref_bad(x, itm.get(x["id"]))),
                     "elements_flagged_volume_only": sum(1 for x in el if x.get("exploded") and not ref_bad(x, itm.get(x["id"]))),
                     "median_time_s": round(B.median([x["t_s"] for x in f if x["status"] == "ok"]) or 0, 3),
                     "total_time_s": round(sum(x["t_s"] for x in f) + sum((x.get("t_relax_s") or 0) + (x.get("t_sp_s") or 0) for x in el), 1),
                     "default_pairs_dropped": dict(dropped[m])}
    tol_mev = B.HULL_TOL_MEV if hasattr(B, "HULL_TOL_MEV") else 0.0
    out = {
        "schema": "orchid-mlip-calibration-v2 (layout of orchid-data espei/dft/CALIBRATION.json: method, pairs_release, "
                  "sources{source: {all, strata}}, lso; plus same_set, mlip_vs_dft, per_compound_median, variants, "
                  "saturation, health, models)",
        "method": {"pairs": "build_espei_bundle.calibration_pairs (orchid-project.org %s): DFT pairs exactly as the DFT "
                            "calibration; an MLIP pair = the MP pair of the same record with the MLIP formation enthalpy "
                            "of the same matched MP entry (structure-matched polymorph when the record names one)" % SITE_SHA[:10],
                   "residual": "MLIP - experiment, kJ/mol-atom (rounded to 1e-3 as dft_parity res)",
                   "sigma_robust": "1.4826 * MAD", "bootstrap": B.N_BOOT, "ci": "percentile 95 %",
                   "ci_cluster": "ci95_cluster_system: resample chemical systems with all their pairs (quote this one)",
                   "min_stratum": B.MIN_STRATUM, "leave_system_out": True, "bias_applied": False,
                   "hull_tolerance_meV_per_atom": tol_mev, "hull_tolerance_kJ_per_mol_atom": round(B.HULL_TOL_KJ, 4),
                   "primary_variant": a.variant,
                   "variants": {"relax": "MLIP relaxation (FIRE + FrechetCellFilter, fmax 0.02 eV/A, <= 1000 steps) of the "
                                         "DFT-relaxed MP structure; elemental references = MP ground states relaxed alike",
                                "sp": "MLIP single point on the DFT-relaxed structure",
                                "corr": "plus the MP energy correction of the matched entries (compound minus composition-"
                                        "weighted elemental corrections), unless the model predicts corrected energies",
                                "raw": "model output without added correction"},
                   "excluded": "pairs whose MLIP relaxation (compound or elemental reference) did not converge, errored or "
                               "exploded; pairs whose matched MP entry is not among the computed structures",
                   "chemistry_classes": B.CHEM_TEXT, "match_quality": B.MATCH_TEXT, "eV_to_kJ_per_mol": KJ,
                   "cuzr_indirect_sigma_kJ": CUZR_INDIRECT},
        "pairs_release": "dft_parity.json sha256 %s (literature sha256 %s); inputs sha256 %s" % (
            sha(dpp), dp["inputs"]["literature"]["sha256"], ", ".join("%s %s" % (os.path.basename(f), sha(f)) for f in a.inputs)),
        "code": {"orchid-project.org": {"commit": SITE_SHA, "build_espei_bundle.py sha256": sha(B.__file__)},
                 "orchid_mlip_benchmark": git(ROOT, "rev-parse", "HEAD"),
                 "orchid-data (reference data)": DATA_SHA},
        "sources": cal["sources"], "lso": cal["lso"],
        "same_set": {"n_records": len(rows), "n_compounds": len(comps), "stats": same},
        "mlip_vs_dft": vs_dft, "per_compound_median": med, "variants": variants, "saturation": sat, "health": health,
        "models": {m: {"spec": (meta.get(m) or {}).get("spec"), "env": (meta.get(m) or {}).get("env"),
                       "fmax": (meta.get(m) or {}).get("fmax"), "steps": (meta.get(m) or {}).get("steps")} for m in models},
    }
    out = B.round_floats(out)
    json.dump(out, open(os.path.join(a.out, "CALIBRATION_MLIP.json"), "w"), indent=1, sort_keys=True)
    import csv
    cols = ["rid", "formula", "system", "chem", "exp", "mp", "oqmd", "mp_mid", "oqmd_mid", "mp_smatch", "mp_match",
            "oqmd_match", "mp_ehull", "mp_npoly", "paper", "method"]
    mcols = []
    for m in models:
        mcols += [m] + ["%s|%s" % (m, v) for v in ("relax_raw", "relax_corr", "sp_raw", "sp_corr")]
    with open(os.path.join(a.out, "pairs_identical_set.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols + mcols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(a.out, "mlip_by_compound.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "id", "src", "formula", "dft_ef_kJ", "ef_relax_corr_kJ", "ef_relax_raw_kJ", "ef_sp_corr_kJ",
                    "ef_sp_raw_kJ", "status", "converged", "exploded", "refs_flag", "nsteps", "vol_ratio", "sg_in",
                    "sg_out", "time_s"])
        for m in models:
            for cid, x in sorted(F[m].items()):
                it = itm[cid]
                g_ = lambda k: round(x[k] * KJ, 3) if k in x else None
                w.writerow([m, cid, it["src"], it["formula"], round(it["dft_ef_eV"] * KJ, 3), g_("ef_relax_corr"),
                            g_("ef_relax_raw"), g_("ef_sp_corr"), g_("ef_sp_raw"), x["status"], x["converged"],
                            x["exploded"], x["refs_flag"], x["nsteps"], x["vol_ratio"], x["sg_in"], x["sg_out"],
                            round(x["t_s"], 3)])
    print(json.dumps({"site": SITE_SHA[:10], "data": DATA_SHA[:10], "tol_meV": tol_mev, "dft_pairs": {k: len(v) for k, v in pairs_dft.items()},
                      "mlip_pairs": {m: len(pairs[m]) for m in models}, "identical_records": len(rows),
                      "identical_compounds": len(comps)}, indent=1))
    for k in ["mp", "oqmd"] + models:
        s_ = same[k]["all"]
        print("%-14s n=%4d MAE=%5.2f bias=%+5.2f sigma=%5.2f cluster%s RMSE=%5.2f" % (
            k, s_["n"], s_["mae"], s_["bias_median"], s_["sigma_robust"], s_["ci95_cluster_system"]["sigma_robust"], s_["rmse"]))


def _corr(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y)); sxx = sum((a - mx) ** 2 for a in x); syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy)


if __name__ == "__main__":
    main()
