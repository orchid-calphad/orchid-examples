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

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estimators as est

root = Path(__file__).resolve().parents[1]
models = [
    "chgnet-0.3.0",
    "grace-2l-omat",
    "mace-mpa-0",
    "mace-omat-0",
    "mattersim-5m",
    "orb-v3-mpa",
    "orb-v3-omat",
    "sevennet-omni",
]
labels = {
    "grace-2l-omat": "GRACE-2L-OMAT",
    "mace-mpa-0": "MACE-MPA-0",
    "mace-omat-0": "MACE-OMAT-0",
    "sevennet-omni": "SevenNet-Omni (mpa head)",
    "orb-v3-mpa": "ORB-v3 cons.-inf. MPA",
    "orb-v3-omat": "ORB-v3 cons.-inf. OMat",
    "mattersim-5m": "MatterSim v1.0.0-5M",
    "chgnet-0.3.0": "CHGNet 0.3.0",
    "mp": "DFT: Materials Project",
    "oqmd": "DFT: OQMD",
}
_chem = ("magnetic_3d", "metal", "pblock_anion")
_match = ("on_hull_single", "polymorph_ambiguous", "above_hull")


def block(ps: list[dict], label: str) -> dict:
    """Residual statistics of one stratum, with a system-cluster interval when it spans several systems.

    Args:
        ps (list[dict]): Pairs, each with "res" (kJ/mol-atom) and "system" keys.
        label (str): Seed label for the bootstrap.

    Returns:
        Rounded statistics (kJ/mol-atom) with mae, and ci95_cluster_system when there is more than one system.
    """
    res = [p["res"] for p in ps]
    s = est.stats_of(res)
    s = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}
    s["mae"] = round(sum(abs(x) for x in res) / len(res), 3)
    if len({p["system"] for p in ps}) > 1:
        s["ci95_cluster_system"] = est.boot_ci_cluster(ps, label)
    return s


def corr(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient of two equally long samples.

    Args:
        x (list[float]): First sample.
        y (list[float]): Second sample, same length as x.

    Returns:
        The correlation coefficient, between -1 and 1.
    """
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy)


def main(out: str | Path) -> float:
    """Run the saturation benchmark, write summary.json and print the leaderboard.

    Args:
        out (str | Path): Output directory (created if missing).

    Returns:
        The largest absolute difference (kJ/mol-atom) from the source benchmark's reference numbers.
    """
    out_dir = Path(out)
    with (root / "inputs" / "pairs_identical_set.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ["exp", "mp", "oqmd", *models]:
            r[k] = float(r[k])
    rows.sort(key=lambda r: r["rid"])
    same = {}
    for key in [*models, "mp", "oqmd"]:
        pts = [
            {
                "res": round(r[key] - r["exp"], 3),
                "system": r["system"],
                "chem": r["chem"],
                "match": r["oqmd_match"] if key == "oqmd" else r["mp_match"],
            }
            for r in rows
        ]
        st = {"all": block(pts, f"same|{key}|all"), "strata": {}}
        for c in _chem:
            q = [p for p in pts if p["chem"] == c]
            st["strata"]["chem:" + c] = block(q, f"same|{key}|chem:{c}") if q else None
        for c in _match:
            q = [p for p in pts if p["match"] == c]
            st["strata"]["match:" + c] = block(q, f"same|{key}|match:{c}") if q else None
        same[key] = st
    comps = {}
    for r in rows:
        comps.setdefault((r["formula"], r["mp_mid"]), r)
    vs_dft = {
        m: {
            s: block([{"res": round(c[m] - c[s], 3), "system": c["system"]} for c in comps.values()], f"vsdft|{m}|{s}")
            for s in ("mp", "oqmd")
        }
        for m in models
    }
    bysys = defaultdict(list)
    for r in rows:
        bysys[r["system"]].append(r)
    systems = sorted(bysys)
    sat = {}
    for m in models:
        rng = random.Random(int(hashlib.sha256(("sat|" + m).encode()).hexdigest()[:8], 16))
        d = []
        for _ in range(est.n_boot):
            rm, rd = [], []
            for _ in range(len(systems)):
                for r in bysys[systems[rng.randrange(len(systems))]]:
                    rm.append(r[m] - r["exp"])
                    rd.append(r["mp"] - r["exp"])
            d.append(est.stats_of(rm)["sigma_robust"] - est.stats_of(rd)["sigma_robust"])
        sat[m] = {
            "delta_sigma_vs_mp": round(same[m]["all"]["sigma_robust"] - same["mp"]["all"]["sigma_robust"], 3),
            "delta_sigma_vs_mp_ci95_cluster_system": [round(est.quantile(d, 0.025), 3), round(est.quantile(d, 0.975), 3)],
            "corr_residual_with_mp": round(corr([r[m] - r["exp"] for r in rows], [r["mp"] - r["exp"] for r in rows]), 3),
        }
    res = {
        "n_records": len(rows),
        "n_compounds": len(comps),
        "n_systems": len(systems),
        "units": "kJ/mol-atom",
        "residual": "computed - experiment",
        "same_set": same,
        "mlip_vs_dft": vs_dft,
        "saturation": sat,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(res, f, indent=1)
    print(
        f"identical set: {len(rows)} records, {len(comps)} compounds, {len(systems)} systems; kJ/mol-atom; 95% system-cluster CIs"
    )
    print(
        f"{'source':28s} {'MAE':>5s} {'bias':>6s} {'robust sigma [95% CI]':>24s} {'d_sigma vs MP [95% CI]':>24s} {'r(res, MP res)':>14s}"
    )
    for key in sorted([*models, "mp", "oqmd"], key=lambda k: same[k]["all"]["sigma_robust"]):
        a = same[key]["all"]
        ci = a["ci95_cluster_system"]["sigma_robust"]
        s = sat.get(key)
        ds = (
            f"{s['delta_sigma_vs_mp']:+.2f} [{s['delta_sigma_vs_mp_ci95_cluster_system'][0]:+.2f}, {s['delta_sigma_vs_mp_ci95_cluster_system'][1]:+.2f}]"
            if s
            else "-"
        )
        print(
            f"{labels[key]:28s} {a['mae']:5.1f} {a['bias_median']:+6.1f} {a['sigma_robust']:8.1f} [{ci[0]:4.1f}, {ci[1]:4.1f}]      {ds:>24s} {s['corr_residual_with_mp'] if s else '-':>14}"
        )
    worst = _reference_difference(same, sat, vs_dft, models)
    print(
        f"max |difference| from the source benchmark (sigma, MAE, cluster CIs, strata, saturation, MLIP vs DFT): {worst:.3g} kJ/mol-atom"
    )
    return worst


def _reference_difference(same: dict, sat: dict, vs_dft: dict, models: list[str]) -> float:
    """Largest absolute difference from the source benchmark's reference numbers.

    Args:
        same (dict): Identical-set statistics computed here.
        sat (dict): Saturation statistics computed here.
        vs_dft (dict): MLIP-vs-DFT statistics computed here.
        models (list[str]): MLIP model keys.

    Returns:
        The largest difference in sigma, MAE, cluster CIs, strata, saturation and MLIP-vs-DFT sigma, in kJ/mol-atom.
    """
    with (root / "cached" / "CALIBRATION_MLIP_reference.json").open() as f:
        ref = json.load(f)
    worst = 0.0
    for key in same:
        r_, n_ = ref["same_set"]["stats"][key]["all"], same[key]["all"]
        worst = max(
            worst,
            abs(r_["sigma_robust"] - n_["sigma_robust"]),
            abs(r_["mae"] - n_["mae"]),
            *[
                abs(a - b)
                for a, b in zip(r_["ci95_cluster_system"]["sigma_robust"], n_["ci95_cluster_system"]["sigma_robust"], strict=True)
            ],
        )
        for st, v in same[key]["strata"].items():
            rv = ref["same_set"]["stats"][key]["strata"].get(st)
            if v and rv:
                worst = max(
                    worst,
                    abs(rv["sigma_robust"] - v["sigma_robust"]),
                    *[
                        abs(a - b)
                        for a, b in zip(
                            rv["ci95_cluster_system"]["sigma_robust"], v["ci95_cluster_system"]["sigma_robust"], strict=True
                        )
                    ],
                )
    for m in models:
        worst = max(
            worst,
            abs(ref["saturation"][m]["corr_residual_with_mp"] - sat[m]["corr_residual_with_mp"]),
            *[
                abs(a - b)
                for a, b in zip(
                    ref["saturation"][m]["delta_sigma_vs_mp_ci95_cluster_system"],
                    sat[m]["delta_sigma_vs_mp_ci95_cluster_system"],
                    strict=True,
                )
            ],
        )
        for s in ("mp", "oqmd"):
            worst = max(worst, abs(ref["mlip_vs_dft"][m][s]["sigma_robust"] - vs_dft[m][s]["sigma_robust"]))
    return worst


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else root / "results")
