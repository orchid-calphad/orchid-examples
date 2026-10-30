#!/usr/bin/env python3
"""MLIP formation enthalpies vs experiment (and vs MP / OQMD) with ORCHID's DFT-benchmark rules.

Full-path analysis of the MLIP benchmark, copied from orchid-mlip-benchmark scripts/analyze.py; only the two repository
paths changed (ORCHID_SITE_REPO, ORCHID_DATA_REPO). Run with --site-rev=181ebd6001a41f93de539a288fbf96cb1f5bdab2
--data-rev=1e5f4ce7d189da3395798135f7726f79af67703f to reproduce the cached pairs.

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

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlip_results import kj_per_ev, load_results, mlip_formation, ref_bad, to_kj
from orchid_vendor import bundle, data_sha, git, root, sha, site, site_sha

_cuzr_indirect = 9.7  # kJ/mol-atom, sigma_DFT (+) MLIP->DFT, Cu-Zr (GRACE-2L-OMAT)


def default(r: dict) -> bool:
    """Whether a record is in the default calibration subset (build_dft_parity / calibration_pairs).

    Args:
        r (dict): Record of dft_parity.json.

    Returns:
        True for an accepted experimental record that is unsigned-free, not sign-lost and not magnitude-suspect.
    """
    return (
        r["ev"] == "experimental"
        and r["acc"]
        and not r["unsigned"]
        and not r["sign_lost"]
        and "magnitude_suspect" not in r["suspect"]
    )


def stats_block(res: list[float], label: str) -> dict | None:
    """Residual statistics with a percentile bootstrap interval.

    Args:
        res (list[float]): Residuals (computed - experiment), kJ/mol-atom.
        label (str): Seed label for the bootstrap.

    Returns:
        Rounded statistics with mae, ci95 and rmse_over_sigma_robust; None for an empty sample.
    """
    s = bundle.stats_of(res)
    if s is None:
        return None
    s = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}
    s["mae"] = round(sum(abs(x) for x in res) / len(res), 3)
    s["ci95"] = bundle.boot_ci(res, label)
    s["rmse_over_sigma_robust"] = round(s["rmse"] / s["sigma_robust"], 2) if s["sigma_robust"] else None
    return s


def with_cluster(ps: list[dict], label: str) -> dict | None:
    """Statistics of pairs, plus the system-cluster bootstrap when there is more than one system.

    Args:
        ps (list[dict]): Pairs, each with "res" (kJ/mol-atom) and "system" keys.
        label (str): Seed label for the bootstrap.

    Returns:
        Statistics as stats_block returns them, with ci95_cluster_system added when applicable; None if empty.
    """
    s = stats_block([p["res"] for p in ps], label)
    if s and len({p["system"] for p in ps}) > 1:
        s["ci95_cluster_system"] = bundle.boot_ci_cluster(ps, label)
    return s


def _corr(x: list[float], y: list[float]) -> float:
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


class _MlipLookup:
    """MLIP formation enthalpies of the MP entries matched to the default DFT points.

    Args:
        formation (dict): Model -> compound id -> row, as returned by mlip_formation.
        pt (dict): Default DFT points keyed by (source, record id).
        itm (dict): Input items keyed by id.
    """

    def __init__(self, formation: dict, pt: dict, itm: dict) -> None:
        """Store the lookup tables.

        Args:
            formation (dict): Model -> compound id -> row, as returned by mlip_formation.
            pt (dict): Default DFT points keyed by (source, record id).
            itm (dict): Input items keyed by id.
        """
        self._formation = formation
        self._pt = pt
        self._itm = itm

    def point(self, src: str, rid: str) -> dict:
        """Default DFT point of a record from one source.

        Args:
            src (str): DFT source, mp or oqmd.
            rid (str): Experimental record id.

        Returns:
            The DFT point.
        """
        return self._pt[(src, rid)]

    def value(self, m: str, rid: str, var: str) -> tuple[float | None, str | None]:
        """MLIP formation enthalpy of the MP entry matched to a record, in kJ/mol-atom.

        Args:
            m (str): MLIP model key.
            rid (str): Experimental record id.
            var (str): Variant, relax|sp x raw|corr, for example relax_corr.

        Returns:
            Tuple (value in kJ/mol-atom rounded to 1e-3, None), or (None, reason) when the value is not usable.
        """
        p = self._pt.get(("mp", rid))
        f = self._formation[m].get(p["mid"]) if p else None
        if not f or ("ef_" + var) not in f:
            if p and p["mid"] not in self._itm:
                return None, f"no MLIP value (MP entry {p['mid']} not in the inputs)"
            return None, "relaxation failed"
        if f.get("exploded") or not f.get("converged") or f.get("refs_flag"):
            return None, "flagged (not converged / exploded / reference)"
        return round(f["ef_" + var] * kj_per_ev, 3), None


def _default_points(dp: dict, records: dict) -> dict:
    """Default-subset DFT points keyed by (source, record id).

    Args:
        dp (dict): Content of dft_parity.json, with "points".
        records (dict): Records keyed by record id, used for points without a "default" flag.

    Returns:
        Dict (source, rid) -> point, for the points in the default subset.
    """

    def point_default(p: dict) -> bool:
        """Whether one DFT point is in the default subset.

        Args:
            p (dict): DFT point of dft_parity.json.

        Returns:
            True if the point is in the default subset.
        """
        return p["default"] if "default" in p else default(records[p["rid"]])

    return {(p["src"], p["rid"]): p for p in dp["points"] if point_default(p)}


def _mlip_pairs(lookup: _MlipLookup, pairs_dft: dict, records: dict, models: list[str], *, variant: str) -> tuple[dict, dict]:
    """MLIP residual pairs for every model, one per DFT calibration pair, and the counts of dropped pairs.

    Args:
        lookup (_MlipLookup): MLIP values of the matched MP entries.
        pairs_dft (dict): DFT pairs of the calibration, keyed by "mp" and "oqmd".
        records (dict): Records keyed by record id (lit = literature value, kJ/mol-atom).
        models (list[str]): MLIP model keys.
        variant (str): Primary variant, for example relax_corr.

    Returns:
        Tuple (pairs, dropped): pairs maps model (and mp, oqmd) to pair lists; dropped maps model to a Counter of
        reasons.
    """
    pairs = {"mp": pairs_dft["mp"], "oqmd": pairs_dft["oqmd"]}
    dropped = {}
    for m in models:
        pairs[m] = []
        dropped[m] = Counter()
        for q in pairs_dft["mp"]:
            v, why = lookup.value(m, q["rid"], variant)
            if v is None:
                dropped[m][why] += 1
                continue
            pairs[m].append(dict(q, res=round(v - records[q["rid"]]["lit"], 3), value=v))
    return pairs, dropped


def _identical_set(pairs: dict, models: list[str]) -> tuple[dict, set]:
    """Statistics on the identical records: default MP pair, default OQMD pair and a valid value from every model.

    Args:
        pairs (dict): Pair lists per model and per DFT source, from _mlip_pairs.
        models (list[str]): MLIP model keys.

    Returns:
        Tuple (same, same_rids): same maps each source to its overall, stratum and leave-system-out statistics;
        same_rids is the set of record ids in the identical set.
    """
    rid_sets = [{p["rid"] for p in pairs[k]} for k in pairs]
    same_rids = set.intersection(*rid_sets)
    same = {}
    for key in [*models, "mp", "oqmd"]:
        pts = [p for p in pairs[key] if p["rid"] in same_rids]
        st = {"all": with_cluster(pts, f"same|{key}|all"), "strata": {}}
        for k in sorted({bundle.stratum_key(p) for p in pts}):
            st["strata"][k] = with_cluster([p for p in pts if bundle.stratum_key(p) == k], f"same|{key}|{k}")
        for c in sorted(bundle.CHEM_TEXT):
            q = [p for p in pts if p["chem"] == c]
            st["strata"]["chem:" + c] = with_cluster(q, f"same|{key}|chem:{c}") if q else None
        for c in ("on_hull_single", "polymorph_ambiguous", "above_hull"):
            q = [p for p in pts if p["match"] == c]
            st["strata"]["match:" + c] = with_cluster(q, f"same|{key}|match:{c}") if q else None
        systems = sorted({p["system"] for p in pts})
        lso = [bundle.stats_of([p["res"] for p in pts if p["system"] != s_])["sigma_robust"] for s_ in systems]
        st["lso_sigma_robust"] = (
            {
                "median": round(bundle.median(lso), 3),
                "min": round(min(lso), 3),
                "max": round(max(lso), 3),
                "n_systems": len(systems),
            }
            if len(lso) > 1
            else None
        )
        same[key] = st
    return same, same_rids


def _rows(records: dict, lookup: _MlipLookup, pairs: dict, models: list[str], same_rids: set, *, variant: str) -> list[dict]:
    """One row per identical record: experimental value, DFT values and every model's value.

    Args:
        records (dict): Records keyed by record id.
        lookup (_MlipLookup): MLIP values of the matched MP entries.
        pairs (dict): Pair lists per model and per DFT source.
        models (list[str]): MLIP model keys.
        same_rids (set): Record ids of the identical set.
        variant (str): Primary variant, for example relax_corr.

    Returns:
        List of row dicts, sorted by record id.
    """
    rows = []
    for rid in sorted(same_rids):
        r = records[rid]
        row = {
            "rid": rid,
            "formula": r["formula"],
            "system": "-".join(sorted(e.upper() for e in r["comp"])),
            "chem": bundle.chem_class([e.upper() for e in r["comp"]]),
            "exp": r["lit"],
            "paper": r["paper"],
            "method": r.get("method"),
        }
        for s_ in ("mp", "oqmd"):
            p = lookup.point(s_, rid)
            row.update(
                {
                    s_: p["dft"],
                    s_ + "_mid": p["mid"],
                    s_ + "_smatch": p.get("smatch"),
                    s_ + "_ehull": p.get("ehull"),
                    s_ + "_npoly": p["npoly"],
                }
            )
        row["mp_match"] = next(q["match"] for q in pairs["mp"] if q["rid"] == rid)
        row["oqmd_match"] = next(q["match"] for q in pairs["oqmd"] if q["rid"] == rid)
        for m in models:
            row[m] = lookup.value(m, rid, variant)[0]
            for v in ("relax_raw", "relax_corr", "sp_raw", "sp_corr"):
                row[f"{m}|{v}"] = lookup.value(m, rid, v)[0]
        rows.append(row)
    return rows


def _compound_tables(rows: list[dict], models: list[str]) -> tuple[dict, dict, dict, dict]:
    """Per-compound tables: MLIP vs DFT, per-compound medians and the variants.

    Args:
        rows (list[dict]): Identical-set rows, from _rows.
        models (list[str]): MLIP model keys.

    Returns:
        Tuple (comps, vs_dft, med, variants): comps maps (formula, MP id) to one row per compound; the other three are
        the statistics tables of summary.json.
    """
    comps = {}
    for r in rows:
        comps.setdefault((r["formula"], r["mp_mid"]), r)
    vs_dft = {}
    for m in models:
        vs_dft[m] = {
            s_: with_cluster([{"res": round(c[m] - c[s_], 3), "system": c["system"]} for c in comps.values()], f"vsdft|{m}|{s_}")
            for s_ in ("mp", "oqmd")
        }
    vs_dft["oqmd_vs_mp"] = {
        "mp": with_cluster(
            [{"res": round(c["oqmd"] - c["mp"], 3), "system": c["system"]} for c in comps.values()], "vsdft|oqmd|mp"
        )
    }
    g = defaultdict(list)
    for r in rows:
        g[(r["formula"], r["mp_mid"])].append(r)
    med = {
        key: with_cluster(
            [{"res": round(v[0][key] - bundle.median([x["exp"] for x in v]), 3), "system": v[0]["system"]} for v in g.values()],
            f"median|{key}",
        )
        for key in [*models, "mp", "oqmd"]
    }
    variants = {
        m: {
            v: with_cluster(
                [
                    {"res": round(r[f"{m}|{v}"] - r["exp"], 3), "system": r["system"]}
                    for r in rows
                    if r.get(f"{m}|{v}") is not None
                ],
                f"var|{m}|{v}",
            )
            for v in ("relax_corr", "relax_raw", "sp_corr", "sp_raw")
        }
        for m in models
    }
    return comps, vs_dft, med, variants


def _saturation(rows: list[dict], models: list[str], same: dict) -> dict:
    """Saturation: sigma difference to MP with a paired system-cluster bootstrap, and residual correlation with MP.

    Args:
        rows (list[dict]): Identical-set rows, from _rows.
        models (list[str]): MLIP model keys.
        same (dict): Identical-set statistics, from _identical_set.

    Returns:
        Dict model -> {delta_sigma_vs_mp, delta_sigma_vs_mp_ci95_cluster_system, corr_residual_with_mp}.
    """
    sat = {}
    for m in models:
        d_sig = []
        bysys = defaultdict(list)
        for r in rows:
            bysys[r["system"]].append(r)
        systems = sorted(bysys)
        rng = random.Random(int(hashlib.sha256(("sat|" + m).encode()).hexdigest()[:8], 16))
        for _ in range(bundle.N_BOOT):
            res_m, res_d = [], []
            for _ in range(len(systems)):
                for r in bysys[systems[rng.randrange(len(systems))]]:
                    res_m.append(r[m] - r["exp"])
                    res_d.append(r["mp"] - r["exp"])
            d_sig.append(bundle.stats_of(res_m)["sigma_robust"] - bundle.stats_of(res_d)["sigma_robust"])
        sat[m] = {
            "delta_sigma_vs_mp": round(same[m]["all"]["sigma_robust"] - same["mp"]["all"]["sigma_robust"], 3),
            "delta_sigma_vs_mp_ci95_cluster_system": [
                round(bundle.quantile(d_sig, 0.025), 3),
                round(bundle.quantile(d_sig, 0.975), 3),
            ],
            "corr_residual_with_mp": round(_corr([r[m] - r["exp"] for r in rows], [r["mp"] - r["exp"] for r in rows]), 3),
        }
    return sat


def _health(formation: dict, results: dict, items: list[dict], dropped: dict, models: list[str]) -> dict:
    """Run-health counts per model: errors, non-converged, exploded, symmetry changes, times, dropped pairs.

    Args:
        formation (dict): Model -> compound id -> row, as returned by mlip_formation.
        results (dict): Model -> id -> raw result record.
        items (list[dict]): Input items.
        dropped (dict): Model -> Counter of dropped-pair reasons.
        models (list[str]): MLIP model keys.

    Returns:
        Dict model -> health counts and times in seconds.
    """
    itm = {i["id"]: i for i in items}
    health = {}
    for m in models:
        f = list(formation[m].values())
        el = [results[m][i["id"]] for i in items if i["kind"] == "element" and i["id"] in results[m]]
        health[m] = {
            "compounds_run": len(f),
            "errors": sum(1 for x in f if x["status"] != "ok"),
            "not_converged": sum(1 for x in f if x["status"] == "ok" and not x["converged"]),
            "exploded": sum(1 for x in f if x.get("exploded")),
            "symmetry_changed": sum(1 for x in f if x["status"] == "ok" and x["sg_in"] != x["sg_out"]),
            "elements_run": len(el),
            "elements_not_ok": sum(1 for x in el if ref_bad(x, itm.get(x["id"]))),
            "elements_flagged_volume_only": sum(1 for x in el if x.get("exploded") and not ref_bad(x, itm.get(x["id"]))),
            "median_time_s": round(bundle.median([x["t_s"] for x in f if x["status"] == "ok"]) or 0, 3),
            "total_time_s": round(
                sum(x["t_s"] for x in f) + sum((x.get("t_relax_s") or 0) + (x.get("t_sp_s") or 0) for x in el),
                1,
            ),
            "default_pairs_dropped": dict(dropped[m]),
        }
    return health


def _write_csvs(out_dir: Path, rows: list[dict], models: list[str], formation: dict, itm: dict) -> None:
    """Write pairs_identical_set.csv (one row per identical record) and mlip_by_compound.csv (one row per compound).

    Args:
        out_dir (Path): Output directory.
        rows (list[dict]): Identical-set rows, from _rows.
        models (list[str]): MLIP model keys.
        formation (dict): Model -> compound id -> row, as returned by mlip_formation.
        itm (dict): Input items keyed by id.
    """
    cols = [
        "rid",
        "formula",
        "system",
        "chem",
        "exp",
        "mp",
        "oqmd",
        "mp_mid",
        "oqmd_mid",
        "mp_smatch",
        "mp_match",
        "oqmd_match",
        "mp_ehull",
        "mp_npoly",
        "paper",
        "method",
    ]
    mcols = []
    for m in models:
        mcols += [m] + [f"{m}|{v}" for v in ("relax_raw", "relax_corr", "sp_raw", "sp_corr")]
    with (out_dir / "pairs_identical_set.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols + mcols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with (out_dir / "mlip_by_compound.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "model",
                "id",
                "src",
                "formula",
                "dft_ef_kJ",
                "ef_relax_corr_kJ",
                "ef_relax_raw_kJ",
                "ef_sp_corr_kJ",
                "ef_sp_raw_kJ",
                "status",
                "converged",
                "exploded",
                "refs_flag",
                "nsteps",
                "vol_ratio",
                "sg_in",
                "sg_out",
                "time_s",
            ]
        )
        for m in models:
            for cid, x in sorted(formation[m].items()):
                it = itm[cid]
                w.writerow(
                    [
                        m,
                        cid,
                        it["src"],
                        it["formula"],
                        round(it["dft_ef_eV"] * kj_per_ev, 3),
                        to_kj(x, "ef_relax_corr"),
                        to_kj(x, "ef_relax_raw"),
                        to_kj(x, "ef_sp_corr"),
                        to_kj(x, "ef_sp_raw"),
                        x["status"],
                        x["converged"],
                        x["exploded"],
                        x["refs_flag"],
                        x["nsteps"],
                        x["vol_ratio"],
                        x["sg_in"],
                        x["sg_out"],
                        round(x["t_s"], 3),
                    ]
                )


def _load_items(paths: list[str]) -> list[dict]:
    """Input items of one or more structure files.

    Args:
        paths (list[str]): Input JSON files, each with an "items" list.

    Returns:
        The items, in file order.
    """
    items = []
    for f in paths:
        with Path(f).open(encoding="utf-8") as fh:
            items.extend(json.load(fh)["items"])
    return items


def _parse_args() -> argparse.Namespace:
    """Parse the command-line options; unknown options are ignored (the vendor options are read earlier).

    Returns:
        The parsed options.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--inputs", nargs="+", default=[str(root / "inputs" / "structures.json")])
    ap.add_argument("--out", required=True)
    ap.add_argument("--site-rev", default="HEAD", help="orchid-project.org revision of the calibration code (use --site-rev=X)")
    ap.add_argument("--data-rev", default="HEAD", help="orchid-data revision of the reference data (use --data-rev=X)")
    ap.add_argument("--variant", default="relax_corr", help="primary MLIP value: relax|sp x raw|corr")
    a, _ = ap.parse_known_args()
    return a


def main() -> None:
    """Run the full-path MLIP analysis and write CALIBRATION_MLIP.json, pairs_identical_set.csv and mlip_by_compound.csv."""
    a = _parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dpp = site / "data" / "literature" / "dft_parity.json"
    pub = bundle.published(str(site / "data" / "literature"))[0]
    # DFT pairs: exactly the DFT calibration's (default subset per point, strata, systems, sigma_exp)
    pairs_dft, dp = bundle.calibration_pairs(str(site / "data"), pub)
    records = dp["records"]
    items = _load_items(a.inputs)
    itm = {i["id"]: i for i in items}
    results, meta = load_results(a.results)
    formation = mlip_formation(results, items)
    models = sorted(formation)

    # the matched DFT entry of every default point (calibration_pairs keeps rid, not mid)
    lookup = _MlipLookup(formation, _default_points(dp, records), itm)
    pairs, dropped = _mlip_pairs(lookup, pairs_dft, records, models, variant=a.variant)
    cal = bundle.calibrate(pairs, {(s_, bundle.stratum_key(p), p["system"]) for s_, pts in pairs.items() for p in pts})

    same, same_rids = _identical_set(pairs, models)
    rows = _rows(records, lookup, pairs, models, same_rids, variant=a.variant)
    comps, vs_dft, med, variants = _compound_tables(rows, models)
    sat = _saturation(rows, models, same)
    health = _health(formation, results, items, dropped, models)
    tol_mev = bundle.HULL_TOL_MEV if hasattr(bundle, "HULL_TOL_MEV") else 0.0
    inputs_hash = ", ".join(f"{Path(f).name} {sha(f)}" for f in a.inputs)
    out = {
        "schema": "orchid-mlip-calibration-v2 (layout of orchid-data espei/dft/CALIBRATION.json: method, pairs_release, "
        "sources{source: {all, strata}}, lso; plus same_set, mlip_vs_dft, per_compound_median, variants, "
        "saturation, health, models)",
        "method": {
            "pairs": f"build_espei_bundle.calibration_pairs (orchid-project.org {site_sha[:10]}): DFT pairs exactly as the DFT "
            "calibration; an MLIP pair = the MP pair of the same record with the MLIP formation enthalpy "
            "of the same matched MP entry (structure-matched polymorph when the record names one)",
            "residual": "MLIP - experiment, kJ/mol-atom (rounded to 1e-3 as dft_parity res)",
            "sigma_robust": "1.4826 * MAD",
            "bootstrap": bundle.N_BOOT,
            "ci": "percentile 95 %",
            "ci_cluster": "ci95_cluster_system: resample chemical systems with all their pairs (quote this one)",
            "min_stratum": bundle.MIN_STRATUM,
            "leave_system_out": True,
            "bias_applied": False,
            "hull_tolerance_meV_per_atom": tol_mev,
            "hull_tolerance_kJ_per_mol_atom": round(bundle.HULL_TOL_KJ, 4),
            "primary_variant": a.variant,
            "variants": {
                "relax": "MLIP relaxation (FIRE + FrechetCellFilter, fmax 0.02 eV/A, <= 1000 steps) of the "
                "DFT-relaxed MP structure; elemental references = MP ground states relaxed alike",
                "sp": "MLIP single point on the DFT-relaxed structure",
                "corr": "plus the MP energy correction of the matched entries (compound minus composition-"
                "weighted elemental corrections), unless the model predicts corrected energies",
                "raw": "model output without added correction",
            },
            "excluded": "pairs whose MLIP relaxation (compound or elemental reference) did not converge, errored or "
            "exploded; pairs whose matched MP entry is not among the computed structures",
            "chemistry_classes": bundle.CHEM_TEXT,
            "match_quality": bundle.MATCH_TEXT,
            "eV_to_kJ_per_mol": kj_per_ev,
            "cuzr_indirect_sigma_kJ": _cuzr_indirect,
        },
        "pairs_release": f"dft_parity.json sha256 {sha(dpp)} (literature sha256 {dp['inputs']['literature']['sha256']}); inputs sha256 {inputs_hash}",
        "code": {
            "orchid-project.org": {"commit": site_sha, "build_espei_bundle.py sha256": sha(bundle.__file__)},
            "orchid_mlip_benchmark": git(str(root), "rev-parse", "HEAD"),
            "orchid-data (reference data)": data_sha,
        },
        "sources": cal["sources"],
        "lso": cal["lso"],
        "same_set": {"n_records": len(rows), "n_compounds": len(comps), "stats": same},
        "mlip_vs_dft": vs_dft,
        "per_compound_median": med,
        "variants": variants,
        "saturation": sat,
        "health": health,
        "models": {
            m: {
                "spec": (meta.get(m) or {}).get("spec"),
                "env": (meta.get(m) or {}).get("env"),
                "fmax": (meta.get(m) or {}).get("fmax"),
                "steps": (meta.get(m) or {}).get("steps"),
            }
            for m in models
        },
    }
    out = bundle.round_floats(out)
    with (out_dir / "CALIBRATION_MLIP.json").open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    _write_csvs(out_dir, rows, models, formation, itm)
    print(
        json.dumps(
            {
                "site": site_sha[:10],
                "data": data_sha[:10],
                "tol_meV": tol_mev,
                "dft_pairs": {k: len(v) for k, v in pairs_dft.items()},
                "mlip_pairs": {m: len(pairs[m]) for m in models},
                "identical_records": len(rows),
                "identical_compounds": len(comps),
            },
            indent=1,
        )
    )
    for k in ["mp", "oqmd", *models]:
        s_ = same[k]["all"]
        ci = s_["ci95_cluster_system"]["sigma_robust"]
        print(
            f"{k:<14} n={s_['n']:4d} MAE={s_['mae']:5.2f} bias={s_['bias_median']:+5.2f} sigma={s_['sigma_robust']:5.2f} cluster{ci} RMSE={s_['rmse']:5.2f}"
        )


if __name__ == "__main__":
    main()
