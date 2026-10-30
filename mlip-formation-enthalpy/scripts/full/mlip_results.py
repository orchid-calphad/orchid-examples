"""Per-structure MLIP results: loading, formation energies and the reference checks.

Reads the JSON-lines result files written by run_mlip.py and turns the relaxed and single-point energies of every
compound into formation enthalpies relative to the elemental references, with and without the MP2020 correction.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

here = Path(__file__).resolve().parent
kj_per_ev = 96.485  # kJ/mol per eV/atom, as build_dft_parity.KJ_PER_EV
_models_spec = json.loads((here / "models.json").read_text(encoding="utf-8"))
_molecular = {"H", "N", "O", "F", "Cl", "Br", "I"}  # molecular-crystal elements: no-vdW MLIPs expand them freely


def load_results(dirs: list[str]) -> tuple[dict, dict]:
    """Read the per-structure MLIP result lines of the given directories.

    Args:
        dirs (list[str]): Directories holding *.jsonl result files.

    Returns:
        Tuple (results, meta): results maps model -> compound or element id -> record; meta maps model -> the first
        metadata line (spec, environment, fmax, steps).
    """
    results = defaultdict(dict)
    meta = {}
    for d in dirs:
        for f in sorted(Path(d).glob("*.jsonl"), key=str):
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except Exception:  # a torn last line from a killed job is ignored and the item rerun
                        continue
                    if r.get("meta"):
                        meta.setdefault(r["model"], r)
                        continue
                    results[r["model"]][r["id"]] = r
    return results, meta


def to_kj(rec: dict, key: str) -> float | None:
    """A stored eV/atom value converted to kJ/mol-atom and rounded to 3 decimals.

    Args:
        rec (dict): Record holding the value.
        key (str): Key of the value in rec.

    Returns:
        The value in kJ/mol-atom, or None if the key is absent.
    """
    return round(rec[key] * kj_per_ev, 3) if key in rec else None


def ref_bad(q: dict, e: dict | None) -> bool:
    """Whether an elemental reference relaxation is unusable.

    Unusable means not converged, or exploded. For molecular-crystal elements the volume criterion is dropped (cell
    expansion without dispersion is the model's physics); the energy criteria stay.

    Args:
        q (dict): Result record of the elemental reference relaxation.
        e (dict | None): Input item of the reference, with its element symbol; None if unknown.

    Returns:
        True if the reference must not be used.
    """
    if q.get("status") != "ok" or not q.get("converged"):
        return True
    if e and e.get("element") in _molecular:
        e1 = q["e_relax_eV_per_atom"]
        return not (-15 < e1 < 5) or e1 < q["e_sp_eV_per_atom"] - 1.0
    return bool(q.get("exploded"))


def mlip_formation(results: dict, items: list[dict]) -> dict:
    """MLIP formation energies of every compound, relaxed and single-point, raw and with the MP correction.

    Correction = raw + MP correction, unless the model's training targets already include the MP2020 corrections
    (models.json predicts_mp2020_corrected), then corr = raw.

    Args:
        results (dict): Model -> id -> result record, as returned by load_results.
        items (list[dict]): Input items (compounds and elemental references).

    Returns:
        Dict model -> compound id -> row with status, flags, formation energies in eV/atom (ef_<relax|sp>_<raw|corr>).
    """
    comp = {i["id"]: i for i in items if i["kind"] == "compound"}
    eref = {(i["src"], i["element"]): i for i in items if i["kind"] == "element"}
    out = {}
    for model, recs in results.items():
        res = {}
        pre = bool((_models_spec.get(model) or {}).get("predicts_mp2020_corrected"))
        for cid, c in comp.items():
            r = recs.get(cid)
            if not r:
                continue
            row = {
                "status": r["status"],
                "converged": r.get("converged"),
                "exploded": r.get("exploded"),
                "nsteps": r.get("nsteps"),
                "vol_ratio": r.get("vol_ratio"),
                "sg_in": r.get("sg_in"),
                "sg_out": r.get("sg_out"),
                "t_s": (r.get("t_relax_s") or 0) + (r.get("t_sp_s") or 0),
            }
            refs = [(eref.get((c["src"], el)), x) for el, x in c["comp"].items()]
            rr = [(recs.get(e["id"]) if e else None, x, e) for e, x in refs]
            ok = r["status"] == "ok" and all(q and q["status"] == "ok" for q, _, _ in rr)
            row["refs_ok"] = all(q and q["status"] == "ok" for q, _, _ in rr)
            row["refs_flag"] = any(q and ref_bad(q, e) for q, _, e in rr)
            if ok:
                corr = c.get("mp_correction_eV_per_atom", 0.0) - sum(
                    x * e.get("mp_correction_eV_per_atom", 0.0) for _, x, e in rr
                )
                for var, key in (("relax", "e_relax_eV_per_atom"), ("sp", "e_sp_eV_per_atom")):
                    raw = r[key] - sum(x * q[key] for q, x, _ in rr)
                    row[f"ef_{var}_raw"] = raw
                    row[f"ef_{var}_corr"] = raw + (0.0 if pre else corr)
                row["mp_corr"] = corr
            res[cid] = row
        out[model] = res
    return out
