# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/build_data.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Assemble the Cu-Zr fit, hold-out and dft_check datasets from the ORCHID ESPEI bundle.

The selection, hold-out and weight rules, and the provenance of the copied v2 script, are recorded verbatim in
``_doc_text``; it is written to data/manifest.json as the ``doc`` field.

Writes data/fit/, data/holdout/ and data/dft_check/ (one JSON file per selected dataset) and data/manifest.json.
Usage: python scripts/full/build_data.py (needs ORCHID_DATA).
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import orchid, sgte_tdb
from _paths import root as work_root

root = str(work_root)
bundle = str(orchid("espei"))
reg_path = str(orchid("phases/systems/CU-ZR.json"))
tdb = str(sgte_tdb)
target = "CuZr-94Zen-LB"
hold_refs = ("10.1016/0022-5088(85)90279-6", "10.1007/s11669-017-0522-2")
hold_acr_t = {1223.0, 1304.0, 1453.0, 1573.0, 1723.0}
hold_hmr_phase = {"CU10ZR7"}
exclude = {
    "experimental/CU-ZR/CuZr-94Zen-LB.hmr-07.json": "phase LIQUID at 298.15 K at compound compositions (ambiguous label)",
    "experimental/CU-ZR/CuZr-94Zen-LB.hmr-08.json": "phase LIQUID at 298.15 K at compound compositions (ambiguous label)",
}
_doc_text = """Copied from orchid-espei-demo scripts/v2/build_data_v2.py (same selection, hold-out and weights, so the two demos use
identical data), with three changes for the PhaseForge demo: output under data/, literature DFT goes to a separate
`dft_check` split (held-out check of the MLIP energies, never fitted: the MLIP supplies the energies), and the target
TDB is CuZr-PhaseForge-MLIP.tdb, whose phase names and sublattice models are those of CuZr-94Zen-LB.tdb (so the
registry mapping to CuZr-94Zen-LB applies unchanged).

Original docstring: v2 (2026-10-23 PM): assemble the Cu-Zr working datasets from the FIXED ORCHID ESPEI bundle (E1-E6 applied in the
bundle) plus the optional phase constraints (invariant brackets, assessed assemblages, liquid stability).

Selection (fixed before any v2 run; PLAN.md Amendment 2):
  phase evidence: experimental/CU-ZR ZPF, invariants/CU-ZR, derived/invariant-brackets/CU-ZR, assessed/CU-ZR/CuZr-94Zen-LB.*,
    derived/liquid-stability/CU-ZR/CuZr-94Zen-LB.* -- a dataset is used when every phase name it uses maps, in the
    ORCHID phase registry, to the same registry phase in its own folder's TDB and in CuZr-94Zen-LB.tdb (for a bracket,
    also every phase of the invariant dataset it was derived from)
  thermochemistry: experimental ACR (Zaitsev 2003 via Wang 2006; liquid reference states, bundle E5), HMR/SMR (Turchanin
    2008) with the bundle's optional minimum-sigma weight (orchid_weight_floor, E6); CuZr-94Zen-LB hmr-07/08 excluded
    (phase LIQUID at 298 K at compound compositions: ambiguous label, as in v1)
  DFT: dft/literature/CU-ZR/CuZr-94Zen-LB.* (Zhou 2010; calibrated sigma, weight per value as written)
Hold-out (as in v1, plus invariant temperatures): ZPF datasets whose reference is Glimois 1985 or the Liu 2017 table, and
  the brackets derived from them; activities at 1223, 1304, 1453, 1573, 1723 K; HMR/SMR of Cu10Zr7.
Per-type weights: each data type's dataset weights are multiplied by w_t = sqrt(Nbar / N_t) (N_t = residuals of type t in
  the fit set, Nbar = mean over types) so that each type's summed log-likelihood scales as 1/N_t, i.e. contributes
  comparably (ESPEI's sigma is default / weight, so the log-likelihood scales with weight^2).
"""


def sha(p: str | Path) -> str:
    """SHA-256 hex digest of a file's bytes.

    Args:
        p (str | Path): File to hash.

    Returns:
        Hex digest of the file contents.
    """
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


reg = json.loads(Path(reg_path).read_text())
pid = collections.defaultdict(dict)  # stem -> phase_name -> registry id
for p in reg["phases"]:
    for m in p.get("models") or []:
        pid[m["tdb"][:-4]][m["phase_name"]] = p["id"]


def stem_of(path: str) -> str:
    """TDB stem of a dataset, from its bundle-relative path.

    Args:
        path (str): Bundle-relative dataset path.

    Returns:
        The file name up to the second-to-last dot, or up to the first dot for names carrying a CU-ZR-p tag.
    """
    name = Path(path).name
    return name.rsplit(".", 2)[0] if not name.count("CU-ZR-p") else name.split(".")[0]


def compatible(d: dict, stem: str) -> bool:
    """Whether every phase name of a dataset maps to the same registry phase as in the target TDB.

    Args:
        d (dict): Parsed bundle dataset; its ``phases`` list is checked.
        stem (str): TDB stem of the dataset's own folder.

    Returns:
        True if each phase name has a registry id in ``stem`` that equals its id in the target TDB.
    """
    for ph in d["phases"]:
        a, b = pid.get(stem, {}).get(ph), pid[target].get(ph)
        if a is None or b is None or a != b:
            return False
    return True


def kind(rel: str, d: dict) -> str:
    """Data type of a bundle dataset, from its folder and its output field.

    Args:
        rel (str): Bundle-relative dataset path.
        d (dict): Parsed bundle dataset.

    Returns:
        One of ZPF_bracket, ZPF_liquid, ZPF_assessed, DFT_HM_FORM, ZPF_literature, ACR, or the dataset's output field.
    """
    if rel.startswith("derived/invariant-brackets"):
        data_type = "ZPF_bracket"
    elif rel.startswith("derived/liquid-stability"):
        data_type = "ZPF_liquid"
    elif rel.startswith("assessed/"):
        data_type = "ZPF_assessed"
    elif rel.startswith("dft/"):
        data_type = "DFT_HM_FORM"
    elif d["output"] == "ZPF":
        data_type = "ZPF_literature"
    elif d["output"].startswith("ACR"):
        data_type = "ACR"
    else:
        data_type = d["output"]
    return data_type


def nres(d: dict) -> int:
    """Number of fit residuals in a dataset.

    Args:
        d (dict): Parsed bundle dataset.

    Returns:
        Count of ZPF points (hyperplane placeholders excluded) or of the scalar values of other outputs.
    """
    if d["output"] == "ZPF":
        return sum(1 for r in d["values"] for v in r if v[0] != "__HYPERPLANE__")
    return sum(len(c) for a in d["values"] for c in a)


def is_holdout_src(d: dict) -> bool:
    """Whether a dataset's reference is one of the hold-out sources.

    Args:
        d (dict): Parsed bundle dataset.

    Returns:
        True if its ``reference`` contains one of the DOIs in ``hold_refs``.
    """
    return any(r in (d.get("reference") or "") for r in hold_refs)


cands = []
for g in ("experimental", "invariants", "derived/invariant-brackets"):
    cands += sorted(str(p) for p in Path(bundle, g, "CU-ZR").glob("*.json"))
cands += [
    f for f in sorted(str(p) for p in Path(bundle, "assessed", "CU-ZR").glob(f"{target}.*.json")) if not f.endswith(".dense.json")
] + sorted(str(p) for p in Path(bundle, "derived", "liquid-stability", "CU-ZR").glob(f"{target}.*.json"))
cands += sorted(str(p) for p in Path(bundle, "dft", "literature", "CU-ZR").glob(f"{target}.*.json"))
man = {"bundle": bundle, "tdb": {"path": tdb, "sha256": sha(tdb)}, "datasets": [], "excluded": [], "doc": _doc_text}
sel = []
for f in cands:
    rel = os.path.relpath(f, bundle)
    d = json.loads(Path(f).read_text())
    if rel in exclude:
        man["excluded"].append({"source": rel, "reason": exclude[rel]})
        continue
    if "model" in (d.get("tags") or []):
        man["excluded"].append({"source": rel, "reason": "model evidence class"})
        continue
    st = target if rel.startswith(("assessed/", "derived/liquid", "dft/")) else stem_of(rel)
    if not compatible(d, st):
        man["excluded"].append({"source": rel, "reason": f"a phase name does not map to the same registry phase in {target}.tdb"})
        continue
    if rel.startswith("derived/invariant-brackets"):
        if not Path(bundle, d["orchid"]["derived_from"]).exists():
            man["excluded"].append(
                {"source": rel, "reason": "bracket source dataset no longer in the bundle (removed in the cleanup)"}
            )
            continue
        src = json.loads(Path(bundle, d["orchid"]["derived_from"]).read_text())
        if not compatible(src, st):
            man["excluded"].append(
                {
                    "source": rel,
                    "reason": f"bracket derived from a dataset whose phase names do not all map to {target}.tdb (source excluded)",
                }
            )
            continue
    k = kind(rel, d)
    w = copy.deepcopy(d)
    part = "fit"
    if k == "ZPF_literature" and is_holdout_src(d):
        part = "holdout"
    if k == "DFT_HM_FORM":
        part = "dft_check"
    if k == "ZPF_bracket":
        src = json.loads(Path(bundle, d["orchid"]["derived_from"]).read_text())
        if is_holdout_src(src):
            part = "holdout"
    if k == "ACR" and float(d["conditions"]["T"]) in hold_acr_t:
        part = "holdout"
    if k in ("HMR", "SMR", "HM_FORM", "SM_FORM"):  # bundle renamed HMR/SMR -> HM_FORM/SM_FORM
        if set(d["phases"]) & hold_hmr_phase:
            part = "holdout"
        fl = d.get("orchid_weight_floor")
        if fl:
            w["weight"] = fl["suggested_weight"]
            w["orchid_demo_note"] = "weight = orchid_weight_floor.suggested_weight (bundle E6)"
    sel.append((rel, k, part, w))
n_fit = collections.Counter()
for _rel, k, part, w in sel:
    if part == "fit":
        n_fit[k] += nres(w)
nbar = sum(n_fit.values()) / len(n_fit)
wt_v2 = {k: math.sqrt(nbar / n) for k, n in n_fit.items()}  # orchid-espei-demo v2 rule (recorded only)
wt = dict.fromkeys(
    n_fit, 1.0
)  # PhaseForge demo: bundle weights unchanged (PhaseForgePlus/paper configuration; PLAN.md amendment 1)
for sub in ("fit", "holdout", "dft_check"):
    (Path(root) / "data" / sub).mkdir(parents=True, exist_ok=True)
    for f in (Path(root) / "data" / sub).glob("*.json"):
        f.unlink()
for rel, k, part, w in sel:
    base = w.get("weight", 1.0)
    if isinstance(base, list):
        w["weight"] = [x * wt.get(k, 1.0) for x in base]
    else:
        w["weight"] = base * wt.get(k, 1.0)
    w["orchid_demo_type_weight"] = wt.get(k, 1.0)
    out = f"{root}/data/{part}/" + rel.replace("/", "__")
    Path(out).write_text(json.dumps(w, indent=1))
    man["datasets"].append(
        {
            "file": os.path.relpath(out, root),
            "source": rel,
            "source_sha256": sha(Path(bundle, rel)),
            "split": part,
            "type": k,
            "n_residuals": nres(w),
            "reference": w.get("reference"),
            "bundle_weight": base,
            "type_weight": wt.get(k, 1.0),
            "tags": w.get("tags"),
        }
    )
man["type_counts_fit"] = dict(n_fit)
man["type_weights"] = wt
man["type_weight_rule"] = (
    f"w_t = 1 (bundle weights); v2 rule sqrt(Nbar/N_t) recorded in type_weights_v2_not_used, Nbar = {nbar:.2f}"
)
man["type_weights_v2_not_used"] = wt_v2
Path(root, "data", "manifest.json").write_text(json.dumps(man, indent=1))
c = collections.Counter((x["split"], x["type"]) for x in man["datasets"])
n = collections.Counter()
for x in man["datasets"]:
    n[(x["split"], x["type"])] += x["n_residuals"]
for key in sorted(c):
    print(key, "datasets", c[key], "residuals", n[key], f"type weight {wt.get(key[1], 1):.3f}")
print("excluded", len(man["excluded"]))
