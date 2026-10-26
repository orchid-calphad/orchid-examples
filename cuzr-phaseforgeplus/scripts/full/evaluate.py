# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/evaluate.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Per-type residuals of a TDB against the assembled ORCHID datasets (data/fit, data/holdout, data/dft_check) and the
MLIP anchor (data/mlip_anchor), with the ESPEI fork's own residual functions (driving force for ZPF in J/mol,
activity as chemical-potential residual in J/mol, HM_FORM / SM_FORM / HM_MIX in J/mol-atom or J/K/mol-atom).

Usage: python evaluate.py <tdb> <label> [--values values.json]   -> results/eval_<label>.json
A TDB without VV symbols (e.g. CuZr-94Zen-LB.tdb) gets one inert VV symbol so ESPEI can build its context.
Non-finite residuals are counted and listed (dataset file), never silently dropped.
"""
import json, sys, glob, os, re, collections, warnings, tempfile
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from pycalphad import Database
from espei.datasets import load_datasets
from espei.error_functions.context import setup_context
from espei.utils import database_symbols_to_fit, unpack_piecewise

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, SGTE_TDB, orchid, PFP_SRC  # noqa: F401 (paths: _paths.py)
MAN = json.load(open(ROOT / "data/manifest.json"))
TYPE = {Path(d["file"]).name: d["type"] for d in MAN["datasets"]}


def load_db(tdb, values=None):
    txt = Path(tdb).read_text()
    if not re.search(r"FUNCTION\s+VV\d+", txt):
        txt = txt.replace("$ Binary system", " FUNCTION VV9999 298.15 0; 6000 N !\n PAR  L(FCC_A1,CU,ZR:VA;5),, +VV9999;,, N inert !\n$ Binary system", 1)
        f = tempfile.NamedTemporaryFile("w", suffix=".tdb", delete=False); f.write(txt); f.close(); tdb = f.name
    db = Database(str(tdb))
    if values:
        db.symbols.update({k: float(v) for k, v in values.items()})
    return db


def group_files():
    g = collections.defaultdict(list)
    for split in ("fit", "holdout", "dft_check"):
        for f in sorted(glob.glob(str(ROOT / f"data/{split}/*.json"))):
            g[(split, TYPE.get(Path(f).name, "?"))].append(f)
    for f in sorted(glob.glob(str(ROOT / "data/mlip_anchor/*.json"))):
        g[("mlip_anchor", json.load(open(f))["output"] + ":" + json.load(open(f))["phases"][0])].append(f)
    return g


def residuals(dbspec, files):
    db = load_db(*dbspec)
    syms = database_symbols_to_fit(db)
    p = np.array([float(unpack_piecewise(db.symbols[s])) for s in syms])
    ctx = setup_context(db, load_datasets(files), syms)
    res, wts, per = [], [], {}
    for r in ctx["residual_objs"]:
        try:
            rr, ww = r.get_residuals(p)
        except ValueError as e:      # residual type with no data in this group (np.concatenate of nothing)
            if "at least one array" in str(e): continue
            raise
        if len(rr):
            per[type(r).__name__] = len(rr)
        res += list(rr); wts += list(ww)
    return np.array(res, float), np.array(wts, float), per


def evaluate(dbspec, only=None):
    db = dbspec
    out = {}
    for key, files in group_files().items():
        if only and key[0] not in only: continue
        r, w, per = residuals(db, files)
        bad = [f for f in files if not np.all(np.isfinite(residuals(db, [f])[0]))] if not np.all(np.isfinite(r)) else []
        fin = r[np.isfinite(r)]
        out["%s|%s" % key] = {"n": int(len(r)), "n_nonfinite": int((~np.isfinite(r)).sum()), "nonfinite_files": [Path(b).name for b in bad],
                              "rmse": float(np.sqrt(np.mean(fin ** 2))) if len(fin) else None,
                              "mean_abs": float(np.mean(np.abs(fin))) if len(fin) else None,
                              "residual_classes": per}
    return out


if __name__ == "__main__":
    tdb, label = sys.argv[1], sys.argv[2]
    vals = json.load(open(sys.argv[4])) if len(sys.argv) > 4 and sys.argv[3] == "--values" else None
    out = {"tdb": str(tdb), "values": vals, "groups": evaluate((tdb, vals))}
    json.dump(out, open(ROOT / f"results/eval_{label}.json", "w"), indent=1)
    for k, v in sorted(out["groups"].items()):
        print("%-34s n=%4d nonfinite=%d rmse=%s %s" % (k, v["n"], v["n_nonfinite"], "%.1f" % v["rmse"] if v["rmse"] is not None else "-", v["nonfinite_files"][:3]))
