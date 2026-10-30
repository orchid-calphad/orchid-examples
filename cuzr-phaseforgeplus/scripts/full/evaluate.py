# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/evaluate.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Per-type residuals of a TDB against the assembled ORCHID datasets and the MLIP anchor.

The datasets are data/fit, data/holdout and data/dft_check, plus data/mlip_anchor. Residuals come from the ESPEI fork's
own residual functions (driving force for ZPF in J/mol, activity as chemical-potential residual in J/mol, HM_FORM /
SM_FORM / HM_MIX in J/mol-atom or J/K/mol-atom).

Usage: python evaluate.py <tdb> <label> [--values values.json]   -> results/eval_<label>.json
A TDB without VV symbols (e.g. CuZr-94Zen-LB.tdb) gets one inert VV symbol so ESPEI can build its context.
Non-finite residuals are counted and listed (dataset file), never silently dropped.
"""

from __future__ import annotations

import collections
import json
import re
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
from espei.datasets import load_datasets
from espei.error_functions.context import setup_context
from espei.utils import database_symbols_to_fit, unpack_piecewise
from pycalphad import Database

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import root

# Set after the imports (a statement between imports is flagged by ruff E402). Only warnings raised during the
# pycalphad/espei can now reach stderr; no written output is affected.
warnings.filterwarnings("ignore")

manifest = json.loads((root / "data/manifest.json").read_text())
dataset_type = {Path(d["file"]).name: d["type"] for d in manifest["datasets"]}


def load_db(tdb: str, values: dict | None = None) -> Database:
    """Load a TDB, adding an inert VV symbol when the file has none, and optionally override symbols.

    Args:
        tdb (str): Path of the TDB file.
        values (dict | None): Optional symbol name -> value overrides, applied to ``db.symbols``.

    Returns:
        The pycalphad Database.
    """
    txt = Path(tdb).read_text()
    if not re.search(r"FUNCTION\s+VV\d+", txt):
        txt = txt.replace(
            "$ Binary system",
            " FUNCTION VV9999 298.15 0; 6000 N !\n PAR  L(FCC_A1,CU,ZR:VA;5),, +VV9999;,, N inert !\n$ Binary system",
            1,
        )
        with tempfile.NamedTemporaryFile("w", suffix=".tdb", delete=False) as f:
            f.write(txt)
        tdb = f.name
    db = Database(str(tdb))
    if values:
        db.symbols.update({k: float(v) for k, v in values.items()})
    return db


def group_files() -> collections.defaultdict:
    """Group the dataset files by (split, data type) and the MLIP anchor files by (output, phase).

    Returns:
        A defaultdict mapping each group key to the list of file paths (strings) in that group.
    """
    g = collections.defaultdict(list)
    for split in ("fit", "holdout", "dft_check"):
        for f in sorted(str(p) for p in (root / f"data/{split}").glob("*.json")):
            g[(split, dataset_type.get(Path(f).name, "?"))].append(f)
    for f in sorted(str(p) for p in (root / "data/mlip_anchor").glob("*.json")):
        d = json.loads(Path(f).read_text())
        g[("mlip_anchor", d["output"] + ":" + d["phases"][0])].append(f)
    return g


def residuals(dbspec: tuple, files: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Residuals and weights of a TDB against a list of ORCHID dataset files.

    Args:
        dbspec (tuple): Arguments for ``load_db``: (tdb path, optional values dict).
        files (list[str]): Dataset file paths to load.

    Returns:
        Tuple of the residual array, the weight array (both float), and a dict counting the residuals per residual class.
    """
    db = load_db(*dbspec)
    syms = database_symbols_to_fit(db)
    p = np.array([float(unpack_piecewise(db.symbols[s])) for s in syms])
    ctx = setup_context(db, load_datasets(files), syms)
    res, wts, per = [], [], {}
    for r in ctx["residual_objs"]:
        try:
            rr, ww = r.get_residuals(p)
        except ValueError as e:  # residual type with no data in this group (np.concatenate of nothing)
            if "at least one array" in str(e):
                continue
            raise
        if len(rr):
            per[type(r).__name__] = len(rr)
        res += list(rr)
        wts += list(ww)
    return np.array(res, float), np.array(wts, float), per


def evaluate(dbspec: tuple, only: set | None = None) -> dict:
    """Residual statistics of a TDB for every dataset group.

    Args:
        dbspec (tuple): Arguments for ``load_db``: (tdb path, optional values dict).
        only (set | None): If given, restrict to groups whose split (or mlip_anchor) is in this set.

    Returns:
        Dict keyed by "split|type" with n (residual count), n_nonfinite, nonfinite_files, rmse and mean_abs (J/mol or
        J/mol-atom, as the residual type), and residual_classes.
    """
    db = dbspec
    out = {}
    for key, files in group_files().items():
        if only and key[0] not in only:
            continue
        r, w, per = residuals(db, files)
        bad = [f for f in files if not np.all(np.isfinite(residuals(db, [f])[0]))] if not np.all(np.isfinite(r)) else []
        fin = r[np.isfinite(r)]
        out[f"{key[0]}|{key[1]}"] = {
            "n": int(len(r)),
            "n_nonfinite": int((~np.isfinite(r)).sum()),
            "nonfinite_files": [Path(b).name for b in bad],
            "rmse": float(np.sqrt(np.mean(fin**2))) if len(fin) else None,
            "mean_abs": float(np.mean(np.abs(fin))) if len(fin) else None,
            "residual_classes": per,
        }
    return out


if __name__ == "__main__":
    tdb, label = sys.argv[1], sys.argv[2]
    vals = json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv) > 4 and sys.argv[3] == "--values" else None
    out = {"tdb": str(tdb), "values": vals, "groups": evaluate((tdb, vals))}
    (root / f"results/eval_{label}.json").write_text(json.dumps(out, indent=1))
    for k, v in sorted(out["groups"].items()):
        rmse_text = f"{v['rmse']:.1f}" if v["rmse"] is not None else "-"
        print(f"{k:<34} n={v['n']:4d} nonfinite={v['n_nonfinite']:d} rmse={rmse_text} {v['nonfinite_files'][:3]}")
