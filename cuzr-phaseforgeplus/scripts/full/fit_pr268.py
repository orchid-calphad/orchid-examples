# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/fit_pr268.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Post-hoc rerun of the PhaseForge+ tuning with ESPEI PR #268 (PLAN.md amendment 3).

ESPEI PR #268 head 9039ea2 + pycalphad develop 9a6939d (environment: requirements-fit.txt). Same data (data/fit +
data/mlip_anchor), same 22 parameters, paper configuration: scipy CG (default hyperparameters, jac=True) on the TDB's
VV values (constants in kJ = the paper's x1000 preconditioning, T terms in J/K); if CG stops without its own
convergence (precision loss / iteration limit), L-BFGS-B on scaled parameters from the CG best, repeated until scipy
reports CONVERGENCE (ftol or projected-gradient criterion) or the wall budget ends.
Usage: fit_pr268.py START_ID (0 = MLIP start, 1-4 = perturbed, seed = id).
Perturbation: constant terms x (1 + 0.2 N(0,1)), T terms + N(0, 5 J/mol/K). Output runs/pr268/s<ID>/.
"""

from __future__ import annotations

import json
import re
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np
from espei.datasets import load_datasets
from espei.error_functions.context import setup_context
from espei.utils import database_symbols_to_fit, unpack_piecewise
from pycalphad import Database
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import root

# Set after the imports (a statement between imports is flagged by ruff E402). Only warnings raised during the
# imports above can now reach stderr; no written output is affected.
warnings.filterwarnings("ignore")


class WallBudgetError(Exception):
    """Raised inside the objective when the wall-time budget is used up."""


def arg(k: str, d):
    """Value following a command-line flag, or a default.

    Args:
        k (str): Flag name, e.g. "--wall".
        d: Value returned when the flag is absent.

    Returns:
        The string after the flag, or ``d``.
    """
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


sid = int(sys.argv[1])
wall = float(arg("--wall", 2.5 * 3600))
out_dir = root / arg("--out", f"runs/pr268/s{sid}")
out_dir.mkdir(parents=True, exist_ok=True)
tdb_path = root / arg("--tdb", "tdb/CuZr-PhaseForge-MLIP.tdb")
data_dir = root / arg("--data", "data/fit")
db = Database(str(tdb_path))
syms = database_symbols_to_fit(db)
xm = np.array([unpack_piecewise(db.symbols[s]) for s in syms], float)
meta = {d["symbol"]: d["parameter"] for d in json.loads((root / "tdb/mlip_fit.json").read_text())["symbols"]}
meta.update(json.loads(arg("--extra-meta", "{}")))
is_t = np.array(["b [" in meta[s] for s in syms])
x0 = xm.copy()
if sid > 0:
    rng = np.random.default_rng(sid)
    x0 = np.where(is_t, xm + rng.normal(0, 5.0, len(xm)), xm * (1 + 0.2 * rng.normal(0, 1, len(xm))))
files = sorted(str(p) for p in data_dir.glob("*.json")) + sorted(str(p) for p in (root / "data/mlip_anchor").glob("*.json"))
ctx = setup_context(db, load_datasets(files), syms)
objs = ctx["residual_objs"]
t_scale = 5.0  # J/mol/K: L-BFGS-B scale of the temperature terms
g_scale = 10.0  # kJ/mol: L-BFGS-B scale of the constants (times atoms per formula unit for formula constants)
natom = {"CU5ZR_C15B": 6, "CU51ZR14": 65, "CU8ZR3": 11, "CU10ZR7": 17, "CUZR_B2": 2, "CUZR2": 3}
scale = np.array(
    [
        t_scale if is_t[i] else (g_scale * natom[meta[s].split("(")[1].split(")")[0]] if "formula" in meta[s] else g_scale)
        for i, s in enumerate(syms)
    ]
)
tr = (out_dir / "trace.jsonl").open("w")
t0 = time.time()
n = [0]
best = [np.inf, x0.copy()]
stage = ["CG"]


def lnl_grad(x):
    """Total log-likelihood, its gradient and the per-residual log-likelihoods at a parameter vector.

    Args:
        x (array-like): Parameter vector, in the order of ``syms``.

    Returns:
        Tuple (lnL, gradient, dict of residual class name -> log-likelihood).
    """
    lik, grad, per = 0.0, np.zeros(len(x)), {}
    for r in objs:
        ll, g = r.get_likelihood_and_gradient(np.asarray(x, float))
        per[type(r).__name__] = float(ll)
        lik += float(ll)
        g = np.atleast_1d(np.asarray(g, float))
        if g.size == len(x):
            grad += g
    return lik, grad, per


def f_raw(x):
    """Objective for the optimiser: negative lnL and its negative gradient, logged to trace.jsonl.

    Args:
        x (array-like): Parameter vector.

    Returns:
        Tuple (-lnL, -gradient), or (1e12, zeros) where the likelihood is not finite.

    Raises:
        WallBudgetError: When the wall-time budget is exceeded.
    """
    lik, grad, per = lnl_grad(x)
    n[0] += 1
    ok = np.isfinite(lik) and np.all(np.isfinite(grad))
    tr.write(
        json.dumps(
            {
                "eval": n[0],
                "stage": stage[0],
                "t_s": round(time.time() - t0, 2),
                "lnL": lik if np.isfinite(lik) else None,
                "per": per,
                "grad_norm": float(np.linalg.norm(grad)) if ok else None,
                "x": list(map(float, x)),
            }
        )
        + "\n"
    )
    tr.flush()
    if ok and -lik < best[0]:
        best[0], best[1] = -lik, np.array(x, float)
    if time.time() - t0 > wall:
        raise WallBudgetError()
    return (-lik, -grad) if ok else (1e12, np.zeros_like(x))


def scaled_f(z, xs):
    """Objective in the scaled variables z, around the point xs (L-BFGS-B stages).

    Args:
        z (array-like): Scaled parameter offsets.
        xs (np.ndarray): Parameter vector the offsets are relative to.

    Returns:
        Tuple (objective value, gradient with respect to z), as returned by ``f_raw``.
    """
    fv = f_raw(xs + scale * z)
    return fv[0], fv[1] * scale


log = []
try:
    r = optimize.minimize(f_raw, x0, jac=True, method="CG", options={"maxiter": 200})
    log.append({"stage": "CG", "message": str(r.message), "nit": int(r.nit), "success": bool(r.success)})
    conv = bool(r.success)
    k = 0
    while not conv:
        k += 1
        stage[0] = f"L-BFGS-B {k}"
        xs = best[1].copy()
        r = optimize.minimize(partial(scaled_f, xs=xs), np.zeros(len(xs)), jac=True, method="L-BFGS-B", options={"maxiter": 500})
        log.append({"stage": stage[0], "message": str(r.message), "nit": int(r.nit), "success": bool(r.success)})
        conv = bool(r.success) and "CONVERGENCE" in str(r.message)
        if k >= 5:
            break
    final = "converged: " + log[-1]["message"] if conv else "not converged after 5 L-BFGS-B restarts"
except WallBudgetError:
    final = "wall-time budget reached"
txt = tdb_path.read_text()
for s, v in zip(syms, best[1], strict=True):
    txt = re.sub(rf"(FUNCTION\s+{s}\s+298\.15\s+)[-+.\dEe]+;", rf"\g<1>{v:.6f};", txt)
(out_dir / "tuned.tdb").write_text(f"$ PhaseForge+ rerun with ESPEI PR #268 9039ea2 (start {sid}); demonstration only\n" + txt)
lik, grad, per = lnl_grad(best[1])
(out_dir / "result.json").write_text(
    json.dumps(
        {
            "start": sid,
            "data": str(data_dir),
            "tdb_start": str(tdb_path),
            "symbols": syms,
            "x_start": x0.tolist(),
            "x_best": best[1].tolist(),
            "lnL_start": None,
            "lnL_best": lik,
            "per_residual": per,
            "grad_norm_best": float(np.linalg.norm(grad)),
            "stages": log,
            "final": final,
            "n_eval": n[0],
            "wall_s": time.time() - t0,
        },
        indent=1,
    )
)
print("done", sid, final, lik, n[0], time.time() - t0)
