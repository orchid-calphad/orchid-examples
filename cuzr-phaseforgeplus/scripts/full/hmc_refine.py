# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/hmc_refine.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Amendment 5 (a): extra refinement step (disclosed), from each NUTS chain's best point.

SciPy BFGS stopped at its first iteration ("precision loss": line search fails on the piecewise ZPF likelihood), so each chain's best point is refined
with the fit's own optimizer instead: L-BFGS-B on the fit's scaled parameters (fit_pr268.py), maxiter 500, until
scipy's own convergence or the wall budget. Usage: hmc_refine.py RUN (e.g. explore_3) --model runs/final2/F10b_s3 [--wall 2400]
Writes runs/hmc/<RUN>/refine.json.
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_hmc_nuts(run_name: str):
    """Import hmc_nuts in its 'refine' mode for one NUTS run, which builds the ESPEI context for --model.

    Args:
        run_name (str): Name of the NUTS run directory under runs/hmc (e.g. explore_3).

    Returns:
        The hmc_nuts module, imported with sys.argv set to its refine-mode arguments.
    """
    rest = sys.argv[2:]
    sys.argv = [sys.argv[0], "refine"] + rest + ["--out", f"runs/hmc/{run_name}/refine"]
    return importlib.import_module("hmc_nuts")


run = sys.argv[1]
nuts = _load_hmc_nuts(run)
t_scale = 5.0  # J/mol/K: L-BFGS-B scale of the temperature terms
g_scale = 10.0  # kJ/mol: L-BFGS-B scale of the constants (times atoms per formula unit for formula constants)

s = json.loads((nuts.root / f"runs/hmc/{run}/summary.json").read_text())
xs = np.array(s["final_best_x"], float)
natom = {"CU5ZR_C15B": 6, "CU51ZR14": 65, "CU8ZR3": 11, "CU10ZR7": 17, "CUZR_B2": 2, "CUZR2": 3}
scale = np.array(
    [
        t_scale
        if nuts.is_t[i]
        else (g_scale * natom[nuts.meta[k].split("(")[1].split(")")[0]] if "formula" in nuts.meta[k] else g_scale)
        for i, k in enumerate(nuts.syms)
    ]
)
wall = float(nuts.arg("--wall", 2400))


def f(z):
    """BFGS objective in the scaled variables: negative lnL and its negative gradient, with the wall-time check.

    Args:
        z (array-like): Scaled parameter offsets from the NUTS run's best point.

    Returns:
        Tuple (-lnL, -gradient * scale), or (1e12, zeros) where lnL is not finite.

    Raises:
        WallBudgetError: When the wall-time budget is exceeded.
    """
    lik, grad = nuts.logp_dlogp(xs + scale * z)
    if time.time() - nuts.t0 > wall:
        raise nuts.WallBudgetError()
    return (-lik, -grad * scale) if np.isfinite(lik) else (1e12, np.zeros(nuts.n_par))


try:
    r = optimize.minimize(f, np.zeros(nuts.n_par), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    st = f"{r.message} (nit {r.nit})"
except nuts.WallBudgetError:
    st = "wall-time budget reached"
out = {
    "run": run,
    "start_lnL": s["final_best_lnL"],
    "status": st,
    "n_eval": nuts.n[0],
    "wall_s": time.time() - nuts.t0,
    "best_lnL": nuts.best[0],
    "best_x": nuts.best[1].tolist(),
}
(nuts.root / f"runs/hmc/{run}/refine.json").write_text(json.dumps(out, indent=1))
print("done", run, st, out["start_lnL"], out["best_lnL"], out["n_eval"])
