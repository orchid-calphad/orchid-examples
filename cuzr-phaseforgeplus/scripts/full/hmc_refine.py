# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/hmc_refine.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Amendment 5 (a), extra refinement step (disclosed): SciPy BFGS from each NUTS chain's best point stopped at its first
iteration ("precision loss": line search fails on the piecewise ZPF likelihood), so each chain's best point is refined
with the fit's own optimizer instead: L-BFGS-B on the fit's scaled parameters (fit_pr268.py), maxiter 500, until
scipy's own convergence or the wall budget. Usage: hmc_refine.py RUN (e.g. explore_3) --model runs/final2/F10b_s3 [--wall 2400]
Writes runs/hmc/<RUN>/refine.json."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, SGTE_TDB, orchid  # noqa: E402,F401
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import optimize
run = sys.argv[1]; rest = sys.argv[2:]
sys.argv = [sys.argv[0], "refine"] + rest + ["--out", f"runs/hmc/{run}/refine"]
sys.path.insert(0, str(Path(__file__).parent))
import hmc_nuts as H   # builds the ESPEI context for --model
s = json.load(open(H.ROOT / f"runs/hmc/{run}/summary.json")); xs = np.array(s["final_best_x"], float)
natom = {"CU5ZR_C15B": 6, "CU51ZR14": 65, "CU8ZR3": 11, "CU10ZR7": 17, "CUZR_B2": 2, "CUZR2": 3}
scale = np.array([5.0 if H.isT[i] else (10.0 * natom[H.meta[k].split("(")[1].split(")")[0]] if "formula" in H.meta[k] else 10.0) for i, k in enumerate(H.syms)])
WALL = float(H.arg("--wall", 2400))
def f(z):
    L, G = H.logp_dlogp(xs + scale * z)
    if time.time() - H.t0 > WALL: raise H.Stop()
    return (-L, -G * scale) if np.isfinite(L) else (1e12, np.zeros(H.N))
try:
    r = optimize.minimize(f, np.zeros(H.N), jac=True, method="L-BFGS-B", options={"maxiter": 500}); st = f"{r.message} (nit {r.nit})"
except H.Stop:
    st = "wall-time budget reached"
out = {"run": run, "start_lnL": s["final_best_lnL"], "status": st, "n_eval": H.n[0], "wall_s": time.time() - H.t0,
       "best_lnL": H.best[0], "best_x": H.best[1].tolist()}
json.dump(out, open(H.ROOT / f"runs/hmc/{run}/refine.json", "w"), indent=1)
print("done", run, st, out["start_lnL"], out["best_lnL"], out["n_eval"])
