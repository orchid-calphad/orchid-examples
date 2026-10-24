# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/fit_pr268.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Post-hoc rerun (PLAN.md amendment 3) of the PhaseForge+ tuning with ESPEI PR #268 head 9039ea2 + pycalphad develop
9a6939d (environment: requirements-fit.txt). Same data (data/fit + data/mlip_anchor), same 22 parameters, paper configuration:
scipy CG (default hyperparameters, jac=True) on the TDB's VV values (constants in kJ = the paper's x1000
preconditioning, T terms in J/K); if CG stops without its own convergence (precision loss / iteration limit),
L-BFGS-B on scaled parameters from the CG best, repeated until scipy reports CONVERGENCE (ftol or projected-gradient
criterion) or the wall budget ends. Usage: fit_pr268.py START_ID (0 = MLIP start, 1-4 = perturbed, seed = id).
Perturbation: constant terms x (1 + 0.2 N(0,1)), T terms + N(0, 5 J/mol/K). Output runs/pr268/s<ID>/."""
import json, glob, sys, time, re, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from scipy import optimize
from pycalphad import Database
from espei.datasets import load_datasets
from espei.error_functions.context import setup_context
from espei.utils import database_symbols_to_fit, unpack_piecewise
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, SGTE_TDB, orchid, DENSE_GRID, PFP_SRC  # noqa: F401 (paths: _paths.py)
def arg(k, d):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
sid = int(sys.argv[1]); WALL = float(arg("--wall", 2.5 * 3600))
OUT = ROOT / arg("--out", f"runs/pr268/s{sid}"); OUT.mkdir(parents=True, exist_ok=True)
TDB = ROOT / arg("--tdb", "tdb/CuZr-PhaseForge-MLIP.tdb")
DATA = ROOT / arg("--data", "data/fit")
db = Database(str(TDB)); syms = database_symbols_to_fit(db)
xm = np.array([unpack_piecewise(db.symbols[s]) for s in syms], float)
meta = {d["symbol"]: d["parameter"] for d in json.load(open(ROOT / "tdb/mlip_fit.json"))["symbols"]}
meta.update(json.loads(arg("--extra-meta", "{}")))
isT = np.array(["b [" in meta[s] for s in syms])
x0 = xm.copy()
if sid > 0:
    rng = np.random.default_rng(sid)
    x0 = np.where(isT, xm + rng.normal(0, 5.0, len(xm)), xm * (1 + 0.2 * rng.normal(0, 1, len(xm))))
files = sorted(glob.glob(str(DATA / "*.json"))) + sorted(glob.glob(str(ROOT / "data/mlip_anchor/*.json")))
ctx = setup_context(db, load_datasets(files), syms); objs = ctx["residual_objs"]
natom = {"CU5ZR_C15B": 6, "CU51ZR14": 65, "CU8ZR3": 11, "CU10ZR7": 17, "CUZR_B2": 2, "CUZR2": 3}
scale = np.array([5.0 if isT[i] else (10.0 * natom[meta[s].split("(")[1].split(")")[0]] if "formula" in meta[s] else 10.0) for i, s in enumerate(syms)])
tr = open(OUT / "trace.jsonl", "w"); t0 = time.time(); n = [0]; best = [np.inf, x0.copy()]; stage = ["CG"]


class Stop(Exception): pass


def lnl_grad(x):
    L, G, per = 0.0, np.zeros(len(x)), {}
    for r in objs:
        l, g = r.get_likelihood_and_gradient(np.asarray(x, float))
        per[type(r).__name__] = float(l); L += float(l)
        g = np.atleast_1d(np.asarray(g, float))
        if g.size == len(x): G += g
    return L, G, per


def f_raw(x):
    L, G, per = lnl_grad(x); n[0] += 1
    ok = np.isfinite(L) and np.all(np.isfinite(G))
    tr.write(json.dumps({"eval": n[0], "stage": stage[0], "t_s": round(time.time() - t0, 2), "lnL": L if np.isfinite(L) else None, "per": per,
                         "grad_norm": float(np.linalg.norm(G)) if ok else None, "x": list(map(float, x))}) + "\n"); tr.flush()
    if ok and -L < best[0]: best[0], best[1] = -L, np.array(x, float)
    if time.time() - t0 > WALL: raise Stop()
    return (-L, -G) if ok else (1e12, np.zeros_like(x))


log = []
try:
    r = optimize.minimize(f_raw, x0, jac=True, method="CG", options={"maxiter": 200})
    log.append({"stage": "CG", "message": str(r.message), "nit": int(r.nit), "success": bool(r.success)})
    conv = bool(r.success)
    k = 0
    while not conv:
        k += 1; stage[0] = f"L-BFGS-B {k}"; xs = best[1].copy()
        g = lambda z: (lambda fv: (fv[0], fv[1] * scale))(f_raw(xs + scale * z))
        r = optimize.minimize(g, np.zeros(len(xs)), jac=True, method="L-BFGS-B", options={"maxiter": 500})
        log.append({"stage": stage[0], "message": str(r.message), "nit": int(r.nit), "success": bool(r.success)})
        conv = bool(r.success) and "CONVERGENCE" in str(r.message)
        if k >= 5: break
    final = "converged: " + log[-1]["message"] if conv else "not converged after 5 L-BFGS-B restarts"
except Stop:
    final = "wall-time budget reached"
txt = TDB.read_text()
for s, v in zip(syms, best[1]):
    txt = re.sub(r"(FUNCTION\s+%s\s+298\.15\s+)[-+.\dEe]+;" % s, lambda mm: mm.group(1) + "%.6f;" % v, txt)
(OUT / "tuned.tdb").write_text(f"$ PhaseForge+ rerun with ESPEI PR #268 9039ea2 (start {sid}); demonstration only\n" + txt)
L, G, per = lnl_grad(best[1])
json.dump({"start": sid, "data": str(DATA), "tdb_start": str(TDB), "symbols": syms, "x_start": x0.tolist(), "x_best": best[1].tolist(), "lnL_start": None, "lnL_best": L, "per_residual": per,
           "grad_norm_best": float(np.linalg.norm(G)), "stages": log, "final": final, "n_eval": n[0], "wall_s": time.time() - t0},
          open(OUT / "result.json", "w"), indent=1)
print("done", sid, final, L, n[0], time.time() - t0)
