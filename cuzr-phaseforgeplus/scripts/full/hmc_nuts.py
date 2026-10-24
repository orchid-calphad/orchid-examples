# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/hmc_nuts.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Gradient-informed HMC (NUTS) for the final PhaseForge+ Cu-Zr model (PLAN.md Amendment 5, post-hoc, PI-directed).
Protocol of Kunselman, Bocklund, Otis, Arroyave, Acta Mater. 315 (2026) 122412 (doi:10.1016/j.actamat.2026.122412):
ESPEI PR #268 likelihood gradients (environment: requirements-fit.txt), NUTS from C. Kunselman's littlemcmc fork
(github.com/cjkunselman18/littlemcmc @ 2917beed, "Implemented callback"), target = ESPEI lnL (improper uniform prior
p(theta) = 1), temperature-independent parameters in kJ (the TDB's VV values; = the paper's 1/1000 scaling), target
acceptance 0.8, max tree depth 6, diagonal mass matrix adapted from identity (littlemcmc QuadPotentialDiagAdapt,
as its adapt_diag init). The fork's callback (called after every NUTS iteration with a Draw record) is used to log
iterations and to stop a chain at its wall budget. No HMC driver is published in her repos; this driver is ours.
Modes:
  explore SID --wall S [--start-from RUN_DIR] [--init-var hessian]       NUTS with active mass-matrix/step-size tuning from fit start SID (fit_pr268.py starts:
                              0 = MLIP-only TDB, k > 0 = perturbed with seed k), until the wall budget; then SciPy BFGS
                              (jac=True) from the chain's best point (wall --bfgs-wall).
  tune --tune N               one chain from the optimum (--x0 result.json), N tuning iterations with adaptation;
                              writes the tuned diagonal mass matrix and step size.
  sample SEED --draws N       fixed mass matrix and step size (tuned step x --step-factor, default 0.5), N draws.
  test --tune N --draws M     short check (one chain).
Common: --model DIR (default runs/final/F10b_s1; data, TDB and symbols from its result.json), --out DIR."""
import json, glob, sys, time, warnings, os
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from scipy import optimize
from pycalphad import Database
from espei.datasets import load_datasets
from espei.error_functions.context import setup_context
from espei.utils import database_symbols_to_fit, unpack_piecewise
import littlemcmc as lmc
from littlemcmc.quadpotential import QuadPotentialDiagAdapt, QuadPotentialDiag

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, SGTE_TDB, orchid, DENSE_GRID, PFP_SRC  # noqa: F401 (paths: _paths.py)
def arg(k, d):
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
mode = sys.argv[1]
MODEL = ROOT / arg("--model", "runs/final/F10b_s1"); res = json.load(open(MODEL / "result.json"))
OUT = ROOT / arg("--out", "runs/hmc/" + mode + ("_" + sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else ""))
OUT.mkdir(parents=True, exist_ok=True)
TDB = Path(res["tdb_start"]); DATA = Path(res["data"])
db = Database(str(TDB)); syms = database_symbols_to_fit(db); assert syms == res["symbols"], (syms, res["symbols"])
xm = np.array([unpack_piecewise(db.symbols[s]) for s in syms], float)
meta = {d["symbol"]: d["parameter"] for d in json.load(open(ROOT / "tdb/mlip_fit.json"))["symbols"]}
meta.setdefault("VV0022", "G(CUZR_B2) b [J/mol/K]")
isT = np.array(["b [" in meta[s] for s in syms])
files = sorted(glob.glob(str(DATA / "*.json"))) + sorted(glob.glob(str(ROOT / "data/mlip_anchor/*.json")))
ctx = setup_context(db, load_datasets(files), syms); objs = ctx["residual_objs"]
N = len(syms)
t0 = time.time(); n = [0]; best = [-np.inf, None]
WALL = float(arg("--wall", 1e9))
ev = open(OUT / "evals.jsonl", "a")


class Stop(Exception): pass


def logp_dlogp(x):
    x = np.asarray(x, float); L, G = 0.0, np.zeros(N)
    for r in objs:
        l, g = r.get_likelihood_and_gradient(x)
        L += float(l); g = np.atleast_1d(np.asarray(g, float))
        if g.size == N: G += g
    n[0] += 1
    ok = bool(np.isfinite(L) and np.all(np.isfinite(G)))
    if ok and L > best[0]: best[0], best[1] = L, x.copy()
    ev.write(json.dumps({"eval": n[0], "t_s": round(time.time() - t0, 2), "lnL": L if ok else None,
                         "best": best[0] if np.isfinite(best[0]) else None}) + "\n"); ev.flush()
    return (L, G) if ok else (-np.inf, np.zeros(N))


it_log = open(OUT / "iters.jsonl", "a")
def callback(trace=None, draw=None):
    st = draw.stats[-1] if draw.stats else {}
    rec = {"iter": int(draw.draw_idx), "tuning": bool(draw.tuning), "t_s": round(time.time() - t0, 2), "n_eval": n[0],
           "best_lnL": best[0], "q": list(map(float, draw.point))}
    for k in ("tree_size", "depth", "mean_tree_accept", "energy", "diverging", "step_size", "step_size_bar", "model_logp", "max_energy_error"):
        if k in st:
            v = st[k]; rec[k] = bool(v) if isinstance(v, (bool, np.bool_)) else float(v)
    it_log.write(json.dumps(rec) + "\n"); it_log.flush()
    if time.time() - t0 > WALL: raise Stop()


def init_diag():
    """--init-var identity (littlemcmc adapt_diag default, the paper's) or hessian (1/|H_ii| of -lnL at the optimum,
    finite-difference step --hstep; the full Hessian is not positive definite, so its inverse is not used)."""
    if arg("--init-var", "identity") == "identity": return np.ones(N)
    h = json.load(open(ROOT / arg("--hessian", "runs/hmc/hessian/hessian.json")))["steps"][arg("--hstep", "0.003")]
    return np.clip(np.array(h["sd_diag_only"]) ** 2, 1e-8, None)


def nuts_step(x0, var=None, step=None):
    kw = dict(target_accept=0.8, max_treedepth=6, early_max_treedepth=6)
    if var is None:
        pot = QuadPotentialDiagAdapt(N, np.asarray(x0, float), init_diag(), 10, dtype="float64")
        return lmc.NUTS(logp_dlogp, N, potential=pot, **kw)
    pot = QuadPotentialDiag(np.asarray(var, float), dtype="float64")
    return lmc.NUTS(logp_dlogp, N, potential=pot, adapt_step_size=False, step_scale=step * N ** 0.25, **kw)


def run(step, x0, tune, draws, seed):
    try:
        tr, stats = lmc.sample(logp_dlogp, N, draws=draws, tune=tune, step=step, chains=1, cores=1, start=np.asarray(x0, float),
                               progressbar=False, random_seed=seed, discard_tuned_samples=False, callback=callback)
        return "finished"
    except Stop:
        return "wall-time budget reached"


def start_point(sid):
    if sid == 0: return xm.copy()
    rng = np.random.default_rng(sid)
    return np.where(isT, xm + rng.normal(0, 5.0, N), xm * (1 + 0.2 * rng.normal(0, 1, N)))


summary = {"mode": mode, "model": str(MODEL.relative_to(ROOT)), "tdb": str(TDB), "data": str(DATA), "symbols": syms,
           "littlemcmc": "cjkunselman18/littlemcmc@2917beed02f7ddee7258aa7b987e87c51391b584", "argv": sys.argv[1:],
           "settings": {"target_accept": 0.8, "max_treedepth": 6, "prior": "improper uniform p(theta)=1",
                        "parameterisation": "TDB VV values: constants kJ/mol (x1000 in TDB), T terms J/mol/K", "mass_matrix": "diagonal", "init_var": arg("--init-var", "identity")}}
if mode == "explore":
    sid = int(sys.argv[2])
    sf = arg("--start-from", None)   # a run directory: start at its L-BFGS-B optimum instead of fit start SID
    x0 = np.array(json.load(open(ROOT / sf / "result.json"))["x_best"], float) if sf else start_point(sid)
    step = nuts_step(x0)
    status = run(step, x0, tune=100000, draws=0, seed=1000 + sid)
    n_nuts, t_nuts = n[0], time.time() - t0; xb, Lb = best[1].copy(), best[0]
    summary.update({"start": sid, "start_from": sf, "x_start": x0.tolist(), "nuts_status": status, "nuts_n_eval": n_nuts, "nuts_wall_s": t_nuts,
                    "nuts_best_lnL": Lb, "nuts_best_x": xb.tolist()})
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    WALL = t_nuts + float(arg("--bfgs-wall", 3600))
    # BFGS with the wall check inside the objective (single evaluation per call)
    def f(z):
        L, G = logp_dlogp(z)
        if time.time() - t0 > WALL: raise Stop()
        return (-L, -G) if np.isfinite(L) else (1e12, np.zeros(N))
    try:
        r = optimize.minimize(f, xb, jac=True, method="BFGS"); bstat = f"{r.message} (nit {r.nit}, success {r.success})"
    except Stop:
        bstat = "BFGS wall-time budget reached"
    summary.update({"bfgs_status": bstat, "bfgs_n_eval": n[0] - n_nuts, "total_n_eval": n[0], "total_wall_s": time.time() - t0,
                    "final_best_lnL": best[0], "final_best_x": best[1].tolist()})
elif mode in ("tune", "test"):
    x0 = np.array(json.load(open(ROOT / arg("--x0", str(MODEL.relative_to(ROOT)) + "/result.json")))["x_best"], float)
    T = int(arg("--tune", 200)); D = int(arg("--draws", 0 if mode == "tune" else 100))
    step = nuts_step(x0); status = run(step, x0, tune=T, draws=D, seed=int(arg("--seed", 7)))
    summary.update({"x0": x0.tolist(), "tune": T, "draws": D, "status": status, "n_eval": n[0], "wall_s": time.time() - t0,
                    "tuned_var": np.asarray(step.potential._var, float).tolist(),
                    "step_size_bar": float(np.exp(step.step_adapt._log_bar)), "step_size_last": float(np.exp(step.step_adapt._log_step)),
                    "best_lnL": best[0], "best_x": best[1].tolist()})
elif mode == "sample":
    seed = int(sys.argv[2]); tj = json.load(open(ROOT / arg("--tuned", "runs/hmc/tune/summary.json")))
    fac = float(arg("--step-factor", 0.5)); eps = tj["step_size_bar"] * fac
    x0 = np.array(tj["x0"] if "--from-opt" in sys.argv else tj["last_q"], float)
    D = int(arg("--draws", 200)); step = nuts_step(x0, var=tj["tuned_var"], step=eps)
    status = run(step, x0, tune=0, draws=D, seed=seed)
    summary.update({"seed": seed, "step_size": eps, "step_factor": fac, "draws": D, "status": status, "n_eval": n[0], "wall_s": time.time() - t0})
if mode in ("tune", "test"):
    its = [json.loads(l) for l in open(OUT / "iters.jsonl")]
    summary["last_q"] = its[-1]["q"]
if mode in ("explore", "tune", "test", "sample"):
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
    print("done", mode, json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, default=float)[:1500])
