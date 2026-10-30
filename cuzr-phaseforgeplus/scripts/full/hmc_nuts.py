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
  explore SID --wall S [--start-from RUN_DIR] [--init-var hessian]       NUTS with active mass-matrix/step-size tuning
                              from fit start SID (fit_pr268.py starts: 0 = MLIP-only TDB, k > 0 = perturbed with seed k),
                              until the wall budget; then SciPy BFGS (jac=True) from the chain's best point
                              (wall --bfgs-wall).
  tune --tune N               one chain from the optimum (--x0 result.json), N tuning iterations with adaptation;
                              writes the tuned diagonal mass matrix and step size.
  sample SEED --draws N       fixed mass matrix and step size (tuned step x --step-factor, default 0.5), N draws.
  test --tune N --draws M     short check (one chain).
Common: --model DIR (default runs/final/F10b_s1; data, TDB and symbols from its result.json), --out DIR.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import littlemcmc as lmc
import numpy as np
from espei.datasets import load_datasets
from espei.error_functions.context import setup_context
from espei.utils import database_symbols_to_fit, unpack_piecewise
from littlemcmc.quadpotential import QuadPotentialDiag, QuadPotentialDiagAdapt
from pycalphad import Database
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import root

# Set after the imports (a statement between imports is flagged by ruff E402). Only warnings raised during the
# imports above can now reach stderr; no written output is affected.
warnings.filterwarnings("ignore")


def arg(k: str, d):
    """Value following a command-line flag, or a default.

    Args:
        k (str): Flag name, e.g. "--wall".
        d: Value returned when the flag is absent.

    Returns:
        The string after the flag, or ``d``.
    """
    return sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


mode = sys.argv[1]
model_dir = root / arg("--model", "runs/final/F10b_s1")
res = json.loads((model_dir / "result.json").read_text(encoding="utf-8"))
out_dir = root / arg(
    "--out", "runs/hmc/" + mode + ("_" + sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "")
)
out_dir.mkdir(parents=True, exist_ok=True)
tdb_path = Path(res["tdb_start"])
data_dir = Path(res["data"])
db = Database(str(tdb_path))
syms = database_symbols_to_fit(db)
assert syms == res["symbols"], (syms, res["symbols"])
xm = np.array([unpack_piecewise(db.symbols[s]) for s in syms], float)
meta = {d["symbol"]: d["parameter"] for d in json.loads((root / "tdb/mlip_fit.json").read_text(encoding="utf-8"))["symbols"]}
meta.setdefault("VV0022", "G(CUZR_B2) b [J/mol/K]")
is_t = np.array(["b [" in meta[s] for s in syms])
files = sorted(str(p) for p in data_dir.glob("*.json")) + sorted(str(p) for p in (root / "data/mlip_anchor").glob("*.json"))
ctx = setup_context(db, load_datasets(files), syms)
objs = ctx["residual_objs"]
n_par = len(syms)
t0 = time.time()
n = [0]
best = [-np.inf, None]
wall = float(arg("--wall", 1e9))
ev = (out_dir / "evals.jsonl").open("a", encoding="utf-8")


class WallBudgetError(Exception):
    """Raised inside the sampler or the BFGS objective when the wall-time budget is used up."""


def logp_dlogp(x):
    """Log-likelihood and its gradient of the fit at a parameter vector, logged to evals.jsonl.

    Args:
        x (array-like): Parameter vector, in the order of ``syms`` (kJ/mol for constants, J/mol/K for T terms).

    Returns:
        Tuple (lnL, gradient). Non-finite values are replaced by (-inf, zeros) so the sampler rejects the point.
    """
    x = np.asarray(x, float)
    lik, grad = 0.0, np.zeros(n_par)
    for r in objs:
        ll, g = r.get_likelihood_and_gradient(x)
        lik += float(ll)
        g = np.atleast_1d(np.asarray(g, float))
        if g.size == n_par:
            grad += g
    n[0] += 1
    ok = bool(np.isfinite(lik) and np.all(np.isfinite(grad)))
    if ok and lik > best[0]:
        best[0], best[1] = lik, x.copy()
    ev.write(
        json.dumps(
            {
                "eval": n[0],
                "t_s": round(time.time() - t0, 2),
                "lnL": lik if ok else None,
                "best": best[0] if np.isfinite(best[0]) else None,
            }
        )
        + "\n"
    )
    ev.flush()
    return (lik, grad) if ok else (-np.inf, np.zeros(n_par))


it_log = (out_dir / "iters.jsonl").open("a", encoding="utf-8")


def callback(trace=None, draw=None):
    """Per-iteration log of the NUTS sampler, and the wall-time stop.

    Args:
        trace: Unused; part of the littlemcmc callback signature.
        draw: littlemcmc Draw record (draw_idx, tuning, point, stats).

    Raises:
        WallBudgetError: When the wall-time budget is exceeded.
    """
    st = draw.stats[-1] if draw.stats else {}
    rec = {
        "iter": int(draw.draw_idx),
        "tuning": bool(draw.tuning),
        "t_s": round(time.time() - t0, 2),
        "n_eval": n[0],
        "best_lnL": best[0],
        "q": list(map(float, draw.point)),
    }
    for k in (
        "tree_size",
        "depth",
        "mean_tree_accept",
        "energy",
        "diverging",
        "step_size",
        "step_size_bar",
        "model_logp",
        "max_energy_error",
    ):
        if k in st:
            v = st[k]
            rec[k] = bool(v) if isinstance(v, (bool, np.bool_)) else float(v)
    it_log.write(json.dumps(rec) + "\n")
    it_log.flush()
    if time.time() - t0 > wall:
        raise WallBudgetError()


def init_diag():
    """Initial diagonal mass matrix: identity, or 1/|H_ii| from a finite-difference Hessian.

    The identity is littlemcmc's adapt_diag default (the paper's). With --init-var hessian, the variances are taken
    from the diagonal of -lnL at the optimum (finite-difference step --hstep); the full Hessian is not positive
    definite, so its inverse is not used.

    Returns:
        Array of length n_par with the diagonal of the mass matrix (variances).
    """
    if arg("--init-var", "identity") == "identity":
        return np.ones(n_par)
    h = json.loads((root / arg("--hessian", "runs/hmc/hessian/hessian.json")).read_text(encoding="utf-8"))["steps"][
        arg("--hstep", "0.003")
    ]
    return np.clip(np.array(h["sd_diag_only"]) ** 2, 1e-8, None)


def nuts_step(x0, var=None, step=None):
    """Build a littlemcmc NUTS step: adaptive diagonal mass matrix, or a fixed one with a fixed step size.

    Args:
        x0 (array-like): Starting point (used only for the adaptive mass matrix).
        var (array-like | None): Fixed diagonal mass matrix (variances); None to adapt it while tuning.
        step (float | None): Step size scale, used only when ``var`` is given.

    Returns:
        A littlemcmc NUTS step object.
    """
    kw = {"target_accept": 0.8, "max_treedepth": 6, "early_max_treedepth": 6}
    if var is None:
        pot = QuadPotentialDiagAdapt(n_par, np.asarray(x0, float), init_diag(), 10, dtype="float64")
        return lmc.NUTS(logp_dlogp, n_par, potential=pot, **kw)
    pot = QuadPotentialDiag(np.asarray(var, float), dtype="float64")
    return lmc.NUTS(logp_dlogp, n_par, potential=pot, adapt_step_size=False, step_scale=step * n_par**0.25, **kw)


def run(step, x0, tune, draws, seed):
    """Run one littlemcmc chain, stopping at the wall-time budget.

    Args:
        step: NUTS step object from ``nuts_step``.
        x0 (array-like): Starting point.
        tune (int): Number of tuning iterations.
        draws (int): Number of sampling iterations after tuning.
        seed (int): Random seed of the chain.

    Returns:
        "finished", or "wall-time budget reached" if the callback stopped the chain.
    """
    try:
        tr, stats = lmc.sample(
            logp_dlogp,
            n_par,
            draws=draws,
            tune=tune,
            step=step,
            chains=1,
            cores=1,
            start=np.asarray(x0, float),
            progressbar=False,
            random_seed=seed,
            discard_tuned_samples=False,
            callback=callback,
        )
        return "finished"
    except WallBudgetError:
        return "wall-time budget reached"


def start_point(sid):
    """Fit start point for a start id: the MLIP-only optimum (0) or a seeded perturbation of it.

    Args:
        sid (int): Start id. 0 returns the TDB values; k > 0 adds Normal(0, 5) to the T terms and a 20 % relative
            Normal(0, 1) change to the other parameters, drawn from seed k.

    Returns:
        Parameter vector of length n_par.
    """
    if sid == 0:
        return xm.copy()
    rng = np.random.default_rng(sid)
    return np.where(is_t, xm + rng.normal(0, 5.0, n_par), xm * (1 + 0.2 * rng.normal(0, 1, n_par)))


summary = {
    "mode": mode,
    "model": model_dir.relative_to(root).as_posix(),
    "tdb": str(tdb_path),
    "data": str(data_dir),
    "symbols": syms,
    "littlemcmc": "cjkunselman18/littlemcmc@2917beed02f7ddee7258aa7b987e87c51391b584",
    "argv": sys.argv[1:],
    "settings": {
        "target_accept": 0.8,
        "max_treedepth": 6,
        "prior": "improper uniform p(theta)=1",
        "parameterisation": "TDB VV values: constants kJ/mol (x1000 in TDB), T terms J/mol/K",
        "mass_matrix": "diagonal",
        "init_var": arg("--init-var", "identity"),
    },
}
if mode == "explore":
    sid = int(sys.argv[2])
    sf = arg("--start-from", None)  # a run directory: start at its L-BFGS-B optimum instead of fit start SID
    x0 = (
        np.array(json.loads((root / sf / "result.json").read_text(encoding="utf-8"))["x_best"], float) if sf else start_point(sid)
    )
    step = nuts_step(x0)
    status = run(step, x0, tune=100000, draws=0, seed=1000 + sid)
    n_nuts, t_nuts = n[0], time.time() - t0
    xb, lb = best[1].copy(), best[0]
    summary.update(
        {
            "start": sid,
            "start_from": sf,
            "x_start": x0.tolist(),
            "nuts_status": status,
            "nuts_n_eval": n_nuts,
            "nuts_wall_s": t_nuts,
            "nuts_best_lnL": lb,
            "nuts_best_x": xb.tolist(),
        }
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    wall = t_nuts + float(arg("--bfgs-wall", 3600))

    # BFGS with the wall check inside the objective (single evaluation per call)
    def f(z):
        """BFGS objective: negative lnL and its negative gradient, with the wall-time check.

        Args:
            z (array-like): Parameter vector.

        Returns:
            Tuple (-lnL, -gradient), or (1e12, zeros) where lnL is not finite.

        Raises:
            WallBudgetError: When the wall-time budget is exceeded.
        """
        lik, grad = logp_dlogp(z)
        if time.time() - t0 > wall:
            raise WallBudgetError()
        return (-lik, -grad) if np.isfinite(lik) else (1e12, np.zeros(n_par))

    try:
        r = optimize.minimize(f, xb, jac=True, method="BFGS")
        bstat = f"{r.message} (nit {r.nit}, success {r.success})"
    except WallBudgetError:
        bstat = "BFGS wall-time budget reached"
    summary.update(
        {
            "bfgs_status": bstat,
            "bfgs_n_eval": n[0] - n_nuts,
            "total_n_eval": n[0],
            "total_wall_s": time.time() - t0,
            "final_best_lnL": best[0],
            "final_best_x": best[1].tolist(),
        }
    )
elif mode in ("tune", "test"):
    x0 = np.array(
        json.loads((root / arg("--x0", str(model_dir.relative_to(root)) + "/result.json")).read_text(encoding="utf-8"))["x_best"],
        float,
    )
    n_tune = int(arg("--tune", 200))
    n_draws = int(arg("--draws", 0 if mode == "tune" else 100))
    step = nuts_step(x0)
    status = run(step, x0, tune=n_tune, draws=n_draws, seed=int(arg("--seed", 7)))
    summary.update(
        {
            "x0": x0.tolist(),
            "tune": n_tune,
            "draws": n_draws,
            "status": status,
            "n_eval": n[0],
            "wall_s": time.time() - t0,
            "tuned_var": np.asarray(step.potential._var, float).tolist(),
            "step_size_bar": float(np.exp(step.step_adapt._log_bar)),
            "step_size_last": float(np.exp(step.step_adapt._log_step)),
            "best_lnL": best[0],
            "best_x": best[1].tolist(),
        }
    )
elif mode == "sample":
    seed = int(sys.argv[2])
    tj = json.loads((root / arg("--tuned", "runs/hmc/tune/summary.json")).read_text(encoding="utf-8"))
    fac = float(arg("--step-factor", 0.5))
    eps = tj["step_size_bar"] * fac
    x0 = np.array(tj["x0"] if "--from-opt" in sys.argv else tj["last_q"], float)
    n_draws = int(arg("--draws", 200))
    step = nuts_step(x0, var=tj["tuned_var"], step=eps)
    status = run(step, x0, tune=0, draws=n_draws, seed=seed)
    summary.update(
        {
            "seed": seed,
            "step_size": eps,
            "step_factor": fac,
            "draws": n_draws,
            "status": status,
            "n_eval": n[0],
            "wall_s": time.time() - t0,
        }
    )
if mode in ("tune", "test"):
    with (out_dir / "iters.jsonl").open(encoding="utf-8") as fh:
        its = [json.loads(line) for line in fh]
    summary["last_q"] = its[-1]["q"]
if mode in ("explore", "tune", "test", "sample"):
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print("done", mode, json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, default=float)[:1500])
