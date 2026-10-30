# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/hmc_hessian.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Curvature of -lnL at the optimum (Amendment 5), by central finite differences.

The differences are of the ESPEI PR #268 analytic gradient, with steps h_i = c * s_i with s_i the fit's L-BFGS-B scale (5 J/mol/K for T terms, 10 kJ/mol x atoms per formula for
compound constants, 10 kJ/mol for interaction constants), c = 1e-3 and 3e-3. Writes runs/hmc/hessian/hessian.json
(symmetrised H per c, Laplace covariance = pinv(H) when H is positive definite, eigenvalues).
Usage: hmc_hessian.py [--model runs/final/F10b_s1]
"""

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_hmc_nuts():
    """Import hmc_nuts in its 'hessian' mode, which runs nothing heavy but builds the ESPEI context.

    Returns:
        The hmc_nuts module, imported with sys.argv set to its hessian-mode arguments.
    """
    sys.argv = [sys.argv[0], "hessian"] + sys.argv[1:] + ["--out", "runs/hmc/hessian"]
    return importlib.import_module("hmc_nuts")


nuts = _load_hmc_nuts()
t_scale = 5.0  # J/mol/K: L-BFGS-B scale of the temperature terms
g_scale = 10.0  # kJ/mol: L-BFGS-B scale of the constants (times atoms per formula unit for formula constants)
x0 = np.array(nuts.res["x_best"], float)
syms = nuts.syms
natom = {"CU5ZR_C15B": 6, "CU51ZR14": 65, "CU8ZR3": 11, "CU10ZR7": 17, "CUZR_B2": 2, "CUZR2": 3}
scale = np.array(
    [
        t_scale
        if nuts.is_t[i]
        else (g_scale * natom[nuts.meta[s].split("(")[1].split(")")[0]] if "formula" in nuts.meta[s] else g_scale)
        for i, s in enumerate(syms)
    ]
)
lnl0, g0 = nuts.logp_dlogp(x0)
out = {"x0": x0.tolist(), "lnL0": lnl0, "grad0": g0.tolist(), "scale": scale.tolist(), "steps": {}}
for c in (1e-3, 3e-3):
    h_raw = np.zeros((nuts.n_par, nuts.n_par))
    for i in range(nuts.n_par):
        e = np.zeros(nuts.n_par)
        e[i] = c * scale[i]
        gp = nuts.logp_dlogp(x0 + e)[1]
        gm = nuts.logp_dlogp(x0 - e)[1]
        h_raw[:, i] = -(gp - gm) / (2 * e[i])
    h_sym = 0.5 * (h_raw + h_raw.T)
    w = np.linalg.eigvalsh(h_sym)
    rec = {
        "H": h_sym.tolist(),
        "asym_rel": float(np.linalg.norm(h_raw - h_raw.T) / np.linalg.norm(h_raw)),
        "eig": w.tolist(),
        "pd": bool(w.min() > 0),
    }
    rec["cov"] = np.linalg.pinv(h_sym).tolist()
    rec["sd"] = np.sqrt(np.abs(np.diag(np.linalg.pinv(h_sym)))).tolist()
    rec["sd_diag_only"] = (1 / np.sqrt(np.abs(np.diag(h_sym)))).tolist()
    out["steps"][str(c)] = rec
out["n_eval"] = nuts.n[0]
out["wall_s"] = time.time() - nuts.t0
(nuts.root / "runs/hmc/hessian/hessian.json").write_text(json.dumps(out, indent=1))
print("done", {c: (r["pd"], min(r["eig"]), max(r["eig"]), r["asym_rel"]) for c, r in out["steps"].items()})
