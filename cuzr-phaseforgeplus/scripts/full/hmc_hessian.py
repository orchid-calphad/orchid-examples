# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/hmc_hessian.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Curvature of -lnL at the optimum (Amendment 5): central finite differences of the ESPEI PR #268 analytic gradient,
steps h_i = c * s_i with s_i the fit's L-BFGS-B scale (5 J/mol/K for T terms, 10 kJ/mol x atoms per formula for
compound constants, 10 kJ/mol for interaction constants), c = 1e-3 and 3e-3. Writes runs/hmc/hessian/hessian.json
(symmetrised H per c, Laplace covariance = pinv(H) when H is positive definite, eigenvalues).
Usage: hmc_hessian.py [--model runs/final/F10b_s1]"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, SGTE_TDB, orchid  # noqa: E402,F401
import json, sys, time
from pathlib import Path
import numpy as np
sys.argv = [sys.argv[0], "hessian"] + sys.argv[1:] + ["--out", "runs/hmc/hessian"]
sys.path.insert(0, str(Path(__file__).parent))
import hmc_nuts as H   # noqa: E402  (runs nothing heavy for an unknown mode; builds the context)
x0 = np.array(H.res["x_best"], float); syms = H.syms
natom = {"CU5ZR_C15B": 6, "CU51ZR14": 65, "CU8ZR3": 11, "CU10ZR7": 17, "CUZR_B2": 2, "CUZR2": 3}
scale = np.array([5.0 if H.isT[i] else (10.0 * natom[H.meta[s].split("(")[1].split(")")[0]] if "formula" in H.meta[s] else 10.0) for i, s in enumerate(syms)])
L0, g0 = H.logp_dlogp(x0); out = {"x0": x0.tolist(), "lnL0": L0, "grad0": g0.tolist(), "scale": scale.tolist(), "steps": {}}
for c in (1e-3, 3e-3):
    Hm = np.zeros((H.N, H.N))
    for i in range(H.N):
        e = np.zeros(H.N); e[i] = c * scale[i]
        gp = H.logp_dlogp(x0 + e)[1]; gm = H.logp_dlogp(x0 - e)[1]
        Hm[:, i] = -(gp - gm) / (2 * e[i])
    Hs = 0.5 * (Hm + Hm.T); w = np.linalg.eigvalsh(Hs)
    rec = {"H": Hs.tolist(), "asym_rel": float(np.linalg.norm(Hm - Hm.T) / np.linalg.norm(Hm)), "eig": w.tolist(), "pd": bool(w.min() > 0)}
    rec["cov"] = np.linalg.pinv(Hs).tolist(); rec["sd"] = np.sqrt(np.abs(np.diag(np.linalg.pinv(Hs)))).tolist()
    rec["sd_diag_only"] = (1 / np.sqrt(np.abs(np.diag(Hs)))).tolist()
    out["steps"][str(c)] = rec
out["n_eval"] = H.n[0]; out["wall_s"] = time.time() - H.t0
json.dump(out, open(H.ROOT / "runs/hmc/hessian/hessian.json", "w"), indent=1)
print("done", {c: (r["pd"], min(r["eig"]), max(r["eig"]), r["asym_rel"]) for c, r in out["steps"].items()})
