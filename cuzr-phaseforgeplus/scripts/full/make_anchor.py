# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/make_anchor.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""MLIP constraint datasets (paper configuration, PLAN.md amendment 1): mixing enthalpies of each solution phase
(FCC_A1, BCC_A2, HCP_A3, LIQUID) computed from the PhaseForge (MLIP-only) model across composition at 298.15 K
(x_Zr = 0.1 ... 0.9), and the MLIP formation enthalpy of each compound (298.15 K; 0 K value), as ESPEI
non-equilibrium thermochemical data. Every datum has sigma_MLIP (results/sigma_mlip.json): ESPEI weight
= 500 J/mol / sigma_MLIP. Experimental formation enthalpies in data/fit keep their own sigma; the likelihood weights
both. Writes data/mlip_anchor/*.json."""
import json
from pathlib import Path
import numpy as np
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, SGTE_TDB, orchid, DENSE_GRID, PFP_SRC  # noqa: F401 (paths: _paths.py)
F = json.load(open(ROOT / "tdb/mlip_fit.json"))
S = json.load(open(ROOT / "results/sigma_mlip.json"))
W = S["espei_weight"]
COMP = {"CU5ZR_C15B": (5, 1), "CU51ZR14": (51, 14), "CU8ZR3": (8, 3), "CU10ZR7": (10, 7), "CUZR_B2": (1, 1), "CUZR2": (1, 2)}
XS = [round(x, 2) for x in np.arange(0.1, 0.91, 0.1)]
out = ROOT / "data/mlip_anchor"; out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*.json"): f.unlink()
for lat, f in F["lattices"].items():
    liq = lat == "LIQUID"
    vals = [x * (1 - x) * (f["L0"] + f["L1"] * ((1 - x) - x)) for x in XS]   # H_mix of the MLIP-only model (b = 0)
    d = {"components": ["CU", "ZR"] + ([] if liq else ["VA"]), "phases": [lat],
         "solver": {"mode": "manual", "sublattice_site_ratios": [1] if liq else {"FCC_A1": [1, 1], "BCC_A2": [1, 3], "HCP_A3": [1, 0.5]}[lat],
                    "sublattice_configurations": [[["CU", "ZR"]] if liq else [["CU", "ZR"], "VA"] for _ in XS],
                    "sublattice_occupancies": [[[1 - x, x]] if liq else [[1 - x, x], 1.0] for x in XS]},
         "conditions": {"P": 101325, "T": 298.15}, "output": "HM_MIX", "values": [[vals]],
         "reference": "PhaseForge MLIP-only model (GRACE-2L-OMAT; this work)", "weight": W,
         "comment": "sigma_MLIP = %.2f kJ/mol-atom; J/mol-atom" % S["sigma_MLIP_kJ"]}
    json.dump(d, open(out / f"mlip_HM_MIX_{lat}.json", "w"), indent=1)
for ph, c in F["compounds"].items():
    a, b = COMP[ph]
    d = {"components": ["CU", "ZR"], "phases": [ph],
         "solver": {"mode": "manual", "sublattice_site_ratios": [a, b], "sublattice_configurations": [["CU", "ZR"]]},
         "conditions": {"P": 101325, "T": 298.15}, "output": "HM_FORM", "values": [[[c["dHf_J_mol_atom"]]]],
         "reference": "PhaseForge MLIP (GRACE-2L-OMAT; this work)", "weight": W,
         "comment": "sigma_MLIP = %.2f kJ/mol-atom" % S["sigma_MLIP_kJ"]}
    json.dump(d, open(out / f"mlip_HM_FORM_{ph}.json", "w"), indent=1)
print(W, sorted(p.name for p in out.glob("*.json")))
