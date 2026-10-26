# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/sigma_mlip.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""sigma_MLIP = sqrt(sigma_DFT^2 + sigma_MLIP->DFT^2) (PI instruction D, PLAN.md amendment 1).

sigma_DFT: ORCHID's calibrated DFT-vs-experiment robust sigma, espei/dft/CALIBRATION.json, source Materials Project
  (PBE, the functional family of OMat24 on which GRACE-2L-OMAT is trained), stratum on_hull_single|metal (Cu-Zr
  compounds are all-metal; the MP Cu-Zr entries used by ORCHID are on-hull single entries).
sigma_MLIP->DFT: RMS of (MLIP - DFT) formation energies of the Cu-Zr compounds over MP and OQMD entries in ORCHID's
  compound pages (lowest entry at the composition) whose space group matches the MLIP structure (others listed, not used).
Writes results/sigma_mlip.json."""
import json, re
from pathlib import Path
import numpy as np
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, SGTE_TDB, orchid, PFP_SRC  # noqa: F401 (paths: _paths.py)
CAL = json.load(open(orchid("espei/dft/CALIBRATION.json")))
s_dft = CAL["sources"]["mp"]["strata"]["on_hull_single|metal"]["sigma_robust"]
PAGES = {"CU5ZR_C15B": "Cu5Zr", "CU51ZR14": "Cu51Zr14", "CU8ZR3": "Cu8Zr3", "CU10ZR7": "Cu10Zr7", "CUZR_B2": "CuZr", "CUZR2": "CuZr2"}
SG = {"CU5ZR_C15B": ["F-43m"], "CU51ZR14": ["P-6", "P6/m"], "CU8ZR3": ["Pnma"], "CU10ZR7": ["Cmce", "Aba2"], "CUZR_B2": ["Pm-3m"], "CUZR2": ["I4/mmm"]}
F = json.load(open(ROOT / "tdb/mlip_fit.json"))
rows, diffs = [], []
for ph, page in PAGES.items():
    md = orchid(f"compounds/{page}.md").read_text()
    mlip = F["compounds"][ph]["dHf_J_mol_atom"] / 1000
    for m in re.finditer(r"^\| (Materials Project|OQMD) \| (\S+) \| ([-\d.]+) \| [-\d.]+ \| (\S+) \| ([-\d.]+) \|", md, re.M):
        src, eid, v, sg, ehull = m.group(1), m.group(2), float(m.group(3)), m.group(4), float(m.group(5))
        used = sg in SG[ph]
        rows.append({"phase": ph, "source": src, "entry": eid, "dft_kJ": v, "space_group": sg, "e_above_hull_eV": ehull,
                     "mlip_kJ": mlip, "diff_kJ": mlip - v, "used": used})
        if used: diffs.append(mlip - v)
s_md = float(np.sqrt(np.mean(np.square(diffs))))
s_mlip = float(np.hypot(s_dft, s_md))
out = {"sigma_DFT_kJ": s_dft, "sigma_DFT_source": "CALIBRATION.json sources.mp.strata['on_hull_single|metal'].sigma_robust",
       "sigma_MLIP_to_DFT_kJ": s_md, "n_pairs": len(diffs), "mean_MLIP_minus_DFT_kJ": float(np.mean(diffs)),
       "sigma_MLIP_kJ": s_mlip, "espei_weight": 500.0 / (s_mlip * 1000), "pairs": rows}
json.dump(out, open(ROOT / "results/sigma_mlip.json", "w"), indent=1)
for r in rows: print("%-11s %-17s %-12s %-7s %7.2f MLIP %7.2f diff %6.2f %s" % (r["phase"], r["source"], r["entry"], r["space_group"], r["dft_kJ"], r["mlip_kJ"], r["diff_kJ"], "used" if r["used"] else "not used (structure differs)"))
print({k: v for k, v in out.items() if k != "pairs"})
