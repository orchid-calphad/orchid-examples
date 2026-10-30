# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/sigma_mlip.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Estimate sigma_MLIP = sqrt(sigma_DFT^2 + sigma_MLIP->DFT^2) (PI instruction D, PLAN.md amendment 1).

sigma_DFT: ORCHID's calibrated DFT-vs-experiment robust sigma, espei/dft/CALIBRATION.json, source Materials Project
  (PBE, the functional family of OMat24 on which GRACE-2L-OMAT is trained), stratum on_hull_single|metal (Cu-Zr
  compounds are all-metal; the MP Cu-Zr entries used by ORCHID are on-hull single entries).
sigma_MLIP->DFT: RMS of (MLIP - DFT) formation energies of the Cu-Zr compounds over MP and OQMD entries in ORCHID's
  compound pages (lowest entry at the composition) whose space group matches the MLIP structure (others listed, not used).
Writes results/sigma_mlip.json. Usage: python scripts/full/sigma_mlip.py (needs ORCHID_DATA).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import orchid, root

j_per_kj = 1000  # J/kJ
weight_scale = 500.0  # J/mol: ESPEI weight = weight_scale / sigma
calibration = json.loads(orchid("espei/dft/CALIBRATION.json").read_text())
s_dft = calibration["sources"]["mp"]["strata"]["on_hull_single|metal"]["sigma_robust"]
pages = {
    "CU5ZR_C15B": "Cu5Zr",
    "CU51ZR14": "Cu51Zr14",
    "CU8ZR3": "Cu8Zr3",
    "CU10ZR7": "Cu10Zr7",
    "CUZR_B2": "CuZr",
    "CUZR2": "CuZr2",
}
space_groups = {
    "CU5ZR_C15B": ["F-43m"],
    "CU51ZR14": ["P-6", "P6/m"],
    "CU8ZR3": ["Pnma"],
    "CU10ZR7": ["Cmce", "Aba2"],
    "CUZR_B2": ["Pm-3m"],
    "CUZR2": ["I4/mmm"],
}
mlip_fit = json.loads((root / "tdb/mlip_fit.json").read_text())
rows, diffs = [], []
for ph, page in pages.items():
    md = orchid(f"compounds/{page}.md").read_text()
    mlip = mlip_fit["compounds"][ph]["dHf_J_mol_atom"] / j_per_kj
    for m in re.finditer(r"^\| (Materials Project|OQMD) \| (\S+) \| ([-\d.]+) \| [-\d.]+ \| (\S+) \| ([-\d.]+) \|", md, re.M):
        src, eid, v, sg, ehull = m.group(1), m.group(2), float(m.group(3)), m.group(4), float(m.group(5))
        used = sg in space_groups[ph]
        rows.append(
            {
                "phase": ph,
                "source": src,
                "entry": eid,
                "dft_kJ": v,
                "space_group": sg,
                "e_above_hull_eV": ehull,
                "mlip_kJ": mlip,
                "diff_kJ": mlip - v,
                "used": used,
            }
        )
        if used:
            diffs.append(mlip - v)
s_md = float(np.sqrt(np.mean(np.square(diffs))))
s_mlip = float(np.hypot(s_dft, s_md))
out = {
    "sigma_DFT_kJ": s_dft,
    "sigma_DFT_source": "CALIBRATION.json sources.mp.strata['on_hull_single|metal'].sigma_robust",
    "sigma_MLIP_to_DFT_kJ": s_md,
    "n_pairs": len(diffs),
    "mean_MLIP_minus_DFT_kJ": float(np.mean(diffs)),
    "sigma_MLIP_kJ": s_mlip,
    "espei_weight": weight_scale / (s_mlip * j_per_kj),
    "pairs": rows,
}
(root / "results/sigma_mlip.json").write_text(json.dumps(out, indent=1))
for r in rows:
    used_text = "used" if r["used"] else "not used (structure differs)"
    print(
        f"{r['phase']:<11} {r['source']:<17} {r['entry']:<12} {r['space_group']:<7} {r['dft_kJ']:7.2f} MLIP "
        f"{r['mlip_kJ']:7.2f} diff {r['diff_kJ']:6.2f} {used_text}"
    )
print({k: v for k, v in out.items() if k != "pairs"})
