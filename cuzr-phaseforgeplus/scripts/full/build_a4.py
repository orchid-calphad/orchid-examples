# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/build_a4.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Build the Amendment 4 fit data (post-hoc, PI-directed) from data/fit.

Steps:
  (1) the six CuZr-94Zen-LB hmr-07/08 HM_FORM datasets are removed: the same printed values as Turchanin 2008's compilation
      (Zeng 1994 table re-listed), as orchid-espei-demo v3 de-duplication;
  (2) experimental HM_FORM sigma is widened per compound to max(printed sigma, 3 kJ/mol-atom, SD of that compound's
      remaining fitted values) (weight = 500 J/mol / sigma);
  (3) + the dense assessed-field ZPF grid of orchid-espei-demo Amendment 4 (derived from Okamoto's Cu-Zr field map, released
      in ORCHID assessed-phase-assemblages dense-field-grid; weight reset to 1, as in the final fit);
  (4) every ZPF weight (literature, brackets, assessed, liquid stability, dense) x F, F in {3, 10, 30}.

MLIP constraint data (data/mlip_anchor) is unchanged and is added by the fit script. Writes data/<PREFIX>_F<F>/ and
data/<PREFIX>_manifest.json, where PREFIX is the first argument (default a4).
Usage: python scripts/full/build_a4.py a4d
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import dense_grid, root

dense = dense_grid()  # released dense assessed-field grid (orchid-data dense_release, sha256 verified)
duplicates = {
    "experimental__CU-ZR__CuZr-94Zen-LB.hmr-07-CU10ZR7.json",
    "experimental__CU-ZR__CuZr-94Zen-LB.hmr-07-CU51ZR14.json",
    "experimental__CU-ZR__CuZr-94Zen-LB.hmr-07-CUZR2.json",
    "experimental__CU-ZR__CuZr-94Zen-LB.hmr-07-CUZR_B2.json",
    "experimental__CU-ZR__CuZr-94Zen-LB.hmr-08-CUZR2.json",
    "experimental__CU-ZR__CuZr-94Zen-LB.hmr-08-CUZR_B2.json",
}
sigma_floor = 3000.0  # J/mol-atom (3 kJ/mol-atom): lower bound on the widened HM_FORM sigma
weight_scale = 500.0  # J/mol: HM_FORM weight = weight_scale / sigma

prefix = sys.argv[1] if len(sys.argv) > 1 else "a4"
files = [f for f in sorted(str(p) for p in (root / "data/fit").glob("*.json")) if Path(f).name not in duplicates]
vals = collections.defaultdict(list)
for f in files:
    d = json.loads(Path(f).read_text(encoding="utf-8"))
    if d["output"] == "HM_FORM":
        vals[d["phases"][0]] += list(np.ravel(d["values"]))
sd = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in vals.items()}
man = {
    "removed_duplicates": sorted(duplicates),
    "sigma_floor_J": sigma_floor,
    "scatter_sd_J": sd,
    "sigma_eff": {},
    "dense": str(dense),
}
for factor in (3, 10, 30):
    out = root / f"data/{prefix}_F{factor}"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.json"):
        f.unlink()
    for f in files + [str(dense)]:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if f == str(dense):
            d["weight"] = 1.0
        if d["output"] == "HM_FORM":
            w = d.get("weight", 1.0)
            w = w[0] if isinstance(w, list) else w
            s = max(weight_scale / w, sigma_floor, sd.get(d["phases"][0], 0.0))
            d["weight"] = weight_scale / s
            man["sigma_eff"][Path(f).name] = s
        if d["output"] == "ZPF":
            w = d.get("weight", 1.0)
            d["weight"] = [x * factor for x in w] if isinstance(w, list) else w * factor
        (out / Path(f).name).write_text(json.dumps(d), encoding="utf-8")
(root / f"data/{prefix}_manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
print(len(files), {k: round(v) for k, v in sd.items()}, sorted({round(v) for v in man["sigma_eff"].values()}))
