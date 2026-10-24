# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/build_a4.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Amendment 4 data (post-hoc, PI-directed): data/fit with
  (1) the six CuZr-94Zen-LB hmr-07/08 HM_FORM datasets removed: the same printed values as Turchanin 2008's compilation
      (Zeng 1994 table re-listed), as orchid-espei-demo v3 de-duplication;
  (2) experimental HM_FORM sigma widened per compound to max(printed sigma, 3 kJ/mol-atom, SD of that compound's
      remaining fitted values) (weight = 500 J/mol / sigma);
  (3) + the dense assessed-field ZPF grid of orchid-espei-demo Amendment 4 (derived from Okamoto's Cu-Zr field map,
      NOT in the ORCHID release; weight reset to 1);
  (4) every ZPF weight (literature, brackets, assessed, liquid stability, dense) x F, F in {3, 10, 30}.
MLIP constraint data (data/mlip_anchor) unchanged, added by the fit script. Writes data/a4_F<F>/ and data/a4_manifest.json."""
import json, glob, collections
from pathlib import Path
import numpy as np
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, SGTE_TDB, orchid, DENSE_GRID, PFP_SRC  # noqa: F401 (paths: _paths.py)
DENSE = DENSE_GRID   # not distributed (see _paths.py); None -> no dense grid
DUP = {"experimental__CU-ZR__CuZr-94Zen-LB.hmr-07-CU10ZR7.json", "experimental__CU-ZR__CuZr-94Zen-LB.hmr-07-CU51ZR14.json",
       "experimental__CU-ZR__CuZr-94Zen-LB.hmr-07-CUZR2.json", "experimental__CU-ZR__CuZr-94Zen-LB.hmr-07-CUZR_B2.json",
       "experimental__CU-ZR__CuZr-94Zen-LB.hmr-08-CUZR2.json", "experimental__CU-ZR__CuZr-94Zen-LB.hmr-08-CUZR_B2.json"}
FLOOR = 3000.0
import sys
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "a4"
files = [f for f in sorted(glob.glob(str(ROOT / "data/fit/*.json"))) if Path(f).name not in DUP]
vals = collections.defaultdict(list)
for f in files:
    d = json.load(open(f))
    if d["output"] == "HM_FORM": vals[d["phases"][0]] += list(np.ravel(d["values"]))
sd = {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in vals.items()}
man = {"removed_duplicates": sorted(DUP), "sigma_floor_J": FLOOR, "scatter_sd_J": sd, "sigma_eff": {}, "dense": str(DENSE)}
for F in (3, 10, 30):
    out = ROOT / f"data/{PREFIX}_F{F}"; out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.json"): f.unlink()
    if DENSE is None:
        print('WARNING: ORCHID_DENSE_GRID not set: the dense assessed-field grid (not in the ORCHID release) is left out; '
              'this is not the configuration of the paper\'s final fit')
    for f in files + ([str(DENSE)] if DENSE is not None else []):
        d = json.load(open(f))
        if f == str(DENSE): d["weight"] = 1.0
        if d["output"] == "HM_FORM":
            w = d.get("weight", 1.0); w = w[0] if isinstance(w, list) else w
            s = max(500.0 / w, FLOOR, sd.get(d["phases"][0], 0.0))
            d["weight"] = 500.0 / s; man["sigma_eff"][Path(f).name] = s
        if d["output"] == "ZPF":
            w = d.get("weight", 1.0); d["weight"] = [x * F for x in w] if isinstance(w, list) else w * F
        json.dump(d, open(out / Path(f).name, "w"))
json.dump(man, open(ROOT / f"data/{PREFIX}_manifest.json", "w"), indent=1)
print(len(files), {k: round(v) for k, v in sd.items()}, sorted(set(round(v) for v in man["sigma_eff"].values())))
