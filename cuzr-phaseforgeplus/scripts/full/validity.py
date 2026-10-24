# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/validity.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Physical-validity check: count (T, x) grid points (500-1300 K step 10, x(Zr) 0.05-0.63 step 0.01) where BCC_A2 or
HCP_A3 is part of the stable assemblage (they are physically stable only on the Zr side above ~1100 K / below ~1140 K).
Usage: validity.py TDB [TDB ...] -> prints count and T/x extent per TDB."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, SGTE_TDB, orchid  # noqa: E402,F401
import sys, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import model as m
for t in sys.argv[1:]:
    p = m.Params.from_tdb(t); bad = []
    for T in range(500, 1301, 10):
        for x in np.arange(0.05, 0.631, 0.01):
            s = m.stable_at(T, p, x)
            if "BCC_A2" in s or "HCP_A3" in s: bad.append((T, round(float(x), 2)))
    ext = (min(b[0] for b in bad), max(b[0] for b in bad), min(b[1] for b in bad), max(b[1] for b in bad)) if bad else None
    print(t, len(bad), ext, flush=True)
