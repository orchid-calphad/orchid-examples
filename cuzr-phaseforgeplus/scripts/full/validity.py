# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/validity.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Count the (T, x) grid points where BCC_A2 or HCP_A3 is stable.

Grid: 500-1300 K in steps of 10 K, x(Zr) 0.05-0.63 in steps of 0.01. The two lattices are physically stable only on
the Zr side above ~1100 K / below ~1140 K.
Usage: validity.py TDB [TDB ...] -> prints count and T/x extent per TDB.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))  # model.py (as _paths.py sets up)
import _paths  # noqa: E402,F401  side effect: work/ tree and cached TDB seeding, as before the restyle
import model as m

for t in sys.argv[1:]:
    p = m.Params.from_tdb(t)
    bad = []
    for temp in range(500, 1301, 10):
        for x in np.arange(0.05, 0.631, 0.01):
            s = m.stable_at(temp, p, x)
            if "BCC_A2" in s or "HCP_A3" in s:
                bad.append((temp, round(float(x), 2)))
    ext = (min(b[0] for b in bad), max(b[0] for b in bad), min(b[1] for b in bad), max(b[1] for b in bad)) if bad else None
    print(t, len(bad), ext, flush=True)
