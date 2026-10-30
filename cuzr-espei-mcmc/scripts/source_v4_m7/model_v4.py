"""v4 parameter sets m6 and m7, their bounds and the TDB writer.

V4_MODEL=m6: the v3 six-parameter model (tdb/v3/CuZr-94Zen-LB_param_v3.tdb). V4_MODEL=m7: m6 plus a third
liquid interaction term 2L (a only, VV0006; start 0), tdb/v4/CuZr-94Zen-LB_param_v4_m7.tdb (model-adequacy test).

fullp(p) maps onto the v2 16-vector (+ L2 as element 16 for m7) used by invariants_v4.

Reference copy: the v3 module (make_start_tdb_v3) is not shipped in this example; its names (ROOT, SYMS, START, SCALE,
full16, write) are read from it unchanged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

here = Path(__file__).absolute().parent
sys.path[:0] = [str(here / ".." / "v3")]
import make_start_tdb_v3 as m3  # noqa: E402

root = m3.ROOT
model_kind = os.environ.get("V4_MODEL", "m6")
if model_kind == "m6":
    syms, start, scale = m3.SYMS, m3.START.copy(), m3.SCALE.copy()
    tdb_path = f"{root}/tdb/v3/CuZr-94Zen-LB_param_v3.tdb"
    # bounds: liquid as in v3; compound energies within +-15 kJ per atom (17 atoms CU10ZR7, 2 atoms CUZR_B2)
    lower = np.array([-120000, -30, -40000, -20, -241750 - 17 * 15000, -20104.24 - 2 * 15000])
    upper = np.array([0, 40, 40000, 20, -241750 + 17 * 15000, -20104.24 + 2 * 15000])
else:
    syms = [*m3.SYMS, "VV0006"]
    start = np.append(m3.START, 0.0)
    scale = np.append(m3.SCALE, 1000.0)
    tdb_path = f"{root}/tdb/v4/CuZr-94Zen-LB_param_v4_m7.tdb"
    lower = np.array([-120000, -30, -40000, -20, -241750 - 17 * 15000, -20104.24 - 2 * 15000, -40000])
    upper = np.array([0, 40, 40000, 20, -241750 + 17 * 15000, -20104.24 + 2 * 15000, 40000])


def fullp(p: np.ndarray) -> np.ndarray:
    """Expand a fitted parameter vector to the v2 16-vector, appending the third liquid term when present.

    Args:
        p (np.ndarray): Fitted parameters; the first 6 map through full16, element 6 (m7 only) is 2L in J/mol.

    Returns:
        np.ndarray: The 16-element vector, or 17 elements when p has a seventh element.
    """
    q = m3.full16(np.asarray(p)[:6])
    return np.append(q, p[6]) if len(p) > 6 else q


def write(path: str | Path, values: dict[str, float], label: str) -> None:
    """Write a v4 TDB; for m7 also add the VV0006 liquid term after the L1 parameter line.

    Args:
        path (str | Path): Output TDB path.
        values (dict[str, float]): Value of each VV symbol.
        label (str): Description written into the review-only header.
    """
    m3.write(path, values, label)
    if model_kind != "m6":
        path = Path(path)
        txt = path.read_text()
        old = "PAR  L(LIQUID,CU,ZR;1),,               VV0002+VV0003*T;,,         N 94Zen !\n"
        assert txt.count(old) == 1
        txt = txt.replace(old, old + " PAR  L(LIQUID,CU,ZR;2),,               VV0006;,,         N demo !\n")
        txt = txt.replace(
            " TEMP-LIM",
            f"$ VV0006: L2(LIQUID) a (added term, model-adequacy test)\n FUNCTION VV0006    298.15 {values['VV0006']:+.6f}; 6000 N !\n TEMP-LIM",
            1,
        )
        path.write_text(txt)


if __name__ == "__main__":
    Path(f"{root}/tdb/v4").mkdir(parents=True, exist_ok=True)
    write(tdb_path, dict(zip(syms, start, strict=True)), "Starting model: SGTE values + L2 = 0 (v4 m7).")
    print(tdb_path)
