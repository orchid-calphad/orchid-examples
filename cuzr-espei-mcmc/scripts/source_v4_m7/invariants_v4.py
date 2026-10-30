"""Invariants of scripts/v2/invariants.py with an optional third liquid term (p[16] = 2L a, m7).

Reference copy: the v2 module is not shipped in this example, so this file only registers the patch on import.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parent / ".." / "v2"))
import invariants as iv  # noqa: E402

_orig = iv.phase_G


def phase_gibbs(temp: float, p: np.ndarray) -> dict:
    """Phase Gibbs energies of the v2 model plus the optional third liquid term.

    Args:
        temp (float): Temperature in K.
        p (np.ndarray): Parameter vector; elements 0-15 are the v2 vector and element 16, when present and nonzero,
            is the third liquid coefficient 2L (J/mol).

    Returns:
        dict: Per phase, (x, G) with G in J/mol of atoms, as returned by the v2 phase_G.
    """
    out = _orig(temp, p[:16])
    if len(p) > 16 and p[16] != 0:
        x, g = out["LIQUID"]
        out["LIQUID"] = (x, g + x * (1 - x) * p[16] * ((1 - x) - x) ** 2)
    return out


iv.phase_G = phase_gibbs
