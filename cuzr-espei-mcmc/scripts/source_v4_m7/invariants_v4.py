"""invariants of scripts/v2/invariants.py with an optional third liquid term (p[16] = 2L a, m7)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'v2'))
import invariants as iv
_orig = iv.phase_G


def phase_G(T, p):
    out = _orig(T, p[:16])
    if len(p) > 16 and p[16] != 0:
        x, G = out['LIQUID']
        out['LIQUID'] = (x, G + x * (1 - x) * p[16] * ((1 - x) - x) ** 2)
    return out


iv.phase_G = phase_G
