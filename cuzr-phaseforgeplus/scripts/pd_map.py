"""Phase-diagram geometry for the binary Cu-Zr models.

On a fine T grid the stable assemblages along x are read from the lower convex hull of all phases' molar Gibbs energies
(exact 1-D equilibrium at fixed T, P; model.py, validated against pycalphad equilibrium). Regions (single-phase solution
fields and two-phase fields) are tracked across T into continuous fields; their edges are the phase boundaries; field
ends away from the plot limits are invariants. Returns plain arrays for plotting. (Copied unchanged from
orchid-phaseforge-demo scripts/pd_map.py.)
"""

from __future__ import annotations

import model as m
import numpy as np

xg_grid = np.concatenate([np.geomspace(1e-7, 0.01, 300), np.linspace(0.01, 0.99, 2941)[1:-1], 1 - np.geomspace(0.01, 1e-7, 300)])
solution_phases = ("LIQUID", "FCC_A1", "BCC_A2", "HCP_A3")
match_tol = 0.02  # mole fraction; a region continues across T steps when its edges move by less than this


def rows_at(temperature, p):
    """Single-phase and two-phase regions of the convex hull at one temperature.

    Args:
        temperature (float): Temperature in K.
        p (m.Params): Parameter set of the TDB.

    Returns:
        list[tuple[str, tuple[str, ...], float, float]]: (kind, phases, x_left, x_right) per region. Kind "1" is a
        single-phase region (one phase), kind "2" a two-phase region (two phases, left and right).
    """
    curves = m.phase_g(temperature, p, xg_grid)
    xs = np.concatenate([v[0] for v in curves.values()])
    gs = np.concatenate([v[1] for v in curves.values()])
    lab = np.concatenate([[k] * len(v[0]) for k, v in curves.items()])
    hx, _hg, hl = m.lower_hull(xs, gs, lab)
    regs, i = [], 0
    while i < len(hx) - 1:
        if hl[i] == hl[i + 1] and hl[i] in solution_phases:  # single-phase run
            j = i
            while j < len(hx) - 1 and hl[j + 1] == hl[i]:
                j += 1
            regs.append(("1", (str(hl[i]),), float(hx[i]), float(hx[j])))
            i = j
        else:
            if hx[i + 1] - hx[i] > 1e-9 and hl[i] != hl[i + 1]:
                regs.append(("2", (str(hl[i]), str(hl[i + 1])), float(hx[i]), float(hx[i + 1])))
            i += 1
    return regs


def track(p, temps):
    """Follow each region across temperature and merge matching regions into continuous fields.

    Args:
        p (m.Params): Parameter set of the TDB.
        temps (np.ndarray): Temperatures in K, in increasing order.

    Returns:
        list[dict]: One track per field with keys "kind", "phases", and arrays "T", "xl", "xr" (K and mole fractions).
    """
    tracks, open_ = [], []
    for temperature in temps:
        regs = rows_at(temperature, p)
        new_open = []
        used = set()
        for kind, ph, xl, xr in regs:
            best = None
            for k, tr in enumerate(open_):
                if k in used or tr["kind"] != kind or tr["phases"] != ph:
                    continue
                pl, pr = tr["xl"][-1], tr["xr"][-1]
                if xr >= pl - match_tol and xl <= pr + match_tol:
                    d = abs(xl - pl) + abs(xr - pr)
                    if best is None or d < best[0]:
                        best = (d, k)
            if best is not None:
                tr = open_[best[1]]
                used.add(best[1])
            else:
                tr = {"kind": kind, "phases": ph, "T": [], "xl": [], "xr": []}
                tracks.append(tr)
            tr["T"].append(temperature)
            tr["xl"].append(xl)
            tr["xr"].append(xr)
            new_open.append(tr)
        open_ = new_open
    for tr in tracks:
        for k in ("T", "xl", "xr"):
            tr[k] = np.array(tr[k])
    return tracks


def compound_ranges(p, temps):
    """Temperature range over which each line compound is stable.

    Args:
        p (m.Params): Parameter set of the TDB.
        temps (np.ndarray): Temperatures in K.

    Returns:
        dict[str, tuple[float, float, float]]: Compound name to (x_Zr, T_min, T_max) for compounds with a
        non-empty stability range on the grid; T in K.
    """
    out = {}
    for ph, (nc, nz) in m.line_compounds.items():
        if ph not in p.G:
            continue
        x = nz / (nc + nz)
        st = [temperature for temperature in temps if ph in m.stable_at(temperature, p, x)]
        if st:
            out[ph] = (x, min(st), max(st))
    return out
