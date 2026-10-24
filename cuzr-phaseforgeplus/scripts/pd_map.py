"""Phase-diagram geometry for the binary Cu-Zr models: on a fine T grid the stable assemblages along x are read from the
lower convex hull of all phases' molar Gibbs energies (exact 1-D equilibrium at fixed T, P; model.py, validated
against pycalphad equilibrium). Regions (single-phase solution fields and two-phase fields) are tracked across T into
continuous fields; their edges are the phase boundaries; field ends away from the plot limits are invariants.
Returns plain arrays for plotting. (Copied unchanged from orchid-phaseforge-demo scripts/pd_map.py.)"""
import numpy as np
import model as m

XG = np.concatenate([np.geomspace(1e-7, 0.01, 300), np.linspace(0.01, 0.99, 2941)[1:-1], 1 - np.geomspace(0.01, 1e-7, 300)])
SOLN = ("LIQUID", "FCC_A1", "BCC_A2", "HCP_A3")


def rows_at(T, p):
    G = m.phase_G(T, p, XG)
    xs = np.concatenate([v[0] for v in G.values()]); gs = np.concatenate([v[1] for v in G.values()])
    lab = np.concatenate([[k] * len(v[0]) for k, v in G.items()])
    hx, hg, hl = m.lower_hull(xs, gs, lab)
    regs, i = [], 0
    while i < len(hx) - 1:
        if hl[i] == hl[i + 1] and hl[i] in SOLN:      # single-phase run
            j = i
            while j < len(hx) - 1 and hl[j + 1] == hl[i]: j += 1
            regs.append(("1", (str(hl[i]),), float(hx[i]), float(hx[j]))); i = j
        else:
            if hx[i + 1] - hx[i] > 1e-9 and hl[i] != hl[i + 1]:
                regs.append(("2", (str(hl[i]), str(hl[i + 1])), float(hx[i]), float(hx[i + 1])))
            i += 1
    # single-phase solution regions touching x = 0 or 1 at the end of the grid
    return regs


def track(p, Ts):
    tracks, open_ = [], []
    for T in Ts:
        regs = rows_at(T, p)
        new_open = []
        used = set()
        for kind, ph, xl, xr in regs:
            best = None
            for k, tr in enumerate(open_):
                if k in used or tr["kind"] != kind or tr["phases"] != ph: continue
                pl, pr = tr["xl"][-1], tr["xr"][-1]
                if xr >= pl - 0.02 and xl <= pr + 0.02:
                    d = abs(xl - pl) + abs(xr - pr)
                    if best is None or d < best[0]: best = (d, k)
            if best is not None:
                tr = open_[best[1]]; used.add(best[1])
            else:
                tr = {"kind": kind, "phases": ph, "T": [], "xl": [], "xr": []}; tracks.append(tr)
            tr["T"].append(T); tr["xl"].append(xl); tr["xr"].append(xr); new_open.append(tr)
        open_ = new_open
    for tr in tracks:
        for k in ("T", "xl", "xr"): tr[k] = np.array(tr[k])
    return tracks


def compound_ranges(p, Ts):
    out = {}
    for ph, (nc, nz) in m.COMP.items():
        if ph not in p.G: continue
        x = nz / (nc + nz)
        st = [T for T in Ts if ph in m.stable_at(T, p, x)]
        if st: out[ph] = (x, min(st), max(st))
    return out
