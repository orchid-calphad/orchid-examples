"""Fast analytic evaluation of the Cu-Zr PhaseForge TDB family (CuZr-PhaseForge-MLIP.tdb and its tuned versions) and
of reference TDBs, for invariant temperatures and phase-diagram lines.

Every phase is evaluated per mole of atoms: unary functions (SGTE, identical in all TDBs used) are tabulated once with
pycalphad on a 1 K grid; solution phases are two-term Redlich-Kister, compounds are line compounds. The stable phases
at (T, x) are read off the lower convex hull (1-D equilibrium for a binary at fixed T, P). Invariant temperatures are
located where a probe phase appears/disappears at a probe composition (adapted from orchid-espei-demo
scripts/v2/invariants.py, same reaction list)

Copied from orchid-phaseforge-demo scripts/model.py; only the two paths (SGTE TDB, unary-table cache) changed.. validate() compares with pycalphad equilibrium.
"""
import json, functools, re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TDB = ROOT / "tdb/CuZr-94Zen-LB.tdb"   # SGTE (Hallstedt 2025, CC BY 4.0); unary functions only (identical in all TDBs used)
R = 8.31451
XG = np.concatenate([np.linspace(1e-6, 0.02, 60), np.linspace(0.02, 0.98, 961)[1:-1], np.linspace(0.98, 1 - 1e-6, 60)])
COMP = {'CU5ZR_C15B': (5, 1), 'CU51ZR14': (51, 14), 'CU8ZR3': (8, 3), 'CU10ZR7': (10, 7), 'CUZR_B2': (1, 1), 'CUZR2': (1, 2)}
SOL = ('LIQUID', 'FCC_A1', 'BCC_A2', 'HCP_A3')
TMIN, TMAX = 500, 2600
REACTIONS = [
    ('L = FCC + Cu5Zr (eutectic)', 0.088, 'LIQUID', 'appears', {'LIQUID', 'FCC_A1', 'CU5ZR_C15B'}),
    ('L + Cu51Zr14 = Cu5Zr (peritectic)', 'CU5ZR_C15B', 'CU5ZR_C15B', 'disappears', {'LIQUID', 'CU51ZR14', 'CU5ZR_C15B'}),
    ('L = Cu51Zr14 (congruent)', 'CU51ZR14', 'LIQUID', 'appears', {'LIQUID', 'CU51ZR14'}),
    ('L + Cu51Zr14 = Cu8Zr3', 'CU8ZR3', 'CU8ZR3', 'disappears', {'LIQUID', 'CU51ZR14', 'CU8ZR3'}),
    ('L = Cu8Zr3 + Cu10Zr7 (eutectic)', 0.342, 'LIQUID', 'appears', {'LIQUID', 'CU8ZR3', 'CU10ZR7'}),
    ('L = Cu10Zr7 (congruent)', 'CU10ZR7', 'LIQUID', 'appears', {'LIQUID', 'CU10ZR7'}),
    ('L = Cu10Zr7 + CuZr (eutectic)', 0.445, 'LIQUID', 'appears', {'LIQUID', 'CU10ZR7', 'CUZR_B2'}),
    ('L = CuZr (congruent)', 'CUZR_B2', 'LIQUID', 'appears', {'LIQUID', 'CUZR_B2'}),
    ('L = CuZr + CuZr2 (eutectic)', 0.555, 'LIQUID', 'appears', {'LIQUID', 'CUZR_B2', 'CUZR2'}),
    ('L = CuZr2 (congruent)', 'CUZR2', 'LIQUID', 'appears', {'LIQUID', 'CUZR2'}),
    ('L = CuZr2 + BCC (eutectic)', 0.72, 'LIQUID', 'appears', {'LIQUID', 'CUZR2', 'BCC_A2'}),
    ('BCC = CuZr2 + HCP (eutectoid)', 0.975, 'BCC_A2', 'appears', {'BCC_A2', 'CUZR2', 'HCP_A3'}),
    ('CuZr = Cu10Zr7 + CuZr2 (eutectoid)', 'CUZR_B2', 'CUZR_B2', 'appears', {'CUZR_B2', 'CU10ZR7', 'CUZR2'}),
]


@functools.lru_cache(maxsize=1)
def unaries():
    cache = ROOT / "inputs/unary_tables.npz"   # cached: pycalphad 0.11.2 tables of the SGTE unaries, 1 K grid
    if cache.exists():
        z = np.load(cache); return {k: z[k] for k in z.files}
    from pycalphad import Database, calculate
    db = Database(str(TDB))
    Ts = np.arange(TMIN, TMAX + 1, 1.0)
    out = {'T': Ts}
    for ph in SOL:
        pts = [[1, 0], [0, 1]] if ph == 'LIQUID' else [[1, 0, 1], [0, 1, 1]]
        r = calculate(db, ['CU', 'ZR', 'VA'], ph, T=Ts, P=101325, N=1, points=np.asarray(pts, float), output='GM')
        g = np.squeeze(r.GM.values)
        # remove the (MLIP) mixing part: endmember points carry only the unary
        out[ph + '_CU'], out[ph + '_ZR'] = g[:, 0], g[:, 1]
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, **out)
    return out


class Params:
    """Parameter set: dict with L[(phase, k)] = (a_J, b_J_per_K) and G[compound] = (a_J_formula, b_J_formula_per_K)."""
    def __init__(self, L, G):
        self.L, self.G = L, G

    @classmethod
    def from_tdb_symbols(cls, symbols):
        """symbols: list of {'symbol','parameter','start'} (tdb/mlip_fit.json) and a value vector or dict."""
        raise NotImplementedError

    @classmethod
    def from_tdb(cls, path, values=None):
        """Parse a TDB written by build_mlip_tdb.py (VV symbols) or a 94Zen-style TDB (numbers)."""
        txt = Path(path).read_text()
        sym = {m.group(1): float(m.group(2)) for m in re.finditer(r"FUNCTION\s+(VV\d+)\s+298\.15\s+([-+.\dEe]+);", txt)}
        if values: sym.update(values)
        def ev(expr):
            expr = expr.replace(' ', '')
            for s, v in sym.items(): expr = expr.replace(s, '(%r)' % v)
            expr = re.sub(r'\d+\*GHSER(CU|ZR)', '', expr).replace('+GHSERCU', '').replace('+GHSERZR', '').replace('GHSERCU', '').replace('GHSERZR', '')
            expr = expr.lstrip('+')
            a = eval(expr.replace('T', '0')); b = eval(expr.replace('T', '1')) - a
            return a, b
        L, G = {}, {}
        flat = re.sub(r"\s+", " ", txt)
        for m in re.finditer(r"PAR\w*\s+L\((\w+)(?::\w+)?,CU,ZR(?::VA)?;(\d)\)(?:,,|\s+[\d.]+)\s*([^;]+);", flat):
            L[(m.group(1), int(m.group(2)))] = ev(m.group(3))
        for m in re.finditer(r"PAR\w*\s+G\((\w+),CU:ZR\)(?:,,|\s+[\d.]+)\s*([^;]+);", flat):
            G[m.group(1)] = ev(m.group(2))
        return cls(L, G)


def phase_G(T, p, xg=XG):
    t = unaries()
    j = int(min(max(np.floor(T - TMIN), 0), len(t['T']) - 2))
    tv = lambda k: t[k][j] + (t[k][j + 1] - t[k][j]) * (T - t['T'][j])
    x = xg
    sid = R * T * (x * np.log(x) + (1 - x) * np.log(1 - x))
    out = {}
    for ph in SOL:
        g = (1 - x) * tv(ph + '_CU') + x * tv(ph + '_ZR') + sid
        for k in (0, 1):
            a, b = p.L.get((ph, k), (0.0, 0.0))
            g = g + x * (1 - x) * (a + b * T) * ((1 - x) - x) ** k
        out[ph] = (x, g)
    gcu, gzr = tv('FCC_A1_CU'), tv('HCP_A3_ZR')
    for ph, (nc, nz) in COMP.items():
        if ph not in p.G: continue
        a, b = p.G[ph]
        out[ph] = (np.array([nz / (nc + nz)]), np.array([(nc * gcu + nz * gzr + a + b * T) / (nc + nz)]))
    return out


def lower_hull(xs, gs, labels):
    o = np.lexsort((gs, xs)); xs, gs, labels = xs[o], gs[o], labels[o]
    h = []
    for k in range(len(xs)):
        while len(h) >= 2:
            i, j = h[-2], h[-1]
            if (xs[j] - xs[i]) * (gs[k] - gs[i]) - (gs[j] - gs[i]) * (xs[k] - xs[i]) <= 0:
                h.pop()
            else:
                break
        h.append(k)
    return xs[h], gs[h], labels[h]


def hull(T, p):
    G = phase_G(T, p)
    xs = np.concatenate([v[0] for v in G.values()]); gs = np.concatenate([v[1] for v in G.values()])
    lab = np.concatenate([[k] * len(v[0]) for k, v in G.items()])
    return lower_hull(xs, gs, lab)


def stable_at(T, p, x0):
    hx, hg, hl = hull(T, p)
    j = np.searchsorted(hx, x0)
    if j < len(hx) and abs(hx[j] - x0) < 1e-9: return {hl[j]}
    j = min(max(j, 1), len(hx) - 1)
    return {hl[j - 1], hl[j]}


def probe_x(spec):
    if isinstance(spec, str):
        nc, nz = COMP[spec]; return nz / (nc + nz)
    return spec


def invariant_T(p, rx, Tguess, window=400.0, step=10.0, tol=0.25):
    name, xs, ph, mode, _ = rx
    x0 = probe_x(xs)
    cond = lambda T: (ph in stable_at(T, p, x0)) if mode == 'appears' else (ph not in stable_at(T, p, x0))
    grid = np.arange(max(TMIN, Tguess - window), min(TMAX - 1, Tguess + window) + 1e-9, step)
    c = [cond(T) for T in grid]
    flips = [k for k in range(len(grid) - 1) if (not c[k]) and c[k + 1]]
    if not flips: return None
    k = min(flips, key=lambda k: abs(grid[k] - Tguess))
    lo, hi = grid[k], grid[k + 1]
    while hi - lo > tol:
        m = 0.5 * (lo + hi)
        if cond(m): hi = m
        else: lo = m
    return 0.5 * (lo + hi)


def phase_boundaries(p, Ts):
    """Two-phase regions on the hull at each T: list of (T, x_left, x_right, phase_left, phase_right)."""
    rows = []
    for T in Ts:
        hx, hg, hl = hull(T, p)
        for i in range(len(hx) - 1):
            if hl[i] != hl[i + 1] and hx[i + 1] - hx[i] > 2e-3:
                rows.append((T, hx[i], hx[i + 1], hl[i], hl[i + 1]))
    return rows


def validate(tdb, p, rxT):
    from pycalphad import Database, equilibrium, variables as v
    db = Database(str(tdb))
    res = []
    for name, T in rxT.items():
        if T is None: continue
        rx = [r for r in REACTIONS if r[0] == name][0]
        x0 = probe_x(rx[1]); ok = []
        for dT in (-3.0, 3.0):
            eq = equilibrium(db, ['CU', 'ZR', 'VA'], list(db.phases), {v.X('ZR'): x0, v.T: T + dT, v.P: 101325, v.N: 1})
            ok.append(sorted(set(str(q) for q in np.squeeze(eq.Phase.values) if q)))
        res.append((name, round(T, 1), ok))
    return res
