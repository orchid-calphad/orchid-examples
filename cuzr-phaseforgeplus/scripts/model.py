"""Fast analytic evaluation of the Cu-Zr PhaseForge TDB family and of reference TDBs.

The family is CuZr-PhaseForge-MLIP.tdb and its tuned versions. The evaluation gives invariant temperatures and
phase-diagram lines.

Every phase is evaluated per mole of atoms: unary functions (SGTE, identical in all TDBs used) are tabulated once with
pycalphad on a 1 K grid; solution phases are two-term Redlich-Kister, compounds are line compounds. The stable phases
at (T, x) are read off the lower convex hull (1-D equilibrium for a binary at fixed T, P). Invariant temperatures are
located where a probe phase appears/disappears at a probe composition (adapted from orchid-espei-demo
scripts/v2/invariants.py, same reaction list)

Copied from orchid-phaseforge-demo scripts/model.py; only the two paths (SGTE TDB, unary-table cache) changed. validate()
compares with pycalphad equilibrium.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sgte_tdb = root / "tdb/CuZr-94Zen-LB.tdb"  # SGTE (Hallstedt 2025, CC BY 4.0); unary functions only (identical in all TDBs used)
gas_constant = 8.31451  # J / (mol*K)
pressure_pa = 101325  # Pa, 1 atm
xg_grid = np.concatenate([np.linspace(1e-6, 0.02, 60), np.linspace(0.02, 0.98, 961)[1:-1], np.linspace(0.98, 1 - 1e-6, 60)])
line_compounds = {
    "CU5ZR_C15B": (5, 1),
    "CU51ZR14": (51, 14),
    "CU8ZR3": (8, 3),
    "CU10ZR7": (10, 7),
    "CUZR_B2": (1, 1),
    "CUZR2": (1, 2),
}  # (Cu atoms, Zr atoms) per formula unit
solution_phases = ("LIQUID", "FCC_A1", "BCC_A2", "HCP_A3")
t_min = 500  # K, lowest tabulated temperature
t_max = 2600  # K, highest tabulated temperature
reactions = [
    ("L = FCC + Cu5Zr (eutectic)", 0.088, "LIQUID", "appears", {"LIQUID", "FCC_A1", "CU5ZR_C15B"}),
    ("L + Cu51Zr14 = Cu5Zr (peritectic)", "CU5ZR_C15B", "CU5ZR_C15B", "disappears", {"LIQUID", "CU51ZR14", "CU5ZR_C15B"}),
    ("L = Cu51Zr14 (congruent)", "CU51ZR14", "LIQUID", "appears", {"LIQUID", "CU51ZR14"}),
    ("L + Cu51Zr14 = Cu8Zr3", "CU8ZR3", "CU8ZR3", "disappears", {"LIQUID", "CU51ZR14", "CU8ZR3"}),
    ("L = Cu8Zr3 + Cu10Zr7 (eutectic)", 0.342, "LIQUID", "appears", {"LIQUID", "CU8ZR3", "CU10ZR7"}),
    ("L = Cu10Zr7 (congruent)", "CU10ZR7", "LIQUID", "appears", {"LIQUID", "CU10ZR7"}),
    ("L = Cu10Zr7 + CuZr (eutectic)", 0.445, "LIQUID", "appears", {"LIQUID", "CU10ZR7", "CUZR_B2"}),
    ("L = CuZr (congruent)", "CUZR_B2", "LIQUID", "appears", {"LIQUID", "CUZR_B2"}),
    ("L = CuZr + CuZr2 (eutectic)", 0.555, "LIQUID", "appears", {"LIQUID", "CUZR_B2", "CUZR2"}),
    ("L = CuZr2 (congruent)", "CUZR2", "LIQUID", "appears", {"LIQUID", "CUZR2"}),
    ("L = CuZr2 + BCC (eutectic)", 0.72, "LIQUID", "appears", {"LIQUID", "CUZR2", "BCC_A2"}),
    ("BCC = CuZr2 + HCP (eutectoid)", 0.975, "BCC_A2", "appears", {"BCC_A2", "CUZR2", "HCP_A3"}),
    ("CuZr = Cu10Zr7 + CuZr2 (eutectoid)", "CUZR_B2", "CUZR_B2", "appears", {"CUZR_B2", "CU10ZR7", "CUZR2"}),
]


@functools.lru_cache(maxsize=1)
def unaries() -> dict[str, np.ndarray]:
    """Tabulated SGTE unary Gibbs energies on a 1 K grid, read from the cache or computed with pycalphad.

    Returns:
        dict[str, np.ndarray]: Key "T" (K) and, per solution phase, "<PHASE>_CU" and "<PHASE>_ZR" (J/mol of atoms).
    """
    cache = root / "inputs/unary_tables.npz"  # cached: pycalphad 0.11.2 tables of the SGTE unaries, 1 K grid
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    from pycalphad import (  # noqa: PLC0415  lazy: pycalphad is heavy, needed only by some callers
        Database,
        calculate,
    )  # lazy: pycalphad is only needed when the cache is missing

    db = Database(str(sgte_tdb))
    temps = np.arange(t_min, t_max + 1, 1.0)
    out = {"T": temps}
    for ph in solution_phases:
        pts = [[1, 0], [0, 1]] if ph == "LIQUID" else [[1, 0, 1], [0, 1, 1]]
        r = calculate(db, ["CU", "ZR", "VA"], ph, T=temps, P=pressure_pa, N=1, points=np.asarray(pts, float), output="GM")
        g = np.squeeze(r.GM.values)
        # remove the (MLIP) mixing part: endmember points carry only the unary
        out[ph + "_CU"], out[ph + "_ZR"] = g[:, 0], g[:, 1]
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, **out)
    return out


class Params:
    """Parameter set of one TDB: Redlich-Kister and line-compound coefficients.

    Args:
        redlich_kister (dict[tuple[str, int], tuple[float, float]]): Coefficients (a, b) of the solution phase
            excess terms, keyed by (phase, order k); a in J/mol, b in J/(mol*K).
        compounds (dict[str, tuple[float, float]]): Coefficients (a, b) of each line compound per formula unit,
            a in J/mol and b in J/(mol*K).
    """

    def __init__(self, redlich_kister, compounds) -> None:
        """Store the Redlich-Kister and compound coefficient dictionaries."""
        self.L, self.G = redlich_kister, compounds

    @classmethod
    def from_tdb_symbols(cls, symbols) -> Params:
        """Build a parameter set from a symbol table and a value vector (not implemented).

        Args:
            symbols (list[dict]): Entries with keys "symbol", "parameter" and "start" (tdb/mlip_fit.json).

        Raises:
            NotImplementedError: Always; use from_tdb.
        """
        raise NotImplementedError

    @classmethod
    def from_tdb(cls, path, values=None) -> Params:
        """Parse a TDB written by build_mlip_tdb.py (VV symbols) or a 94Zen-style TDB (numbers).

        Args:
            path (str | Path): TDB file to parse.
            values (dict[str, float] | None): Optional overrides of the VV symbol values.

        Returns:
            Params: Coefficients as (a, b) pairs with the Gibbs energy a + b*T; a in J/mol, b in J/(mol*K).
        """
        txt = Path(path).read_text(encoding="utf-8")
        sym = {m.group(1): float(m.group(2)) for m in re.finditer(r"FUNCTION\s+(VV\d+)\s+298\.15\s+([-+.\dEe]+);", txt)}
        if values:
            sym.update(values)

        def ev(expr):
            """Evaluate a TDB expression at T=0 and T=1 and return (a, b) of a + b*T."""
            expr = expr.replace(" ", "")
            for s, v in sym.items():
                expr = expr.replace(s, f"({v!r})")
            expr = re.sub(r"\d+\*GHSER(CU|ZR)", "", expr)
            expr = expr.replace("+GHSERCU", "").replace("+GHSERZR", "").replace("GHSERCU", "").replace("GHSERZR", "")
            expr = expr.lstrip("+")
            a = eval(expr.replace("T", "0"))
            b = eval(expr.replace("T", "1")) - a
            return a, b

        lk, gc = {}, {}
        flat = re.sub(r"\s+", " ", txt)
        for m in re.finditer(r"PAR\w*\s+L\((\w+)(?::\w+)?,CU,ZR(?::VA)?;(\d)\)(?:,,|\s+[\d.]+)\s*([^;]+);", flat):
            lk[(m.group(1), int(m.group(2)))] = ev(m.group(3))
        for m in re.finditer(r"PAR\w*\s+G\((\w+),CU:ZR\)(?:,,|\s+[\d.]+)\s*([^;]+);", flat):
            gc[m.group(1)] = ev(m.group(2))
        return cls(lk, gc)


def phase_g(temperature, p, xg=xg_grid):
    """Molar Gibbs energy curves of every phase over a composition grid at one temperature.

    Args:
        temperature (float): Temperature in K.
        p (Params): Parameter set of the TDB.
        xg (np.ndarray): Zr mole fractions of the solution-phase grid.

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]: Per phase, (x, G) with G in J/mol of atoms. Line compounds are single
        points at their stoichiometric x.
    """
    t = unaries()
    j = int(min(max(np.floor(temperature - t_min), 0), len(t["T"]) - 2))

    def tv(k):
        """Unary energy of table key k interpolated linearly at the temperature, in J/mol of atoms."""
        return t[k][j] + (t[k][j + 1] - t[k][j]) * (temperature - t["T"][j])

    x = xg
    sid = gas_constant * temperature * (x * np.log(x) + (1 - x) * np.log(1 - x))
    out = {}
    for ph in solution_phases:
        g = (1 - x) * tv(ph + "_CU") + x * tv(ph + "_ZR") + sid
        for k in (0, 1):
            a, b = p.L.get((ph, k), (0.0, 0.0))
            g = g + x * (1 - x) * (a + b * temperature) * ((1 - x) - x) ** k
        out[ph] = (x, g)
    gcu, gzr = tv("FCC_A1_CU"), tv("HCP_A3_ZR")
    for ph, (nc, nz) in line_compounds.items():
        if ph not in p.G:
            continue
        a, b = p.G[ph]
        out[ph] = (
            np.array([nz / (nc + nz)]),
            np.array([(nc * gcu + nz * gzr + a + b * temperature) / (nc + nz)]),
        )
    return out


def lower_hull(xs, gs, labels):
    """Lower convex hull of labelled points (x, G), sorted by x.

    Args:
        xs (np.ndarray): Mole fractions of Zr, one per candidate point.
        gs (np.ndarray): Molar Gibbs energies in J/mol of atoms, one per point.
        labels (np.ndarray): Phase label of each point.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: x, G and label of the hull vertices, in increasing x.
    """
    order = np.lexsort((gs, xs))
    xs, gs, labels = xs[order], gs[order], labels[order]
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


def hull(temperature, p):
    """Convex hull of all phases' Gibbs energy curves at one temperature.

    Args:
        temperature (float): Temperature in K.
        p (Params): Parameter set of the TDB.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Hull x, G (J/mol of atoms) and phase label, from lower_hull.
    """
    curves = phase_g(temperature, p)
    xs = np.concatenate([v[0] for v in curves.values()])
    gs = np.concatenate([v[1] for v in curves.values()])
    lab = np.concatenate([[k] * len(v[0]) for k, v in curves.items()])
    return lower_hull(xs, gs, lab)


def stable_at(temperature, p, x0):
    """Phases in equilibrium at a composition and temperature (one phase on the hull, or two in a two-phase field).

    Args:
        temperature (float): Temperature in K.
        p (Params): Parameter set of the TDB.
        x0 (float): Zr mole fraction.

    Returns:
        set[str]: Labels of the stable phase(s) at x0.
    """
    hx, _hg, hl = hull(temperature, p)
    j = np.searchsorted(hx, x0)
    if j < len(hx) and abs(hx[j] - x0) < 1e-9:
        return {hl[j]}
    j = min(max(j, 1), len(hx) - 1)
    return {hl[j - 1], hl[j]}


def probe_x(spec):
    """Zr mole fraction of a probe composition given as a line-compound name or a fraction.

    Args:
        spec (str | float): Key of line_compounds, or a Zr mole fraction.

    Returns:
        float: Zr mole fraction.
    """
    if isinstance(spec, str):
        nc, nz = line_compounds[spec]
        return nz / (nc + nz)
    return spec


def invariant_temperature(p, rx, t_guess, *, window=400.0, step=10.0, tol=0.25):
    """Temperature of an invariant reaction, located as the flip of a probe phase near a guessed temperature.

    Args:
        p (Params): Parameter set of the TDB.
        rx (tuple): One entry of reactions: (name, probe composition, probe phase, "appears" or "disappears", phases).
        t_guess (float): Guessed invariant temperature in K; the search is centred here.
        window (float): Half-width of the search range in K.
        step (float): Scan step in K.
        tol (float): Bisection tolerance in K.

    Returns:
        float | None: Invariant temperature in K, or None when the probe phase does not flip inside the window.
    """
    name, xs, ph, mode, _ = rx
    x0 = probe_x(xs)

    def cond(temperature):
        """Whether the probe phase is in the state the reaction requires at the temperature."""
        present = ph in stable_at(temperature, p, x0)
        return present if mode == "appears" else not present

    grid = np.arange(max(t_min, t_guess - window), min(t_max - 1, t_guess + window) + 1e-9, step)
    c = [cond(temperature) for temperature in grid]
    flips = [k for k in range(len(grid) - 1) if (not c[k]) and c[k + 1]]
    if not flips:
        return None
    k = min(flips, key=lambda k: abs(grid[k] - t_guess))
    lo, hi = grid[k], grid[k + 1]
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if cond(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def phase_boundaries(p, temps):
    """Two-phase regions on the hull at each temperature.

    Args:
        p (Params): Parameter set of the TDB.
        temps (np.ndarray): Temperatures in K.

    Returns:
        list[tuple[float, float, float, str, str]]: (T, x_left, x_right, phase_left, phase_right) per two-phase region
        wider than 2e-3 mole fraction.
    """
    rows = []
    for temperature in temps:
        hx, _hg, hl = hull(temperature, p)
        for i in range(len(hx) - 1):
            if hl[i] != hl[i + 1] and hx[i + 1] - hx[i] > 2e-3:
                rows.append((temperature, hx[i], hx[i + 1], hl[i], hl[i + 1]))
    return rows


def validate(tdb, p, rx_temps):
    """Compare invariant temperatures with exact pycalphad equilibria just below and above each one.

    Args:
        tdb (str | Path): TDB file read by pycalphad.
        p (Params): Parameter set of the same TDB (unused by pycalphad; kept for the call signature).
        rx_temps (dict[str, float | None]): Reaction name to invariant temperature in K; None entries are skipped.

    Returns:
        list[tuple[str, float, list[list[str]]]]: (reaction name, T rounded to 0.1 K, sorted phase lists at T-3 K and
        T+3 K).
    """
    from pycalphad import (  # noqa: PLC0415  lazy: pycalphad is heavy, needed only by some callers
        Database,
        equilibrium,
    )  # lazy: pycalphad is needed only for validation
    from pycalphad import variables as v  # noqa: PLC0415  lazy: pycalphad is heavy and only some callers need it

    db = Database(str(tdb))
    res = []
    for name, temperature in rx_temps.items():
        if temperature is None:
            continue
        rx = [r for r in reactions if r[0] == name][0]
        x0 = probe_x(rx[1])
        ok = []
        for d_t in (-3.0, 3.0):
            eq = equilibrium(
                db,
                ["CU", "ZR", "VA"],
                list(db.phases),
                {v.X("ZR"): x0, v.T: temperature + d_t, v.P: pressure_pa, v.N: 1},
            )
            ok.append(sorted({str(q) for q in np.squeeze(eq.Phase.values) if q}))
        res.append((name, round(temperature, 1), ok))
    return res
