"""Fast analytic evaluation of the Cu-Zr PhaseForge TDB family and of reference TDBs.

Used for invariant temperatures and phase-diagram lines. Every phase is evaluated per mole of atoms: unary functions
(SGTE, identical in all TDBs used) are tabulated once with pycalphad on a 1 K grid; solution phases are two-term
Redlich-Kister, compounds are line compounds. The stable phases at (T, x) are read off the lower convex hull (1-D
equilibrium for a binary at fixed T, P). Invariant temperatures are located where a probe phase appears or disappears
at a probe composition (adapted from orchid-espei-demo scripts/v2/invariants.py, same reaction list).

Copied from orchid-phaseforge-demo scripts/model.py; only the two paths (SGTE TDB, unary-table cache) changed.
validate() compares the result with pycalphad equilibrium.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sgte_tdb = root / "tdb/CuZr-94Zen-LB.tdb"  # SGTE (Hallstedt 2025, CC BY 4.0); unary functions only (identical in all TDBs used)
gas_constant = 8.31451  # J/(mol*K)
xgrid = np.concatenate([np.linspace(1e-6, 0.02, 60), np.linspace(0.02, 0.98, 961)[1:-1], np.linspace(0.98, 1 - 1e-6, 60)])
stoichiometry = {
    "CU5ZR_C15B": (5, 1),
    "CU51ZR14": (51, 14),
    "CU8ZR3": (8, 3),
    "CU10ZR7": (10, 7),
    "CUZR_B2": (1, 1),
    "CUZR2": (1, 2),
}
solution_phases = ("LIQUID", "FCC_A1", "BCC_A2", "HCP_A3")
tmin, tmax = 500, 2600
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
    """Tabulated SGTE unary Gibbs energies on a 1 K grid, read from the cache or rebuilt with pycalphad.

    Returns:
        Dict with key "T" (K, tmin to tmax in 1 K steps) and "<PHASE>_CU", "<PHASE>_ZR" (J per mole of atoms).
    """
    cache = root / "inputs/unary_tables.npz"  # cached: pycalphad 0.11.2 tables of the SGTE unaries, 1 K grid
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    from pycalphad import Database, calculate  # noqa: PLC0415  lazy: pycalphad is only needed to rebuild the cache

    db = Database(str(sgte_tdb))
    temps = np.arange(tmin, tmax + 1, 1.0)
    out = {"T": temps}
    for ph in solution_phases:
        pts = [[1, 0], [0, 1]] if ph == "LIQUID" else [[1, 0, 1], [0, 1, 1]]
        r = calculate(db, ["CU", "ZR", "VA"], ph, T=temps, P=101325, N=1, points=np.asarray(pts, float), output="GM")
        g = np.squeeze(r.GM.values)
        # remove the (MLIP) mixing part: endmember points carry only the unary
        out[ph + "_CU"], out[ph + "_ZR"] = g[:, 0], g[:, 1]
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, **out)
    return out


class Params:
    """Parameter set of the analytic model.

    Args:
        liquid (dict[tuple[str, int], tuple[float, float]]): Liquid Redlich-Kister terms keyed by (phase, order), each
            as (a, b) with a in J/mol and b in J/(mol*K).
        compound (dict[str, tuple[float, float]]): Line-compound Gibbs energies keyed by phase, each as (a, b) per
            mole of formula unit, with a in J/mol and b in J/(mol*K).
    """

    def __init__(self, liquid: dict, compound: dict) -> None:
        """Store the liquid and compound parameter tables."""
        self.liquid = liquid
        self.compound = compound

    @classmethod
    def from_tdb_symbols(cls, symbols: list[dict]) -> Params:
        """Build a parameter set from a symbol list (not implemented; kept for the API of the source model).

        Args:
            symbols (list[dict]): Entries with keys 'symbol', 'parameter' and 'start'.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError

    @classmethod
    def from_tdb(cls, path: str | Path, values: dict[str, float] | None = None) -> Params:
        """Parse a TDB written by make_param_tdb.py (VV symbols) or a 94Zen-style TDB (numbers).

        Args:
            path (str | Path): TDB file to read.
            values (dict[str, float] | None): Optional VV-symbol values that override the FUNCTION values of the file.

        Returns:
            Params: Liquid and compound coefficients, each as (a, b) with the T-independent and T-linear parts.
        """
        txt = Path(path).read_text()
        sym = {m.group(1): float(m.group(2)) for m in re.finditer(r"FUNCTION\s+(VV\d+)\s+298\.15\s+([-+.\dEe]+);", txt)}
        if values:
            sym.update(values)

        def ev(expr: str) -> tuple[float, float]:
            """Evaluate a TDB expression as (a, b) with G = a + b*T.

            Args:
                expr (str): TDB expression with VV symbols and GHSER terms.

            Returns:
                tuple[float, float]: Constant part a and linear-in-T part b, in the units of the expression.
            """
            expr = expr.replace(" ", "")
            for s, v in sym.items():
                expr = expr.replace(s, f"({v!r})")
            expr = re.sub(r"\d+\*GHSER(CU|ZR)", "", expr).replace("+GHSERCU", "").replace("+GHSERZR", "")
            expr = expr.replace("GHSERCU", "").replace("GHSERZR", "")
            expr = expr.lstrip("+")
            a = eval(expr.replace("T", "0"))
            b = eval(expr.replace("T", "1")) - a
            return a, b

        liquid, compound = {}, {}
        flat = re.sub(r"\s+", " ", txt)
        for m in re.finditer(r"PAR\w*\s+L\((\w+)(?::\w+)?,CU,ZR(?::VA)?;(\d)\)(?:,,|\s+[\d.]+)\s*([^;]+);", flat):
            liquid[(m.group(1), int(m.group(2)))] = ev(m.group(3))
        for m in re.finditer(r"PAR\w*\s+G\((\w+),CU:ZR\)(?:,,|\s+[\d.]+)\s*([^;]+);", flat):
            compound[m.group(1)] = ev(m.group(2))
        return cls(liquid, compound)


def phase_gibbs(temp: float, p: Params, xg: np.ndarray = xgrid) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Gibbs energy per mole of atoms of every phase on a composition grid at one temperature.

    Args:
        temp (float): Temperature in K.
        p (Params): Parameter set.
        xg (np.ndarray): Atomic fraction of Zr on the grid for the solution phases.

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]: Per phase, (x, G) with x the Zr fraction and G in J/mol of atoms;
        line compounds have a single point.
    """
    t = unaries()
    j = int(min(max(np.floor(temp - tmin), 0), len(t["T"]) - 2))

    def tv(k: str) -> float:
        """Interpolate a tabulated unary energy at the current temperature.

        Args:
            k (str): Key of the table, e.g. "LIQUID_CU".

        Returns:
            float: Unary Gibbs energy in J/mol of atoms.
        """
        return t[k][j] + (t[k][j + 1] - t[k][j]) * (temp - t["T"][j])

    x = xg
    sid = gas_constant * temp * (x * np.log(x) + (1 - x) * np.log(1 - x))
    out = {}
    for ph in solution_phases:
        g = (1 - x) * tv(ph + "_CU") + x * tv(ph + "_ZR") + sid
        for k in (0, 1):
            a, b = p.liquid.get((ph, k), (0.0, 0.0))
            g = g + x * (1 - x) * (a + b * temp) * ((1 - x) - x) ** k
        out[ph] = (x, g)
    gcu, gzr = tv("FCC_A1_CU"), tv("HCP_A3_ZR")
    for ph, (nc, nz) in stoichiometry.items():
        if ph not in p.compound:
            continue
        a, b = p.compound[ph]
        out[ph] = (np.array([nz / (nc + nz)]), np.array([(nc * gcu + nz * gzr + a + b * temp) / (nc + nz)]))
    return out


def lower_hull(xs: np.ndarray, gs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lower convex hull of points (x, G) with labels, by a monotone-chain scan.

    Args:
        xs (np.ndarray): Composition coordinates (atomic fraction of Zr).
        gs (np.ndarray): Gibbs energies per mole of atoms, in J/mol.
        labels (np.ndarray): Phase label of each point.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Hull vertices as (x, G, label), sorted by x.
    """
    o = np.lexsort((gs, xs))
    xs, gs, labels = xs[o], gs[o], labels[o]
    h: list[int] = []
    for k in range(len(xs)):
        while len(h) >= 2:
            i, j = h[-2], h[-1]
            if (xs[j] - xs[i]) * (gs[k] - gs[i]) - (gs[j] - gs[i]) * (xs[k] - xs[i]) <= 0:
                h.pop()
            else:
                break
        h.append(k)
    return xs[h], gs[h], labels[h]


def hull(temp: float, p: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lower convex hull of all phases at one temperature.

    Args:
        temp (float): Temperature in K.
        p (Params): Parameter set.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Hull vertices as (x, G in J/mol of atoms, phase label).
    """
    phases = phase_gibbs(temp, p)
    xs = np.concatenate([v[0] for v in phases.values()])
    gs = np.concatenate([v[1] for v in phases.values()])
    labels = np.concatenate([[k] * len(v[0]) for k, v in phases.items()])
    return lower_hull(xs, gs, labels)


def stable_at(temp: float, p: Params, x0: float) -> set[str]:
    """Phases on the hull at a composition, one phase inside a single-phase field or two at a boundary.

    Args:
        temp (float): Temperature in K.
        p (Params): Parameter set.
        x0 (float): Atomic fraction of Zr.

    Returns:
        set[str]: Stable phase label(s) at (temp, x0).
    """
    hx, _hg, hl = hull(temp, p)
    j = np.searchsorted(hx, x0)
    if j < len(hx) and abs(hx[j] - x0) < 1e-9:
        return {hl[j]}
    j = min(max(j, 1), len(hx) - 1)
    return {hl[j - 1], hl[j]}


def probe_x(spec: str | float) -> float:
    """Atomic fraction of Zr of a probe composition.

    Args:
        spec (str | float): Compound key in stoichiometry, or an explicit atomic fraction of Zr.

    Returns:
        float: Atomic fraction of Zr.
    """
    if isinstance(spec, str):
        nc, nz = stoichiometry[spec]
        return nz / (nc + nz)
    return spec


def invariant_temperature(
    p: Params,
    rx: tuple,
    tguess: float,
    *,
    window: float = 400.0,
    step: float = 10.0,
    tol: float = 0.25,
) -> float | None:
    """Temperature of an invariant reaction, found where its probe phase appears or disappears.

    Args:
        p (Params): Parameter set.
        rx (tuple): Reaction entry of `reactions` (name, probe composition, probe phase, mode, phase set).
        tguess (float): Printed or guessed temperature in K; the nearest flip to it is used.
        window (float): Half-width in K of the scan around tguess.
        step (float): Scan step in K.
        tol (float): Bisection tolerance in K.

    Returns:
        float | None: Invariant temperature in K, or None when no flip is found in the window.
    """
    _name, xs, ph, mode, _ = rx
    x0 = probe_x(xs)

    def cond(temp: float) -> bool:
        """Whether the probe phase is (appears mode) or is not (disappears mode) stable at temp.

        Args:
            temp (float): Temperature in K.

        Returns:
            bool: The condition of the reaction at temp.
        """
        if mode == "appears":
            return ph in stable_at(temp, p, x0)
        return ph not in stable_at(temp, p, x0)

    grid = np.arange(max(tmin, tguess - window), min(tmax - 1, tguess + window) + 1e-9, step)
    c = [cond(temp) for temp in grid]
    flips = [k for k in range(len(grid) - 1) if (not c[k]) and c[k + 1]]
    if not flips:
        return None
    k = min(flips, key=lambda idx: abs(grid[idx] - tguess))
    lo, hi = grid[k], grid[k + 1]
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if cond(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def phase_boundaries(p: Params, temps: np.ndarray) -> list[tuple]:
    """Two-phase regions on the hull at each temperature.

    Args:
        p (Params): Parameter set.
        temps (np.ndarray): Temperatures in K.

    Returns:
        list[tuple]: Rows (T, x_left, x_right, phase_left, phase_right), one per two-phase region.
    """
    rows = []
    for temp in temps:
        hx, _hg, hl = hull(temp, p)
        for i in range(len(hx) - 1):
            if hl[i] != hl[i + 1] and hx[i + 1] - hx[i] > 2e-3:
                rows.append((temp, hx[i], hx[i + 1], hl[i], hl[i + 1]))
    return rows


def validate(tdb_path: str | Path, p: Params, rx_temps: dict) -> list[tuple]:
    """Compare invariant temperatures with pycalphad equilibrium at +-3 K.

    Args:
        tdb_path (str | Path): TDB used by pycalphad.
        p (Params): Parameter set (unused by the equilibrium check; kept for the source interface).
        rx_temps (dict): Reaction name to invariant temperature in K (None entries are skipped).

    Returns:
        list[tuple]: Rows (name, T rounded to 0.1 K, sorted phase sets at T-3 K and T+3 K).
    """
    from pycalphad import Database, equilibrium  # noqa: PLC0415  lazy: needs pycalphad
    from pycalphad import variables as v  # noqa: PLC0415

    db = Database(str(tdb_path))
    res = []
    for name, temp in rx_temps.items():
        if temp is None:
            continue
        rx = [r for r in reactions if r[0] == name][0]
        x0 = probe_x(rx[1])
        ok = []
        for dtemp in (-3.0, 3.0):
            eq = equilibrium(db, ["CU", "ZR", "VA"], list(db.phases), {v.X("ZR"): x0, v.T: temp + dtemp, v.P: 101325, v.N: 1})
            ok.append(sorted({str(q) for q in np.squeeze(eq.Phase.values) if q}))
        res.append((name, round(temp, 1), ok))
    return res
