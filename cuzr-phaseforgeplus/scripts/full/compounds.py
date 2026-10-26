# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/compounds.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Starting structures of the Cu-Zr intermetallics (phase names as in CuZr-94Zen-LB.tdb).

Source: G. Ghosh, Acta Mater. 55 (2007) 3347-3374, doi:10.1016/j.actamat.2007.01.037: Table 2 (space group,
prototype), Table 11 (experimental lattice parameters, nm) and Table 12 (internal coordinates from his ab initio
relaxation; fractions printed as 1/4, 1/3, 1/2, 2/3). All structures are fully relaxed with the MLIP afterwards, so the
starting coordinates only fix the topology.

CU51ZR14: Ghosh's Ag51Gd14-type model fills all 68 sites of P6/m (Cu54Zr14); its formation energy is ~6 kJ/mol-atom
above the hull (Ghosh) and the MLIP reproduces that (-8.1 kJ/mol-atom, first attempt, runs/phaseforge/COMPOUNDS_hp68).
Zhong et al., Acta Mater. 61 (2013) 660 (doi:10.1016/j.actamat.2012.10.014) report for the isotypic Cu51Hf14 that the
Cu 6j site (the hexagon ring around the c axis at z = 0, nearest-neighbour Cu-Cu 2.24 A when full) is half occupied
(54 - 3 = 51 Cu); their lowest configuration has the three Cu evenly distributed on the ring (alternate sites). We use
that ordered 65-atom model (P-6), as does Materials Project's P-6 Cu51Zr14 entry.
Writes structures/<PHASE>.vasp and structures/compounds.json.
"""
import json
from pathlib import Path
import numpy as np
from pymatgen.core import Structure, Lattice

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, SGTE_TDB, orchid, PFP_SRC  # noqa: F401 (paths: _paths.py)
q, t, h, tt = 0.25, 1 / 3, 0.5, 2 / 3
DEF = {
    "CU5ZR_C15B": ("F-43m", Lattice.cubic(6.8700), [("Cu", (q, q, q)), ("Cu", (0.62427,) * 3), ("Zr", (0, 0, 0))], (5, 1)),
    "CU51ZR14": ("P6/m", Lattice.hexagonal(11.2444, 8.2815),
                 [("Cu", (t, tt, 0)), ("Cu", (t, tt, 0.28729)), ("Cu", (0.21646, 0.17611, 0)), ("Cu", (0.23273, 0.03419, h)),
                  ("Cu", (0.28065, 0.16606, 0.26004)), ("Cu", (0.11953, 0.48406, 0.15264)), ("Cu", (0.44506, 0.08468, 0.33818)),
                  ("Zr", (0, 0, 0.29961)), ("Zr", (0.42579, 0.12154, 0)), ("Zr", (0.16438, 0.47635, h))], (51, 14)),
    "CU8ZR3": ("Pnma", Lattice.orthorhombic(7.8693, 8.1547, 9.9848),
               [("Cu", (0.28419, q, 0.27181)), ("Cu", (0.11674, q, 0.90223)), ("Cu", (0.37074, q, 0.76560)),
                ("Cu", (0.32298, q, 0.51665)), ("Cu", (0.37770, 0.57554, 0.15958)), ("Cu", (0.19836, 0.49967, 0.40619)),
                ("Zr", (0.46266, q, 0.046036)), ("Zr", (0.06035, 0.44744, 0.13974))], (8, 3)),
    "CU10ZR7": ("Aba2", Lattice.orthorhombic(9.3466, 9.3163, 12.6729),
                [("Cu", (0.20468, 0.02099, 0.80082)), ("Cu", (0.29534, 0.02100, 0.59167)), ("Cu", (0.00865, 0.29583, 0.59218)),
                 ("Cu", (0.00863, 0.29582, 0.30029)), ("Cu", (0.10131, 0.10572, 0.44619)),
                 ("Zr", (0, 0, 0.94623)), ("Zr", (0, 0, 0.63601)), ("Zr", (0, 0, 0.25646)),
                 ("Zr", (0.25, 0.26673, 0.69622)), ("Zr", (0.30485, 0.30945, 0.44623))], (10, 7)),
    "CUZR_B2": ("Pm-3m", Lattice.cubic(3.2587), [("Cu", (0, 0, 0)), ("Zr", (h, h, h))], (1, 1)),
    "CUZR2": ("I4/mmm", Lattice.tetragonal(3.2204, 11.1832), [("Cu", (0, 0, 0)), ("Zr", (0, 0, 0.34848))], (1, 2)),
}
# Note: Ghosh prints Cu(2a) z = 0.00278 for CuZr2; 2a is (0,0,0) by symmetry (rounding artefact), used as such.
# CU10ZR7 Aba2 Zr4 x printed 0.25000 (Table 12).

out = {}
for name, (sg, lat, sites, (na, nb)) in DEF.items():
    s = Structure.from_spacegroup(sg, lat, [e for e, _ in sites], [c for _, c in sites], tol=1e-3)
    s.merge_sites(tol=0.05, mode="delete")
    if name == "CU51ZR14":
        # Cu3 6j ring at z=0 around (0,0): keep alternate sites (angles 0, 120, 240 deg), remove the other three
        ring = [i for i, site in enumerate(s) if site.specie.symbol == "Cu" and abs(np.mod(site.frac_coords[2] + 0.5, 1) - 0.5) < 1e-3
                and 1.5 < np.linalg.norm(s.lattice.get_cartesian_coords(((site.frac_coords[:2] + 0.5) % 1 - 0.5).tolist() + [0])) < 3.0]
        assert len(ring) == 6, ring
        ang = {i: np.degrees(np.arctan2(*s.lattice.get_cartesian_coords(((s[i].frac_coords[:2] + 0.5) % 1 - 0.5).tolist() + [0])[[1, 0]])) % 360 for i in ring}
        order = sorted(ring, key=lambda i: ang[i])
        s.remove_sites(order[1::2])
    comp = s.composition
    ncu, nzr = int(round(comp["Cu"])), int(round(comp["Zr"]))
    k = ncu // na
    assert ncu == k * na and nzr == k * nb, (name, ncu, nzr)
    dm = s.distance_matrix + np.eye(len(s)) * 99
    prim = s.get_primitive_structure(tolerance=0.01)
    prim.to(filename=str(ROOT / "structures" / f"{name}.vasp"), fmt="poscar")
    out[name] = {"space_group_input": sg, "n_atoms_conventional": len(s), "n_atoms_primitive": len(prim),
                 "formula_units_primitive": len(prim) // (na + nb), "min_distance_A": float(dm.min()),
                 "sg_detected": prim.get_space_group_info(symprec=0.05)}
    print(name, out[name])
json.dump(out, open(ROOT / "structures/compounds.json", "w"), indent=1)
