# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/validate_sqs.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Independent check of the m_scraps SQS (icet cluster vectors, not m_scraps' own accounting) against the
random-alloy targets, and the same check for the ATAT sqsdb SQS that `sqs2tdb -cp` copied for the PhaseForgePlus
Pt-W example (relabelled Pt->Cu, W->Zr and rescaled to the same lattice), evaluated with the same cluster spaces.

Correlation of a cluster of order n in the random alloy: Pi* = <theta>^n, with <theta> the point correlation.
Reports, per structure: max |Pi - Pi*| over pairs / triplets / quadruplets inside the target cutoffs, and over pairs
out to an extended radius (beyond the optimised range). Output: results/sqs_validation.json + table on stdout.
"""
import json, sys
from pathlib import Path
import numpy as np
from icet import ClusterSpace
from icet.tools import map_structure_to_reference
from pymatgen.io.atat import Mcsqs
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
sys.path.insert(0, str(Path(__file__).parent))
from sqs_mscraps import prim, SETUP

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, SGTE_TDB, orchid, DENSE_GRID, PFP_SRC  # noqa: F401 (paths: _paths.py)
RUN = ROOT / "runs/phaseforge"
ATAT_EX = (PFP_SRC / "example/PhaseForge") if PFP_SRC else None
EXT = {"FCC_A1": 3.0, "BCC_A2": 2.6, "HCP_A3": 2.6, "LIQUID": 2.05}   # extended pair radius, units of NN distance


def check(atoms, lat, x):
    p, d1 = prim(lat, x)
    cut = [c * d1 for c in SETUP[lat]["cut"]]
    cs = ClusterSpace(p, cut, [["Cu", "Zr"]] * len(p))
    atoms, _info = map_structure_to_reference(atoms, p, inert_species=None)
    cv = cs.get_cluster_vector(atoms)
    orb = cs.as_list
    pt = cv[1]
    out = {}
    for order, lab in ((2, "pairs"), (3, "triplets"), (4, "quadruplets")):
        idx = [i for i, o in enumerate(orb) if o["order"] == order]
        out[lab] = float(np.max(np.abs(cv[idx] - pt ** order))) if idx else None
        out["n_" + lab] = len(idx)
    from ase.geometry import cell_to_cellpar
    img = min(np.linalg.norm(np.array(v) @ atoms.cell) for v in __import__("itertools").product(range(-2, 3), repeat=3) if any(v))
    rext = min(EXT[lat] * d1, 0.99 * img)
    out["extended_radius_nn"] = rext / d1; out["shortest_image_nn"] = img / d1
    cs2 = ClusterSpace(p, [rext], [["Cu", "Zr"]] * len(p))
    cv2 = cs2.get_cluster_vector(atoms)
    idx2 = [i for i, o in enumerate(cs2.as_list) if o["order"] == 2]
    dev = np.abs(cv2[idx2] - cv2[1] ** 2)
    out["pairs_extended_max"] = float(dev.max()); out["n_pairs_extended"] = len(idx2)
    out["point"] = float(pt)
    return out


def rescale(atoms, lat, x):
    p, _ = prim(lat, x)
    v = p.get_volume() / len(p)
    atoms = atoms.copy()
    atoms.set_cell(atoms.cell * (v * len(atoms) / atoms.get_volume()) ** (1 / 3), scale_atoms=True)
    return atoms


res = []
for lat in ("FCC_A1", "BCC_A2", "HCP_A3", "LIQUID"):
    for d in sorted((RUN / lat).glob("sqs_lev=[12]_*")):
        m = json.loads((d / "sqs_meta.json").read_text())
        a = rescale(read(d / "POSCAR", format="vasp"), lat, m["x_Zr"])
        r = check(a, lat, m["x_Zr"]); r.update(source="m_scraps", lattice=lat, x_Zr=m["x_Zr"], n_atoms=len(a), dir=d.name)
        res.append(r)
    if lat in ("FCC_A1", "BCC_A2"):
        for d in (sorted((ATAT_EX / lat).glob("sqs_lev=[12]_*")) if ATAT_EX else []):   # optional ATAT comparison (PHASEFORGEPLUS_SRC)
            s = Mcsqs.structure_from_str((d / "str.out").read_text())
            s.replace_species({"Pt": "Cu", "W": "Zr"})
            a = AseAtomsAdaptor.get_atoms(s)
            x = float(sum(1 for z in a.get_chemical_symbols() if z == "Zr")) / len(a)
            a = rescale(a, lat, x)
            r = check(a, lat, x); r.update(source="ATAT sqsdb (PhaseForgePlus Pt-W example)", lattice=lat, x_Zr=x, n_atoms=len(a), dir=d.name)
            res.append(r)
(ROOT / "results").mkdir(exist_ok=True)
json.dump(res, open(ROOT / "results/sqs_validation.json", "w"), indent=1)
print("%-7s %-5s %-9s %4s %9s %9s %9s %9s" % ("lat", "x_Zr", "source", "N", "pairs", "trip", "quad", "pairs_ext"))
for r in res:
    f = lambda v: "-" if v is None else "%.4f" % v
    print("%-7s %-5.2f %-9s %4d %9s %9s %9s %9s  (%d/%d/%d orbits; ext %d to %.2f d1, image %.2f d1)" % (r["lattice"], r["x_Zr"], r["source"][:8], r["n_atoms"], f(r["pairs"]), f(r["triplets"]), f(r["quadruplets"]), f(r["pairs_extended_max"]), r["n_pairs"], r["n_triplets"], r["n_quadruplets"], r["n_pairs_extended"], r["extended_radius_nn"], r["shortest_image_nn"]))
