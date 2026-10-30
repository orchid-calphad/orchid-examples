# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/validate_sqs.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Independent check of the m_scraps SQS against the random-alloy targets.

The check uses icet cluster vectors (not m_scraps' own accounting). The same check is made for the ATAT sqsdb SQS that
`sqs2tdb -cp` copied for the PhaseForgePlus
Pt-W example (relabelled Pt->Cu, W->Zr and rescaled to the same lattice), evaluated with the same cluster spaces.

Correlation of a cluster of order n in the random alloy: Pi* = <theta>^n, with <theta> the point correlation.
Reports, per structure: max |Pi - Pi*| over pairs / triplets / quadruplets inside the target cutoffs, and over pairs
out to an extended radius (beyond the optimised range). Output: results/sqs_validation.json + table on stdout.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read
from icet import ClusterSpace
from icet.tools import map_structure_to_reference
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.atat import Mcsqs

sys.path.insert(0, str(Path(__file__).parent))
from sqs_mscraps import prim, setups

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import pfp_src, root

run_dir = root / "runs/phaseforge"
atat_ex = (pfp_src / "example/PhaseForge") if pfp_src else None
ext_radius = {"FCC_A1": 3.0, "BCC_A2": 2.6, "HCP_A3": 2.6, "LIQUID": 2.05}  # extended pair radius, units of NN distance


def check(atoms, lat: str, x: float) -> dict:
    """Cluster-vector check of one SQS against the random-alloy correlations.

    Args:
        atoms (ase.Atoms): SQS to check (already rescaled to the reference lattice).
        lat (str): Lattice name.
        x (float): Zr fraction.

    Returns:
        Dict with the max deviation from <theta>^n for pairs, triplets and quadruplets (n_* orbit counts), the extended
        pair radius and shortest image (in units of the NN distance), the extended-pair deviation, and the point
        correlation.
    """
    p, d1 = prim(lat, x)
    cut = [c * d1 for c in setups[lat]["cut"]]
    cs = ClusterSpace(p, cut, [["Cu", "Zr"]] * len(p))
    atoms, _info = map_structure_to_reference(atoms, p, inert_species=None)
    cv = cs.get_cluster_vector(atoms)
    orb = cs.as_list
    pt = cv[1]
    out = {}
    for order, lab in ((2, "pairs"), (3, "triplets"), (4, "quadruplets")):
        idx = [i for i, o in enumerate(orb) if o["order"] == order]
        out[lab] = float(np.max(np.abs(cv[idx] - pt**order))) if idx else None
        out["n_" + lab] = len(idx)
    img = min(np.linalg.norm(np.array(v) @ atoms.cell) for v in itertools.product(range(-2, 3), repeat=3) if any(v))
    rext = min(ext_radius[lat] * d1, 0.99 * img)
    out["extended_radius_nn"] = rext / d1
    out["shortest_image_nn"] = img / d1
    cs2 = ClusterSpace(p, [rext], [["Cu", "Zr"]] * len(p))
    cv2 = cs2.get_cluster_vector(atoms)
    idx2 = [i for i, o in enumerate(cs2.as_list) if o["order"] == 2]
    dev = np.abs(cv2[idx2] - cv2[1] ** 2)
    out["pairs_extended_max"] = float(dev.max())
    out["n_pairs_extended"] = len(idx2)
    out["point"] = float(pt)
    return out


def rescale(atoms, lat: str, x: float):
    """Scale a cell to the volume per atom of the reference primitive cell at the composition.

    Args:
        atoms (ase.Atoms): Structure to rescale (copied, not modified).
        lat (str): Lattice name.
        x (float): Zr fraction.

    Returns:
        The rescaled copy of ``atoms`` (positions scaled with the cell).
    """
    p, _ = prim(lat, x)
    v = p.get_volume() / len(p)
    atoms = atoms.copy()
    atoms.set_cell(atoms.cell * (v * len(atoms) / atoms.get_volume()) ** (1 / 3), scale_atoms=True)
    return atoms


res = []
for lat in ("FCC_A1", "BCC_A2", "HCP_A3", "LIQUID"):
    for d in sorted((run_dir / lat).glob("sqs_lev=[12]_*")):
        m = json.loads((d / "sqs_meta.json").read_text(encoding="utf-8"))
        a = rescale(read(d / "POSCAR", format="vasp"), lat, m["x_Zr"])
        r = check(a, lat, m["x_Zr"])
        r.update(source="m_scraps", lattice=lat, x_Zr=m["x_Zr"], n_atoms=len(a), dir=d.name)
        res.append(r)
    if lat in ("FCC_A1", "BCC_A2"):
        for d in (
            sorted((atat_ex / lat).glob("sqs_lev=[12]_*")) if atat_ex else []
        ):  # optional ATAT comparison (PHASEFORGEPLUS_SRC)
            s = Mcsqs.structure_from_str((d / "str.out").read_text(encoding="utf-8"))
            s.replace_species({"Pt": "Cu", "W": "Zr"})
            a = AseAtomsAdaptor.get_atoms(s)
            x = float(sum(1 for z in a.get_chemical_symbols() if z == "Zr")) / len(a)
            a = rescale(a, lat, x)
            r = check(a, lat, x)
            r.update(source="ATAT sqsdb (PhaseForgePlus Pt-W example)", lattice=lat, x_Zr=x, n_atoms=len(a), dir=d.name)
            res.append(r)
(root / "results").mkdir(exist_ok=True)
(root / "results/sqs_validation.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def fmt_value(v) -> str:
    """Table cell for one validation value.

    Args:
        v (float | None): Deviation value, or None when the cluster order has no orbits.

    Returns:
        "-" for None, otherwise the value with 4 decimals.
    """
    return "-" if v is None else f"{v:.4f}"


print(f"{'lat':<7} {'x_Zr':<5} {'source':<9} {'N':>4} {'pairs':>9} {'trip':>9} {'quad':>9} {'pairs_ext':>9}")
for r in res:
    print(
        f"{r['lattice']:<7} {r['x_Zr']:<5.2f} {r['source'][:8]:<9} {r['n_atoms']:4d} {fmt_value(r['pairs']):>9} "
        f"{fmt_value(r['triplets']):>9} {fmt_value(r['quadruplets']):>9} {fmt_value(r['pairs_extended_max']):>9}  "
        f"({r['n_pairs']:d}/{r['n_triplets']:d}/{r['n_quadruplets']:d} orbits; ext {r['n_pairs_extended']:d} to "
        f"{r['extended_radius_nn']:.2f} d1, image {r['shortest_image_nn']:.2f} d1)"
    )
