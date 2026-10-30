# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/sqs_mscraps.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""m_scraps replacement for ATAT `sqs2tdb -cp` (PhaseForge `sqscal` step 1).

`sqs2tdb -cp -sp=Cu,Zr -l=<LAT> -lv=2` copies pre-generated mcsqs structures from ATAT's sqsdb into
<LAT>/sqs_lev=<l>_a_Cu=<x>,a_Zr=<y>/str.out and touches `wait`. This adapter writes the same directory layout and
files (str.out in ATAT format, lat.in, POSCAR, wait, plus sqs_meta.json), but generates every SQS with m_scraps
(decorated pair/triplet/quadruplet correlation objective, deterministic seed) instead of copying mcsqs output.

Levels as in ATAT: lev 0 = pure endmembers (primitive cell), lev 1 = x 0.5, lev 2 = x 0.25 / 0.75.
LIQUID: PhaseForge/ATAT use fcc-based SQS as liquid starting configurations; here a 108-atom fcc SQS (3x3x3
conventional) per composition, including the pure endmembers, which then melt in MD (liquid_md.py).

Usage: python sqs_mscraps.py <outdir> [LATTICE ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import _paths  # noqa: E402,F401  side effect: work/ tree and cached TDB seeding, as before the restyle
import m_scraps
import numpy as np
from ase.build import bulk, find_optimal_cell_shape, make_supercell

species = ["Cu", "Zr"]
volume_a3 = {"Cu": 11.81, "Zr": 23.27}  # A^3/atom, elemental room-temperature volumes (Vegard start geometry only)
levels = {0: [0.0, 1.0], 1: [0.5], 2: [0.25, 0.75]}  # x_Zr
c_over_a = np.sqrt(8 / 3)


def vol(x: float) -> float:
    """Vegard-law volume per atom of a Cu-Zr composition.

    Args:
        x (float): Zr fraction.

    Returns:
        Volume per atom in A^3.
    """
    return (1 - x) * volume_a3["Cu"] + x * volume_a3["Zr"]


def prim(lat: str, x: float, conventional: bool = False):
    """Start cell of a lattice at a composition, with its nearest-neighbour distance.

    Args:
        lat (str): Lattice name: FCC_A1, LIQUID, BCC_A2 or HCP_A3.
        x (float): Zr fraction (sets the lattice constant through ``vol``).
        conventional (bool): Return the conventional (cubic) cell for FCC and BCC instead of the primitive one.

    Returns:
        Tuple (ase Atoms of pure Cu in the start cell, nearest-neighbour distance in A).

    Raises:
        ValueError: If the lattice is not known.
    """
    v = vol(x)
    if lat in ("FCC_A1", "LIQUID"):
        a = (4 * v) ** (1 / 3)
        return bulk("Cu", "fcc", a=a, cubic=conventional), a / np.sqrt(2)
    if lat == "BCC_A2":
        a = (2 * v) ** (1 / 3)
        return bulk("Cu", "bcc", a=a, cubic=conventional), a * np.sqrt(3) / 2
    if lat == "HCP_A3":
        a = (2 * v / (np.sqrt(3) / 2 * c_over_a)) ** (1 / 3)
        return bulk("Cu", "hcp", a=a, c=c_over_a * a), a
    raise ValueError(lat)


# supercell (of the cell returned by prim(..., conventional=True)) and figure cutoffs in units of the NN distance
setups = {
    "FCC_A1": {"cell_dim": (2, 2, 4), "cut": (2.05, 1.45, 1.05)},  # 64 atoms; pairs to 4th NN, triplets 2nd, quads 1st
    "BCC_A2": {"cell_dim": (4, 4, 2), "cut": (2.05, 1.20, 1.05)},  # 64 atoms; pairs to 5th NN, triplets 2nd, quads 1st
    "HCP_A3": {
        "cell_dim": (4, 2, 4),
        "cut": (2.05, 1.45, 1.05),
    },  # 64 atoms; pairs to 2.05 d1 (7 orbits), triplets 2nd (in-plane), quads 1st
    "LIQUID": {"cell_dim": (3, 3, 3), "cut": (1.45, 1.05, 1.05)},  # 108 atoms (fcc start for MD); pairs 2 shells
}


def write_str_out(atoms, path) -> None:
    """Write an ATAT str.out with a Cartesian (identity) coordinate system, as extract_MLIP writes str_relax.out.

    Args:
        atoms (ase.Atoms): Structure to write (cell and positions in A).
        path (str | Path): Output file.
    """
    with Path(path).open("w") as f:
        for r in np.eye(3):
            f.write(f"{r[0]:.6f} {r[1]:.6f} {r[2]:.6f}\n")
        for r in atoms.cell:
            f.write(f"{r[0]:.8f} {r[1]:.8f} {r[2]:.8f}\n")
        for p, s in zip(atoms.positions, atoms.get_chemical_symbols(), strict=True):
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {s}\n")


def lat_in(lat: str) -> str:
    """Contents of the ATAT lat.in file for a lattice.

    Args:
        lat (str): Lattice name.

    Returns:
        The lat.in text (the HCP c/a ratio is written with 6 decimals).
    """
    return {
        "FCC_A1": "1 1 1 90 90 90\n0 0.5 0.5\n0.5 0 0.5\n0.5 0.5 0\n0 0 0 Cu,Zr\n",
        "LIQUID": "1 1 1 90 90 90\n0 0.5 0.5\n0.5 0 0.5\n0.5 0.5 0\n0 0 0 Cu,Zr\n",
        "BCC_A2": "1 1 1 90 90 90\n-0.5 0.5 0.5\n0.5 -0.5 0.5\n0.5 0.5 -0.5\n0 0 0 Cu,Zr\n",
        "HCP_A3": f"1 1 {c_over_a:.6f} 90 90 120\n1 0 0\n0 1 0\n0 0 1\n0 0 0 Cu,Zr\n0.333333 0.666667 0.5 Cu,Zr\n",
    }[lat]


def make(lat: str, x: float, outdir, seed: int = 2026) -> Path:
    """Generate (or skip, if present) the SQS directory of one lattice and composition.

    Args:
        lat (str): Lattice name.
        x (float): Zr fraction.
        outdir (str | Path): Root output directory; the SQS goes to <outdir>/<lat>/sqs_lev=<l>_<composition>/.
        seed (int): Random seed of the m_scraps search.

    Returns:
        The SQS directory. Writes str.out, lat.in, POSCAR, wait and sqs_meta.json in it.
    """
    lev = [lv for lv, xs in levels.items() if x in xs][0]
    name = f"sqs_lev={lev:d}_" + ",".join(f"a_{s}={c:g}" for s, c in zip(species, (1 - x, x), strict=True) if c > 0)
    d = Path(outdir) / lat / name
    if (d / "str.out").exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    meta = {"lattice": lat, "x_Zr": x, "level": lev, "generator": f"m_scraps {m_scraps.__version__}"}
    if x in (0.0, 1.0) and lat != "LIQUID":
        a, d1 = prim(lat, x)
        a = a.copy()
        a.set_chemical_symbols(["Zr" if x == 1 else "Cu"] * len(a))
        meta.update(n_atoms=len(a), note="pure endmember, primitive cell (as ATAT sqsdb lev=0)")
    else:
        cfg = setups[lat]
        a0, d1 = prim(lat, x, conventional=True)
        n = len(a0) * int(np.prod(cfg["cell_dim"]))
        if lat != "LIQUID":  # most compact supercell shape of n atoms (largest shortest-image distance), as for mcsqs cells
            p0, _ = prim(lat, x)
            p_mat = find_optimal_cell_shape(p0.cell, n // len(p0), "fcc", verbose=False)
            a0 = make_supercell(p0, p_mat)
            cfg = {**cfg, "cell_dim": (1, 1, 1)}
            meta["supercell_matrix"] = p_mat.tolist()
        nzr = int(round(x * n))
        if x in (0.0, 1.0):  # pure liquid starting configuration: no chemical search needed
            a = a0.repeat(cfg["cell_dim"])
            a.set_chemical_symbols(["Zr" if x == 1 else "Cu"] * n)
            meta.update(n_atoms=n, note="pure liquid start (fcc supercell)")
        else:
            cut = [c * d1 for c in cfg["cut"]]
            wd = d / "m_scraps"
            a = m_scraps.run_sqs_from_atoms(
                a0,
                species=species,
                element_counts=[n - nzr, nzr],
                cell_dim=cfg["cell_dim"],
                figure_cutoffs=cut,
                figure_objective="weighted_l1",
                figure_wd=0.0,
                figure_wn=1.0,
                auto_budget=2,
                rng_seed=seed,
                ranks=4,
                work_dir=str(wd),
            )
            meta.update(
                n_atoms=len(a),
                counts=[n - nzr, nzr],
                cell_dim=cfg["cell_dim"],
                figure_cutoffs_A=cut,
                figure_cutoffs_nn=cfg["cut"],
                objective="weighted_l1 (wd=0, wn=1)",
                auto_budget=2,
                rng_seed=seed,
                final_fitness=a.info.get("scraps_final_fitness"),
            )
    a.pbc = True
    write_str_out(a, d / "str.out")
    (d / "lat.in").write_text(lat_in(lat))
    a.write(d / "POSCAR", format="vasp", direct=True, sort=True)
    (d / "wait").write_text("")
    meta["seconds"] = round(time.time() - t0, 1)
    (d / "sqs_meta.json").write_text(json.dumps(meta, indent=1))
    print(lat, name, meta.get("n_atoms"), meta.get("final_fitness"), meta["seconds"], "s", flush=True)
    return d


if __name__ == "__main__":
    out = sys.argv[1]
    lats = sys.argv[2:] or ["FCC_A1", "BCC_A2", "HCP_A3", "LIQUID"]
    for lat in lats:
        for lev in (0, 1, 2):
            for x in levels[lev]:
                make(lat, x, out)
