"""Start geometries and special quasirandom structures (SQS) of the PhaseForge input set.

Endmembers and liquid starts are written in the primitive or fcc-supercell form, the other lattices as SQS from
m_scraps. Each structure gets the files that the SQS generator and the MLIP step read (POSCAR, str.out, lat.in,
wait, sqs_meta.json).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import m_scraps
import numpy as np
from ase.build import bulk, find_optimal_cell_shape, make_supercell
from pymatgen.core import Element

_ca = np.sqrt(8 / 3)  # c/a of the ideal hcp cell
levels = {0: [0.0, 1.0], 1: [0.5], 2: [0.25, 0.75]}  # SQS level -> mole fractions of the second element
solid_lattices = ["FCC_A1", "BCC_A2", "HCP_A3"]
# supercell (of the conventional cell) and figure cutoffs in units of the nearest-neighbour distance (as the demo)
_setup = {
    "FCC_A1": {"cell_dim": (2, 2, 4), "cut": (2.05, 1.45, 1.05)},
    "BCC_A2": {"cell_dim": (4, 4, 2), "cut": (2.05, 1.20, 1.05)},
    "HCP_A3": {"cell_dim": (4, 2, 4), "cut": (2.05, 1.45, 1.05)},
    "LIQUID": {"cell_dim": (3, 3, 3), "cut": (1.45, 1.05, 1.05)},
}
# Elemental room-temperature volumes (A^3/atom), Vegard start geometry only. The demo's Cu and Zr values are pinned so that
# Cu-Zr reproduces; other elements use pymatgen's molar volume rounded to 0.01 A^3.
_pinned_volumes = {"Cu": 11.81, "Zr": 23.27}
_avogadro_1e23 = 0.602214076  # Avogadro constant in 1e23 / mol: molar volume (cm^3/mol) / this = A^3/atom


def elem_volume(el: str) -> float:
    """Room-temperature volume per atom of an element, for the Vegard start geometry.

    Args:
        el (str): Element symbol.

    Returns:
        Volume in A^3/atom: the pinned value if there is one, otherwise pymatgen's molar volume rounded to 0.01 A^3.

    Raises:
        SystemExit: If the element has neither a pinned value nor a molar volume.
    """
    if el in _pinned_volumes:
        return _pinned_volumes[el]
    mv = Element(el).molar_volume
    if mv is None:
        raise SystemExit(f"no elemental volume for {el}; add it to _pinned_volumes")
    return round(float(mv) / _avogadro_1e23, 2)


class SQSMaker:
    """Builds the starting geometry and the special quasirandom structure (SQS) of one lattice of a binary system.

    Args:
        el_a (str): First element symbol (the matrix at x = 0).
        el_b (str): Second element symbol (the matrix at x = 1).
        seed (int): Random seed of the SQS search (rng_seed of m_scraps).
        ranks (int): MPI ranks of m_scraps.
    """

    def __init__(self, el_a: str, el_b: str, seed: int, ranks: int) -> None:
        """Store the species, their Vegard volumes and the SQS settings.

        Args:
            el_a (str): First element symbol.
            el_b (str): Second element symbol.
            seed (int): Random seed of the SQS search.
            ranks (int): MPI ranks of m_scraps.
        """
        self.species = [el_a, el_b]
        self.volumes = {el_a: elem_volume(el_a), el_b: elem_volume(el_b)}
        self.seed = seed
        self.ranks = ranks

    def vol(self, x: float) -> float:
        """Vegard volume of the mixture.

        Args:
            x (float): Mole fraction of the second element.

        Returns:
            Volume per atom in A^3.
        """
        return (1 - x) * self.volumes[self.species[0]] + x * self.volumes[self.species[1]]

    def prim(self, lat: str, x: float, conventional: bool = False) -> tuple[object, float]:
        """Primitive (or conventional) cell of a lattice at the Vegard volume.

        Args:
            lat (str): Lattice key: FCC_A1, BCC_A2, HCP_A3 or LIQUID.
            x (float): Mole fraction of the second element.
            conventional (bool): Return the conventional cubic cell (fcc and bcc) instead of the primitive cell.

        Returns:
            Tuple (ASE Atoms of the cell, nearest-neighbour distance in A).

        Raises:
            ValueError: If the lattice key is unknown.
        """
        v = self.vol(x)
        if lat in ("FCC_A1", "LIQUID"):
            a = (4 * v) ** (1 / 3)
            return bulk(self.species[0], "fcc", a=a, cubic=conventional), a / np.sqrt(2)
        if lat == "BCC_A2":
            a = (2 * v) ** (1 / 3)
            return bulk(self.species[0], "bcc", a=a, cubic=conventional), a * np.sqrt(3) / 2
        if lat == "HCP_A3":
            a = (2 * v / (np.sqrt(3) / 2 * _ca)) ** (1 / 3)
            return bulk(self.species[0], "hcp", a=a, c=_ca * a), a
        raise ValueError(lat)

    def lat_in(self, lat: str) -> str:
        """Contents of the lat.in file that the SQS generator reads for a lattice.

        Args:
            lat (str): Lattice key.

        Returns:
            The lat.in text.
        """
        s = ",".join(self.species)
        return {
            "FCC_A1": f"1 1 1 90 90 90\n0 0.5 0.5\n0.5 0 0.5\n0.5 0.5 0\n0 0 0 {s}\n",
            "LIQUID": f"1 1 1 90 90 90\n0 0.5 0.5\n0.5 0 0.5\n0.5 0.5 0\n0 0 0 {s}\n",
            "BCC_A2": f"1 1 1 90 90 90\n-0.5 0.5 0.5\n0.5 -0.5 0.5\n0.5 0.5 -0.5\n0 0 0 {s}\n",
            "HCP_A3": f"1 1 {_ca:.6f} 90 90 120\n1 0 0\n0 1 0\n0 0 1\n0 0 0 {s}\n0.333333 0.666667 0.5 {s}\n",
        }[lat]

    def name(self, lev: int, x: float) -> str:
        """Directory name of an SQS, encoding its level and composition.

        Args:
            lev (int): SQS level.
            x (float): Mole fraction of the second element.

        Returns:
            Name such as sqs_lev=1_a_Cu=0.5,a_Zr=0.5 (species with zero fraction are left out).
        """
        return f"sqs_lev={lev}_" + ",".join(f"a_{s}={c:g}" for s, c in zip(self.species, (1 - x, x), strict=True) if c > 0)

    def make(self, lat: str, x: float, outdir: str | Path, workroot: str | Path) -> tuple[Path, dict]:
        """Write the structure of one lattice at one composition and its SQS metadata.

        Args:
            lat (str): Lattice key: FCC_A1, BCC_A2, HCP_A3 or LIQUID.
            x (float): Mole fraction of the second element (0 and 1 are the pure endmembers).
            outdir (str | Path): Output root; the structure goes to <outdir>/<lat>/<name>.
            workroot (str | Path): Scratch root for the m_scraps working files, kept out of the output.

        Returns:
            Tuple (output directory, metadata dict also written as sqs_meta.json).
        """
        lev = [lv for lv, xs in levels.items() if x in xs][0]
        name = self.name(lev, x)
        d = Path(outdir) / lat / name
        d.mkdir(parents=True, exist_ok=True)
        el_a, el_b = self.species
        t0 = time.time()
        meta = {"lattice": lat, f"x_{el_b}": x, "level": lev, "generator": f"m_scraps {m_scraps.__version__}"}
        if x in (0.0, 1.0) and lat != "LIQUID":
            a, _ = self.prim(lat, x)
            a = a.copy()
            a.set_chemical_symbols([el_b if x == 1 else el_a] * len(a))
            meta.update(n_atoms=len(a), note="pure endmember, primitive cell (as ATAT sqsdb lev=0)")
        else:
            cfg = _setup[lat]
            a0, d1 = self.prim(lat, x, conventional=True)
            n = len(a0) * int(np.prod(cfg["cell_dim"]))
            if lat != "LIQUID":
                p0, _ = self.prim(lat, x)
                p_sup = find_optimal_cell_shape(p0.cell, n // len(p0), "fcc", verbose=False)
                a0 = make_supercell(p0, p_sup)
                cfg = dict(cfg, cell_dim=(1, 1, 1))
                meta["supercell_matrix"] = p_sup.tolist()
            nb = int(round(x * n))
            if x in (0.0, 1.0):
                a = a0.repeat(cfg["cell_dim"])
                a.set_chemical_symbols([el_b if x == 1 else el_a] * n)
                meta.update(n_atoms=n, note="pure liquid start (fcc supercell)")
            else:
                cut = [c * d1 for c in cfg["cut"]]
                wd = Path(workroot) / lat / name  # scratch (m_scraps writes its working files here), kept out of the output
                wd.mkdir(parents=True, exist_ok=True)
                a = m_scraps.run_sqs_from_atoms(
                    a0,
                    species=self.species,
                    element_counts=[n - nb, nb],
                    cell_dim=cfg["cell_dim"],
                    figure_cutoffs=cut,
                    figure_objective="weighted_l1",
                    figure_wd=0.0,
                    figure_wn=1.0,
                    auto_budget=2,
                    rng_seed=self.seed,
                    ranks=self.ranks,
                    work_dir=str(wd),
                )
                meta.update(
                    n_atoms=len(a),
                    counts=[n - nb, nb],
                    cell_dim=cfg["cell_dim"],
                    figure_cutoffs_A=cut,
                    figure_cutoffs_nn=cfg["cut"],
                    objective="weighted_l1 (wd=0, wn=1)",
                    auto_budget=2,
                    rng_seed=self.seed,
                    final_fitness=a.info.get("scraps_final_fitness"),
                )
        a.pbc = True
        write_str_out(a, d / "str.out")
        (d / "lat.in").write_text(self.lat_in(lat))
        a.write(d / "POSCAR", format="vasp", direct=True, sort=True)
        (d / "wait").write_text("")
        meta["seconds"] = round(time.time() - t0, 2)
        (d / "sqs_meta.json").write_text(json.dumps(meta, indent=1))
        return d, meta


def write_str_out(atoms: object, path: str | Path) -> None:
    """Write the str.out file of the SQS generator: unit vectors, cell and atoms in Cartesian coordinates.

    Args:
        atoms (object): ASE Atoms of the structure.
        path (str | Path): Output file.
    """
    with Path(path).open("w") as f:
        for r in np.eye(3):
            f.write(f"{r[0]:.6f} {r[1]:.6f} {r[2]:.6f}\n")
        for r in atoms.cell:
            f.write(f"{r[0]:.8f} {r[1]:.8f} {r[2]:.8f}\n")
        for p, s in zip(atoms.positions, atoms.get_chemical_symbols(), strict=True):
            f.write(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {s}\n")
