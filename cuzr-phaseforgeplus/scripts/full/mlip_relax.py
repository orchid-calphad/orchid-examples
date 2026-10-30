# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/mlip_relax.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""PhaseForge `runstruct_MLIP` equivalent (MLIPrelax + extract_MLIP) without ATAT's runstruct_vasp.

For each directory containing POSCAR and `wait`: full relaxation (positions + cell, ASE FrechetCellFilter, FIRE,
fmax = 0.001 eV/A, max 1000 steps; PhaseForge MLIPrelax settings) with the MaterialsFramework calculator used by
PhaseForge (GraceCalculator, model GRACE-2L-OMAT). Writes, as extract_MLIP does: energy (eV/cell), CONTCAR,
str_relax.out, force.out, stress.out; plus relax.json (energy per atom, steps, converged, volume, relaxation strain).
The `wait` marker is removed on success. Usage: python mlip_relax.py <dir> [<dir> ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import _paths  # noqa: E402,F401  side effect: work/ tree and cached TDB seeding, as before the restyle
import numpy as np
from materialsframework.calculators import GraceCalculator
from pymatgen.core import Structure

model_name = "GRACE-2L-OMAT"


def strain_measure(c0: np.ndarray, c1: np.ndarray) -> float:
    """ATAT-checkrelax-like measure: norm of the volume-normalised Green-Lagrange strain from the input to the relaxed cell.

    Args:
        c0 (np.ndarray): 3x3 lattice matrix of the input cell (Angstrom).
        c1 (np.ndarray): 3x3 lattice matrix of the relaxed cell (Angstrom).

    Returns:
        Frobenius norm of the volume-normalised Green-Lagrange strain (dimensionless).
    """
    defgrad = np.linalg.solve(c0, c1).T
    defgrad = defgrad / np.cbrt(np.linalg.det(defgrad))
    green = 0.5 * (defgrad.T @ defgrad - np.eye(3))
    return float(np.linalg.norm(green))


def run(d, calc) -> None:
    """Relax the structure in one directory and write the PhaseForge-style output files.

    Args:
        d (str | Path): Directory holding POSCAR and the `wait` marker.
        calc (GraceCalculator): Calculator configured with fmax and the step limit.

    Writes energy (eV/cell), CONTCAR, str_relax.out, force.out, stress.out and relax.json in ``d``, and removes the
    `wait` marker on success.
    """
    d = Path(d)
    s = Structure.from_file(d / "POSCAR")
    t0 = time.time()
    r = calc.relax(s)
    fs = r["final_structure"]
    e = float(r["energy"])
    (d / "energy").write_text(f"{e:.8f}\n")
    fs.to(filename=str(d / "CONTCAR"), fmt="poscar")
    with (d / "str_relax.out").open("w") as f:
        for row in fs.lattice.matrix:
            f.write(f"{row[0]:.8f} {row[1]:.8f} {row[2]:.8f}\n")
        f.write("1 0 0\n0 1 0\n0 0 1\n")
        for site in fs:
            fx, fy, fz = site.frac_coords
            f.write(f"{fx:.8f} {fy:.8f} {fz:.8f} {site.species_string}\n")
    np.savetxt(d / "force.out", np.asarray(r["forces"]), fmt="%.7e")
    st = np.asarray(r["stress"]).ravel()
    if st.size == 6:
        st = np.array([[st[0], st[5], st[4]], [st[5], st[1], st[3]], [st[4], st[3], st[2]]])
    np.savetxt(d / "stress.out", st.reshape(3, 3), fmt="%.7e")
    info = {
        "energy_eV_cell": e,
        "n_atoms": len(fs),
        "energy_eV_atom": e / len(fs),
        "formula": fs.composition.formula,
        "volume_A3_atom": fs.volume / len(fs),
        "volume0_A3_atom": s.volume / len(s),
        "relax_strain": strain_measure(s.lattice.matrix, fs.lattice.matrix),
        "converged": bool(calc.converged),
        "max_force": float(np.abs(np.asarray(r["forces"])).max()),
        "seconds": round(time.time() - t0, 1),
        "sg_relaxed": fs.get_space_group_info(symprec=0.05),
        "model": model_name,
        "fmax": calc.fmax,
        "steps_max": calc.steps,
    }
    (d / "relax.json").write_text(json.dumps(info, indent=1))
    if (d / "wait").exists():
        (d / "wait").unlink()
    print(
        d,
        f"{info['energy_eV_atom']:.5f} eV/atom",
        "conv",
        info["converged"],
        f"strain {info['relax_strain']:.3f}",
        info["sg_relaxed"],
        info["seconds"],
        "s",
        flush=True,
    )


if __name__ == "__main__":
    calc = GraceCalculator(model=model_name, fmax=0.001, steps=1000)
    for d in sys.argv[1:]:
        if (Path(d) / "relax.json").exists():
            continue
        run(d, calc)
