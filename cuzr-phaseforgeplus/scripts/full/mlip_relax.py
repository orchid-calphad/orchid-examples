# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/mlip_relax.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""PhaseForge `runstruct_MLIP` equivalent (MLIPrelax + extract_MLIP) without ATAT's runstruct_vasp.

For each directory containing POSCAR and `wait`: full relaxation (positions + cell, ASE FrechetCellFilter, FIRE,
fmax = 0.001 eV/A, max 1000 steps; PhaseForge MLIPrelax settings) with the MaterialsFramework calculator used by
PhaseForge (GraceCalculator, model GRACE-2L-OMAT). Writes, as extract_MLIP does: energy (eV/cell), CONTCAR,
str_relax.out, force.out, stress.out; plus relax.json (energy per atom, steps, converged, volume, relaxation strain).
The `wait` marker is removed on success. Usage: python mlip_relax.py <dir> [<dir> ...]
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import ROOT, SGTE_TDB, orchid  # noqa: E402,F401
import json, sys, time
from pathlib import Path
import numpy as np
from pymatgen.core import Structure
from materialsframework.calculators import GraceCalculator

MODEL = "GRACE-2L-OMAT"


def strain_measure(c0, c1):
    """ATAT-checkrelax-like measure: norm of the volume-normalised Green-Lagrange strain from the input to the relaxed cell."""
    F = np.linalg.solve(c0, c1).T
    F = F / np.cbrt(np.linalg.det(F))
    E = 0.5 * (F.T @ F - np.eye(3))
    return float(np.linalg.norm(E))


def run(d, calc):
    d = Path(d)
    s = Structure.from_file(d / "POSCAR")
    t0 = time.time()
    r = calc.relax(s)
    fs = r["final_structure"]
    e = float(r["energy"])
    (d / "energy").write_text("%.8f\n" % e)
    fs.to(filename=str(d / "CONTCAR"), fmt="poscar")
    with open(d / "str_relax.out", "w") as f:
        for row in fs.lattice.matrix: f.write("%.8f %.8f %.8f\n" % tuple(row))
        f.write("1 0 0\n0 1 0\n0 0 1\n")
        for site in fs: f.write("%.8f %.8f %.8f %s\n" % (*site.frac_coords, site.species_string))
    np.savetxt(d / "force.out", np.asarray(r["forces"]), fmt="%.7e")
    st = np.asarray(r["stress"]).ravel()
    if st.size == 6:
        st = np.array([[st[0], st[5], st[4]], [st[5], st[1], st[3]], [st[4], st[3], st[2]]])
    np.savetxt(d / "stress.out", st.reshape(3, 3), fmt="%.7e")
    info = {"energy_eV_cell": e, "n_atoms": len(fs), "energy_eV_atom": e / len(fs), "formula": fs.composition.formula,
            "volume_A3_atom": fs.volume / len(fs), "volume0_A3_atom": s.volume / len(s),
            "relax_strain": strain_measure(s.lattice.matrix, fs.lattice.matrix), "converged": bool(calc.converged),
            "max_force": float(np.abs(np.asarray(r["forces"])).max()), "seconds": round(time.time() - t0, 1),
            "sg_relaxed": fs.get_space_group_info(symprec=0.05), "model": MODEL, "fmax": calc.fmax, "steps_max": calc.steps}
    (d / "relax.json").write_text(json.dumps(info, indent=1))
    if (d / "wait").exists(): (d / "wait").unlink()
    print(d, "%.5f eV/atom" % info["energy_eV_atom"], "conv", info["converged"], "strain %.3f" % info["relax_strain"], info["sg_relaxed"], info["seconds"], "s", flush=True)


if __name__ == "__main__":
    calc = GraceCalculator(model=MODEL, fmax=0.001, steps=1000)
    for d in sys.argv[1:]:
        if (Path(d) / "relax.json").exists():
            continue
        run(d, calc)
