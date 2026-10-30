#!/usr/bin/env python3
"""Relax (positions + cell) and single-point every input structure with one MLIP; append one JSON line per item.

  python scripts/run_mlip.py --model mace-mpa-0 --inputs data/inputs/structures.json --out results/raw \
      [--shard 0 --nshards 8] [--ids-file ids.txt] [--device cuda] [--fmax 0.02 --steps 1000]

Protocol (identical for every model): FIRE + FrechetCellFilter (MaterialsFramework BaseCalculator.relax), fmax
0.02 eV/A (forces and cell stress via the filter), at most 1000 steps, no symmetry constraint; then a single
point on the final structure. Also a single point on the DFT-relaxed input structure. Elemental ground states
are relaxed exactly like compounds. Resumable and idempotent: ids already present in the output file are skipped;
each line is flushed and fsynced. Exceptions are recorded (status "error"), never fatal.
Flags: converged (steps < max), exploded (|dV/V| > 0.5, or E/atom outside [-15, 5] eV, or relaxed energy more
than 1 eV/atom below the single point), symmetry change (spglib space group before/after, symprec 0.1).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

here = Path(__file__).resolve().parent


def _js(o: object) -> object:
    """Fallback JSON encoder for values json cannot serialise.

    Args:
        o (object): Value to encode, typically a numpy array or scalar.

    Returns:
        A list for arrays, a float for float-like values, otherwise the string form.
    """
    return o.tolist() if hasattr(o, "tolist") else float(o) if hasattr(o, "__float__") else str(o)


def build_calc(name: str, device: str | None, fmax: float, steps: int) -> tuple[object, dict]:
    """Build the MaterialsFramework calculator of one model.

    Args:
        name (str): Model key in models.json.
        device (str | None): Torch device, for example cuda; ignored for GRACE models and when empty.
        fmax (float): Force convergence criterion, eV/A.
        steps (int): Maximum number of relaxation steps.

    Returns:
        Tuple (calculator, model spec from models.json).
    """
    spec = json.loads((here / "models.json").read_text())[name]
    # lazy on purpose: the MLIP environment is needed only when there is work to do (a finished shard returns early)
    calculators = importlib.import_module("materialsframework.calculators")
    cls = getattr(calculators, spec["class"])
    kw = dict(spec["kwargs"])
    if spec["class"] != "GraceCalculator" and device:
        kw["device"] = device
    calc = cls(fmax=fmax, steps=steps, optimizer="FIRE", relax_cell=True, fix_symmetry=False, **kw)
    return calc, spec


def sg(structure: Structure) -> int | None:
    """Space-group number of a structure at symprec 0.1.

    Args:
        structure (Structure): Structure to analyse.

    Returns:
        The space-group number, or None if the analysis fails (a diagnostic must not stop the run).
    """
    try:
        return SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number()
    except Exception:  # symmetry label is diagnostic only
        return None


def env_info() -> dict:
    """Python, platform and installed package versions, recorded once per model.

    Returns:
        Dict of environment details; packages that are missing or fail to import are left out.
    """
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "host": platform.node()}
    for m in (
        "torch",
        "tensorflow",
        "mace",
        "sevenn",
        "orb_models",
        "mattersim",
        "chgnet",
        "fairchem.core",
        "tensorpotential",
        "materialsframework",
        "ase",
        "pymatgen",
    ):
        with contextlib.suppress(Exception):  # an optional package that is absent or broken is simply not recorded
            mod = __import__(m, fromlist=["__version__"])
            info[m] = getattr(mod, "__version__", "?")
    with contextlib.suppress(Exception):  # no torch, or no CUDA: the entry is left out
        torch = importlib.import_module("torch")
        info["cuda"] = torch.cuda.is_available() and torch.cuda.get_device_name(0)
    return info


def _done_ids(outp: Path) -> set:
    """Ids already present in an output file, so that a rerun skips them.

    Args:
        outp (Path): Output JSON-lines file; may not exist yet.

    Returns:
        Set of item ids found in the file.
    """
    done = set()
    if outp.exists():
        with outp.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:  # a torn last line from a killed job is ignored and the item rerun
                    continue
    return done


def _evaluate_item(calc: object, it: dict, model: str) -> dict:
    """Relax and single-point one input structure; a failure is recorded in the result, not raised.

    Args:
        calc (object): MaterialsFramework calculator from build_calc.
        it (dict): Input item with id, src, kind, nsites and a serialised structure.
        model (str): Model key, stored in the record.

    Returns:
        Result record: single-point and relaxed energies (eV/atom), forces, stress, volume ratio, space groups, timings,
        the final structure and the flags; with status "ok", or status "error" and the error text.
    """
    rec = {"id": it["id"], "src": it["src"], "kind": it["kind"], "nsites": it["nsites"], "model": model}
    try:
        s0 = Structure.from_dict(it["structure"])
        n = len(s0)
        ts = time.time()
        sp = calc.calculate(s0)
        rec["e_sp_eV_per_atom"] = float(sp["energy"]) / n
        f = sp.get("forces")
        rec["fmax_sp"] = float(np.abs(np.asarray(f)).max()) if f is not None else None
        rec["t_sp_s"] = round(time.time() - ts, 3)
        tr = time.time()
        r = calc.relax(s0)
        s1 = r["final_structure"]
        nsteps = len(r["trajectory"]) - 1  # observer frames (one per step + the final call)
        fin = calc.calculate(s1)
        e1 = float(fin["energy"]) / n
        ff = np.asarray(fin["forces"]) if fin.get("forces") is not None else None
        rec.update(
            {
                "e_relax_eV_per_atom": e1,
                "converged": bool(calc.converged),
                "nsteps": nsteps,
                "fmax_final": float(np.linalg.norm(ff, axis=1).max()) if ff is not None else None,
                "stress_final_GPa": (np.asarray(fin["stress"]).tolist() if fin.get("stress") is not None else None),
                "vol_ratio": s1.volume / s0.volume,
                "sg_in": sg(s0),
                "sg_out": sg(s1),
                "t_relax_s": round(time.time() - tr, 3),
                "final_structure": s1.as_dict(),
            }
        )
        rec["exploded"] = bool(abs(rec["vol_ratio"] - 1) > 0.5 or not (-15 < e1 < 5) or e1 < rec["e_sp_eV_per_atom"] - 1.0)
        rec["status"] = "ok"
    except Exception as e:  # every failure is recorded in the result (status "error"), never fatal
        rec.update({"status": "error", "error": repr(e)[:500], "tb": traceback.format_exc()[-1500:]})
    return rec


def main() -> int:
    """Run the relaxations of one shard of one model, appending results to its JSON-lines file.

    Returns:
        Process exit status, 0 on success.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--ids-file")
    ap.add_argument("--device", default=None)
    ap.add_argument("--fmax", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=1000)
    a = ap.parse_args()
    items = json.loads(Path(a.inputs).read_text())["items"]
    # sorted by id, then split by index modulo nshards: a fixed, reproducible partition of the items
    items.sort(key=lambda it: it["id"])
    if a.ids_file:
        with Path(a.ids_file).open() as f:
            keep = {line.strip() for line in f if line.strip()}
        items = [it for it in items if it["id"] in keep]
    items = [it for i, it in enumerate(items) if i % a.nshards == a.shard]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    outp = out_dir / f"{a.model}.shard{a.shard:03d}-of-{a.nshards:03d}.jsonl"
    done = _done_ids(outp)
    todo = [it for it in items if it["id"] not in done]
    print(f"model {a.model} shard {a.shard}/{a.nshards}: {len(items)} items, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        return 0
    t0 = time.time()
    calc, spec = build_calc(a.model, a.device, a.fmax, a.steps)
    meta = {
        "meta": True,
        "model": a.model,
        "spec": spec,
        "fmax": a.fmax,
        "steps": a.steps,
        "env": env_info(),
        "inputs_sha256": hashlib.sha256(Path(a.inputs).read_bytes()).hexdigest(),
        "t_load_s": None,
    }
    with outp.open("a") as fh:
        for k, it in enumerate(todo):
            rec = _evaluate_item(calc, it, a.model)
            if k == 0:
                meta["t_first_item_s"] = round(time.time() - t0, 2)
                fh.write(json.dumps(meta, default=_js) + "\n")
            fh.write(json.dumps(rec, default=_js) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            if k % 20 == 0:
                print(f"{a.model} {k + 1}/{len(todo)} {time.time() - t0:.1f}s {rec.get('status')}", flush=True)
    print(f"done {a.model} shard {a.shard} in {time.time() - t0:.1f}s", flush=True)
    outp.with_name(outp.name + ".done").write_text(f"{len(items)} items\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
