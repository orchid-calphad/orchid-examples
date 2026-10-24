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
import argparse, hashlib, json, os, platform, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))


def _js(o):
    return o.tolist() if hasattr(o, "tolist") else float(o) if hasattr(o, "__float__") else str(o)


def build_calc(name, device, fmax, steps):
    spec = json.load(open(os.path.join(HERE, "models.json")))[name]
    import materialsframework.calculators as C
    cls = getattr(C, spec["class"])
    kw = dict(spec["kwargs"])
    if spec["class"] != "GraceCalculator" and device:
        kw["device"] = device
    calc = cls(fmax=fmax, steps=steps, optimizer="FIRE", relax_cell=True, fix_symmetry=False, **kw)
    return calc, spec


def sg(structure):
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        return SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number()
    except Exception:
        return None


def env_info():
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "host": platform.node()}
    for m in ("torch", "tensorflow", "mace", "sevenn", "orb_models", "mattersim", "chgnet", "fairchem.core",
              "tensorpotential", "materialsframework", "ase", "pymatgen"):
        try:
            mod = __import__(m, fromlist=["__version__"])
            info[m] = getattr(mod, "__version__", "?")
        except Exception:
            pass
    try:
        import torch
        info["cuda"] = torch.cuda.is_available() and torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def main():
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
    import numpy as np
    from pymatgen.core import Structure
    items = json.load(open(a.inputs))["items"]
    # largest first within the shard order is irrelevant; shard by stable hash so shards are balanced and fixed
    items.sort(key=lambda it: it["id"])
    if a.ids_file:
        keep = {l.strip() for l in open(a.ids_file) if l.strip()}
        items = [it for it in items if it["id"] in keep]
    items = [it for i, it in enumerate(items) if i % a.nshards == a.shard]
    os.makedirs(a.out, exist_ok=True)
    outp = os.path.join(a.out, "%s.shard%03d-of-%03d.jsonl" % (a.model, a.shard, a.nshards))
    done = set()
    if os.path.exists(outp):
        for line in open(outp):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass  # a torn last line from a killed job is ignored and the item rerun
    todo = [it for it in items if it["id"] not in done]
    print("model %s shard %d/%d: %d items, %d done, %d to run" % (a.model, a.shard, a.nshards, len(items), len(done),
                                                                   len(todo)), flush=True)
    if not todo:
        return 0
    t0 = time.time()
    calc, spec = build_calc(a.model, a.device, a.fmax, a.steps)
    meta = {"meta": True, "model": a.model, "spec": spec, "fmax": a.fmax, "steps": a.steps, "env": env_info(),
            "inputs_sha256": hashlib.sha256(open(a.inputs, "rb").read()).hexdigest(), "t_load_s": None}
    fh = open(outp, "a")
    for k, it in enumerate(todo):
        rec = {"id": it["id"], "src": it["src"], "kind": it["kind"], "nsites": it["nsites"], "model": a.model}
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
            rec.update({"e_relax_eV_per_atom": e1, "converged": bool(calc.converged), "nsteps": nsteps,
                        "fmax_final": float(np.linalg.norm(ff, axis=1).max()) if ff is not None else None,
                        "stress_final_GPa": (np.asarray(fin["stress"]).tolist() if fin.get("stress") is not None
                                             else None),
                        "vol_ratio": s1.volume / s0.volume, "sg_in": sg(s0), "sg_out": sg(s1),
                        "t_relax_s": round(time.time() - tr, 3), "final_structure": s1.as_dict()})
            rec["exploded"] = bool(abs(rec["vol_ratio"] - 1) > 0.5 or not (-15 < e1 < 5)
                                   or e1 < rec["e_sp_eV_per_atom"] - 1.0)
            rec["status"] = "ok"
        except Exception as e:
            rec.update({"status": "error", "error": repr(e)[:500], "tb": traceback.format_exc()[-1500:]})
        if k == 0:
            meta["t_first_item_s"] = round(time.time() - t0, 2)
            fh.write(json.dumps(meta, default=_js) + "\n")
        fh.write(json.dumps(rec, default=_js) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        if k % 20 == 0:
            print("%s %d/%d %.1fs %s" % (a.model, k + 1, len(todo), time.time() - t0, rec.get("status")), flush=True)
    print("done %s shard %d in %.1fs" % (a.model, a.shard, time.time() - t0), flush=True)
    open(outp + ".done", "w").write("%d items\n" % len(items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
