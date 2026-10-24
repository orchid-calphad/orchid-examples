#!/usr/bin/env python
"""Prepare the PhaseForge input structure set of a binary system on a CPU machine (no GPU, no network).

The set is the one of the one-shot Cu-Zr benchmark (orchid-phaseforge-demo, orchid-bench-pd bundles):
  * endmembers   : FCC_A1, BCC_A2, HCP_A3 pure elements (PhaseForge level 0, primitive cell)
  * SQS          : FCC_A1, BCC_A2, HCP_A3 at x = 0.25, 0.5, 0.75 (64 atoms, m_scraps, level 1 and 2)
  * liquid starts: 108-atom fcc SQS at x = 0, 0.25, 0.5, 0.75, 1 (start of the liquid MD)
  * compounds    : the line compounds of the ORCHID phase registry (assigned structure type), with starting
                   coordinates taken, in order of preference, from
                     1. --curated DIR/<PHASE>.vasp           (hand-built or literature structures)
                     2. the Materials Project structure of the matching ORCHID DFT-hull vertex
                        (same composition and space group as the registry phase), read from a local file
                     3. the pymatgen AFLOW prototype library (same Pearson symbol, space group and composition)
                     4. last resort: the hull vertex of the right composition in another space group (flagged)
                   A compound for which none of the three gives a structure is listed under "skipped" in
                   manifest.json with the reason; nothing is invented.
The MLIP relaxes every structure afterwards, so starting coordinates fix only the topology.

The SQS generator is the code of orchid-phaseforge-demo/scripts/sqs_mscraps.py (same cells, cutoffs, objective, seed and
MPI ranks), so Cu-Zr reproduces the shipped structures bit for bit. Output layout is that of the Vision/Spark bundles'
inputs/structures: <LAT>/sqs_lev=<l>_a_<A>=<..>,a_<B>=<..>/{POSCAR,str.out,lat.in,wait,sqs_meta.json},
COMPOUNDS/<PHASE>/POSCAR, relax_list.txt, plus manifest.json (timings, versions, provenance).

Usage:
  python make_structures.py Cu-Zr --out out/CuZr [--orchid-data ~/projects/orchid-data] [--curated DIR]
"""
import argparse, csv, gzip, json, os, platform, re, subprocess, sys, time
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.build import bulk, find_optimal_cell_shape, make_supercell
from ase.io import read as ase_read
import m_scraps

HERE = Path(__file__).resolve().parent
CA = np.sqrt(8 / 3)
LEVELS = {0: [0.0, 1.0], 1: [0.5], 2: [0.25, 0.75]}
SOLID = ["FCC_A1", "BCC_A2", "HCP_A3"]
# supercell (of the conventional cell) and figure cutoffs in units of the nearest-neighbour distance (as the demo)
SETUP = {
    "FCC_A1": dict(cell_dim=(2, 2, 4), cut=(2.05, 1.45, 1.05)),
    "BCC_A2": dict(cell_dim=(4, 4, 2), cut=(2.05, 1.20, 1.05)),
    "HCP_A3": dict(cell_dim=(4, 2, 4), cut=(2.05, 1.45, 1.05)),
    "LIQUID": dict(cell_dim=(3, 3, 3), cut=(1.45, 1.05, 1.05)),
}
# Elemental room-temperature volumes (A^3/atom), Vegard start geometry only. The demo's Cu and Zr values are pinned so that
# Cu-Zr reproduces; other elements use pymatgen's molar volume rounded to 0.01 A^3.
V_PINNED = {"Cu": 11.81, "Zr": 23.27}


def elem_volume(el):
    if el in V_PINNED:
        return V_PINNED[el]
    from pymatgen.core import Element
    mv = Element(el).molar_volume
    if mv is None:
        raise SystemExit("no elemental volume for %s; add it to V_PINNED" % el)
    return round(float(mv) / 0.602214076, 2)


class SQSMaker:
    def __init__(self, A, B, seed, ranks):
        self.SP = [A, B]; self.V = {A: elem_volume(A), B: elem_volume(B)}; self.seed = seed; self.ranks = ranks

    def vol(self, x):
        return (1 - x) * self.V[self.SP[0]] + x * self.V[self.SP[1]]

    def prim(self, lat, x, conventional=False):
        v = self.vol(x)
        if lat in ("FCC_A1", "LIQUID"):
            a = (4 * v) ** (1 / 3)
            return bulk(self.SP[0], "fcc", a=a, cubic=conventional), a / np.sqrt(2)
        if lat == "BCC_A2":
            a = (2 * v) ** (1 / 3)
            return bulk(self.SP[0], "bcc", a=a, cubic=conventional), a * np.sqrt(3) / 2
        if lat == "HCP_A3":
            a = (2 * v / (np.sqrt(3) / 2 * CA)) ** (1 / 3)
            return bulk(self.SP[0], "hcp", a=a, c=CA * a), a
        raise ValueError(lat)

    def lat_in(self, lat):
        s = ",".join(self.SP)
        return {"FCC_A1": "1 1 1 90 90 90\n0 0.5 0.5\n0.5 0 0.5\n0.5 0.5 0\n0 0 0 %s\n" % s,
                "LIQUID": "1 1 1 90 90 90\n0 0.5 0.5\n0.5 0 0.5\n0.5 0.5 0\n0 0 0 %s\n" % s,
                "BCC_A2": "1 1 1 90 90 90\n-0.5 0.5 0.5\n0.5 -0.5 0.5\n0.5 0.5 -0.5\n0 0 0 %s\n" % s,
                "HCP_A3": "1 1 %.6f 90 90 120\n1 0 0\n0 1 0\n0 0 1\n0 0 0 %s\n0.333333 0.666667 0.5 %s\n" % (CA, s, s)}[lat]

    def name(self, lev, x):
        return "sqs_lev=%d_" % lev + ",".join("a_%s=%s" % (s, "%g" % c) for s, c in zip(self.SP, (1 - x, x)) if c > 0)

    def make(self, lat, x, outdir, workroot):
        lev = [l for l, xs in LEVELS.items() if x in xs][0]
        name = self.name(lev, x)
        d = Path(outdir) / lat / name
        d.mkdir(parents=True, exist_ok=True)
        A, B = self.SP
        t0 = time.time()
        meta = {"lattice": lat, "x_%s" % B: x, "level": lev, "generator": "m_scraps %s" % m_scraps.__version__}
        if x in (0.0, 1.0) and lat != "LIQUID":
            a, d1 = self.prim(lat, x)
            a = a.copy(); a.set_chemical_symbols([B if x == 1 else A] * len(a))
            meta.update(n_atoms=len(a), note="pure endmember, primitive cell (as ATAT sqsdb lev=0)")
        else:
            cfg = SETUP[lat]
            a0, d1 = self.prim(lat, x, conventional=True)
            n = len(a0) * int(np.prod(cfg["cell_dim"]))
            if lat != "LIQUID":
                p0, _ = self.prim(lat, x)
                P = find_optimal_cell_shape(p0.cell, n // len(p0), "fcc", verbose=False)
                a0 = make_supercell(p0, P); cfg = dict(cfg, cell_dim=(1, 1, 1)); meta["supercell_matrix"] = P.tolist()
            nb = int(round(x * n))
            if x in (0.0, 1.0):
                a = a0.repeat(cfg["cell_dim"]); a.set_chemical_symbols([B if x == 1 else A] * n)
                meta.update(n_atoms=n, note="pure liquid start (fcc supercell)")
            else:
                cut = [c * d1 for c in cfg["cut"]]
                wd = Path(workroot) / lat / name          # scratch (m_scraps writes its working files here), kept out of the output
                wd.mkdir(parents=True, exist_ok=True)
                a = m_scraps.run_sqs_from_atoms(a0, species=self.SP, element_counts=[n - nb, nb], cell_dim=cfg["cell_dim"],
                                                figure_cutoffs=cut, figure_objective="weighted_l1", figure_wd=0.0, figure_wn=1.0,
                                                auto_budget=2, rng_seed=self.seed, ranks=self.ranks, work_dir=str(wd))
                meta.update(n_atoms=len(a), counts=[n - nb, nb], cell_dim=cfg["cell_dim"], figure_cutoffs_A=cut,
                            figure_cutoffs_nn=cfg["cut"], objective="weighted_l1 (wd=0, wn=1)", auto_budget=2,
                            rng_seed=self.seed, final_fitness=a.info.get("scraps_final_fitness"))
        a.pbc = True
        write_str_out(a, d / "str.out")
        (d / "lat.in").write_text(self.lat_in(lat))
        a.write(d / "POSCAR", format="vasp", direct=True, sort=True)
        (d / "wait").write_text("")
        meta["seconds"] = round(time.time() - t0, 2)
        (d / "sqs_meta.json").write_text(json.dumps(meta, indent=1))
        return d, meta


def write_str_out(atoms, path):
    with open(path, "w") as f:
        for r in np.eye(3):
            f.write("%.6f %.6f %.6f\n" % tuple(r))
        for r in atoms.cell:
            f.write("%.8f %.8f %.8f\n" % tuple(r))
        for p, s in zip(atoms.positions, atoms.get_chemical_symbols()):
            f.write("%.8f %.8f %.8f %s\n" % (p[0], p[1], p[2], s))


# ---------------------------------------------------------------- registry and compounds
def registry_rows(odata, sysid):
    with open(Path(odata) / "phases/registry.csv", newline="") as f:
        return [r for r in csv.DictReader(f) if r["system"] == sysid]


def ideal_composition(row, A, B):
    """Stoichiometry of a registry phase as a dict {el: fraction}, or None for a solution/liquid phase.
    Line compounds carry `stoichiometry`; ordered sublattice phases (D0_24, B2, B19) take the species marked '%'
    (the majority species of the model) on each sublattice, weighted by the site ratios of the PHASE line."""
    s = row["stoichiometry"].strip()
    if s:
        d = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in s.split(";")}
        return {k.capitalize(): v for k, v in d.items()}
    name = row["preferred_name"]
    if name in ("FCC_A1", "BCC_A2", "HCP_A3", "LIQUID", "B2_BCC") or row["structure_type"] == "st:liquid" or not row["structure_type"]:
        return None
    ph, cn = row["tdb_phase_line"].split(), row["tdb_constituent_line"]
    try:
        toks = re.findall(r"[-+0-9.eE]+", " ".join(ph[3:ph.index("!")]))
        nsub = int(float(toks[0])); sites = [float(t) for t in toks[1:1 + nsub]]
        subs = cn.split(":")[1:1 + nsub]
        comp = {}
        for site, sub in zip(sites, subs):
            sp = sub.replace("!", "").split()
            if "VA" in sp and len(sp) == 1:
                continue
            sp = [t for t in sp if t != "VA"]
            maj = [t.rstrip("%") for t in sp if t.endswith("%")] or ([sp[0]] if len(sp) == 1 else [])
            if len(maj) != 1:
                return None
            comp[maj[0].capitalize()] = comp.get(maj[0].capitalize(), 0) + site
        tot = sum(comp.values())
        return {k: v / tot for k, v in comp.items()} if set(comp) == {A, B} else None
    except Exception:
        return None


def load_mp(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    return d["items"] if isinstance(d, dict) else d


def hull_vertices(odata, A, B):
    p = Path(odata) / "hulls/dft-mp" / A.upper() / ("%s-%s.json" % (A.upper(), B.upper()))
    if not p.exists():
        return {}
    h = json.loads(p.read_text())
    v = {}
    for iso in h["isotherms"][:1]:                     # the 0 K hull
        for iv in iso["intervals"]:
            for side in ("left", "right"):
                v[iv[side]["phase"]] = iv[side]["x"]
    return v


def sg_number(struct, symprec=0.1):
    import spglib
    cell = (struct.lattice.matrix, struct.frac_coords, [s.specie.Z for s in struct])
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    return ds.number if ds else None


def structure_from_mp(row, x_target, B, mp_items, hull_v, A, other_sg=False):
    from pymatgen.core import Structure
    sgn = int(row["space_group_number"]) if row["space_group_number"] else None
    best = {True: None, False: None}                     # keyed by "space group equals the registry's"
    for e in mp_items:
        c = e.get("comp", {})
        if set(c) != {A, B} or abs(c[B] / sum(c.values()) - x_target) > 2e-3:
            continue
        s = Structure.from_dict(e["structure"])
        same = (sgn is None) or sg_number(s) == sgn
        if not same and (not other_sg or e["id"] not in hull_v):   # a different space group: last resort, hull vertices only
            continue
        key = (e["id"] not in hull_v, e.get("dft_ef_eV", 0.0))        # hull vertices first, then lowest formation energy
        if best[same] is None or key < best[same][0]:
            best[same] = (key, e, s)
    pick = best[True] or (best[False] if other_sg else None)
    if pick is None:
        return None
    key, e, s = pick
    note = "" if best[True] else "; space group %s differs from the registry's %s (ordered sub/supergroup variant)" % (sg_number(s), sgn)
    return s, {"source": "Materials Project structure %s (ORCHID DFT hull %s)%s" % (e["id"], "entry, not a vertex" if key[0] else "vertex", note),
               "material_id": e["id"], "nsites": e.get("nsites")}


def structure_from_aflow(row, comp, A, B):
    import gzip as gz
    import pymatgen.analysis.prototypes as pp
    from pymatgen.core import Structure
    f = Path(pp.__file__).parent / "aflow_prototypes.json.gz"
    protos = json.load(gz.open(f))
    pearson, sgn = row["pearson"], row["space_group_number"]
    xB = comp[B]
    out = []
    for p in protos:
        t = p["tags"]
        lab = t["aflow"].split("_")
        if lab[1] != pearson or lab[2] != sgn:
            continue
        s = Structure.from_dict(p["snl"])
        els = sorted({sp.symbol for sp in s.species})
        if len(els) != 2:
            continue
        n = {e: sum(1 for sp in s.species if sp.symbol == e) for e in els}
        for m in ((els[0], els[1]), (els[1], els[0])):                    # try both decorations
            nB = n[m[1]] / sum(n.values())
            if abs(nB - xB) < 2e-3:
                out.append((s, m, t, n[els[0]] == n[els[1]]))
                break
    if not out:
        return None
    s, m, t, amb = out[0]
    s = s.copy(); s.replace_species({m[0]: A, m[1]: B})
    v = (1 - xB) * elem_volume(A) + xB * elem_volume(B)
    s.scale_lattice(v * len(s))
    return s, {"source": "pymatgen AFLOW prototype %s (%s), cell scaled to Vegard volume" % (t["aflow"], t.get("mineral") or t.get("strukturbericht") or ""),
               "decoration_ambiguous": bool(amb)}


def compounds(args, A, B, rows, outdir):
    from pymatgen.core import Structure
    mp_items = load_mp(args.mp_structures) if args.mp_structures and Path(args.mp_structures).exists() else []
    hv = hull_vertices(args.orchid_data, A, B)
    cur = Path(args.curated) if args.curated else None
    made, skipped = [], []
    cdir = Path(outdir) / "COMPOUNDS"
    for r in rows:
        comp = ideal_composition(r, A, B)
        if comp is None or r["status"] != "assigned" or r["structure_type"] in ("", "st:a1-cu-cf4", "st:a2-w-ci2", "st:a3-mg-hp2"):
            continue
        name = r["preferred_name"]
        if name in args.exclude_compounds:
            continue
        t0 = time.time(); s = None; info = None
        if cur and (cur / (name + ".vasp")).exists():
            s = Structure.from_file(cur / (name + ".vasp")); info = {"source": "curated file %s" % (cur / (name + ".vasp"))}
        if s is None and mp_items:
            res = structure_from_mp(r, comp[B], B, mp_items, hv, A)
            if res: s, info = res
        if s is None:
            res = structure_from_aflow(r, comp, A, B)
            if res: s, info = res
        if s is None and mp_items:                       # last resort: hull vertex of the right composition, other space group
            res = structure_from_mp(r, comp[B], B, mp_items, hv, A, other_sg=True)
            if res: s, info = res
        if s is None:
            skipped.append({"phase": name, "structure_type": r["structure_type"], "pearson": r["pearson"], "space_group": r["space_group_symbol"],
                            "x_%s" % B: round(comp[B], 6), "reason": "no curated file, no matching Materials Project hull structure (same composition and space group), no AFLOW prototype"})
            continue
        s = s.get_primitive_structure(tolerance=0.01)
        nB = sum(1 for sp in s.species if sp.symbol == B)
        d = cdir / name; d.mkdir(parents=True, exist_ok=True)
        s.to(filename=str(d / "POSCAR"), fmt="poscar")
        made.append(dict(info, phase=name, n_atoms=len(s), x_target=round(comp[B], 6), x_structure=round(nB / len(s), 6),
                         composition_matches=abs(nB / len(s) - comp[B]) < 2e-3,
                         registry_space_group=r["space_group_symbol"], detected_space_group_number=sg_number(s), seconds=round(time.time() - t0, 2)))
    return made, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system", help="binary system, e.g. Cu-Zr")
    ap.add_argument("--out", required=True, help="output directory (created)")
    ap.add_argument("--orchid-data", default=os.environ.get("ORCHID_DATA", str(Path.home() / "projects/orchid-data")))
    ap.add_argument("--mp-structures", default=str(HERE.parents[1] / "mlip-formation-enthalpy/inputs/structures.json.gz"),
                    help="local file of Materials Project structures keyed like ORCHID's hulls (default: the formation-enthalpy example's)")
    ap.add_argument("--curated", help="directory of <PHASE>.vasp files that override the automatic compound structures")
    ap.add_argument("--exclude-compounds", nargs="*", default=[], metavar="PHASE", help="registry phase names to leave out (the Cu-Zr set of the bundles omits CUZR_B33)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ranks", type=int, default=4, help="MPI ranks of m_scraps (4 reproduces the shipped Cu-Zr set)")
    ap.add_argument("--work", help="scratch directory for m_scraps files (default: <out>/../.make_structures_work)")
    ap.add_argument("--no-sqs", action="store_true"); ap.add_argument("--no-compounds", action="store_true")
    ap.add_argument("--no-liquid", action="store_true")
    args = ap.parse_args()

    els = sorted(e.strip().capitalize() for e in re.split(r"[-_ ,]", args.system) if e.strip())
    if len(els) != 2:
        sys.exit("need a binary system, e.g. Cu-Zr")
    A, B = els
    sysid = "%s-%s" % (A.upper(), B.upper())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work) if args.work else out.resolve().parent / (".make_structures_work_" + sysid)
    rows = registry_rows(args.orchid_data, sysid)
    if not rows:
        sys.exit("system %s not in %s/phases/registry.csv" % (sysid, args.orchid_data))
    have = {r["preferred_name"] for r in rows}
    lats = [l for l in SOLID if l in have] + ([] if args.no_liquid or "LIQUID" not in have else ["LIQUID"])
    for l in SOLID:
        if l not in have:
            print("warning: %s not in the registry for %s; skipped" % (l, sysid))
    T0 = time.time(); timings = []
    sqs_meta = []
    mk = SQSMaker(A, B, args.seed, args.ranks)
    if not args.no_sqs:
        for lat in lats:
            for lev in (0, 1, 2):
                for x in LEVELS[lev]:
                    t = time.time()
                    d, meta = mk.make(lat, x, out, work)
                    timings.append({"item": "%s/%s" % (lat, d.name), "seconds": round(time.time() - t, 2), "n_atoms": meta["n_atoms"]})
                    sqs_meta.append(meta)
                    print("%-8s %-36s %4d atoms %7.2f s" % (lat, d.name, meta["n_atoms"], timings[-1]["seconds"]), flush=True)
    made, skipped = ([], [])
    if not args.no_compounds:
        tc = time.time()
        made, skipped = compounds(args, A, B, rows, out)
        timings.append({"item": "COMPOUNDS (%d made, %d skipped)" % (len(made), len(skipped)), "seconds": round(time.time() - tc, 2)})
        for m in made:
            print("compound %-12s %3d atoms x=%.4f  %s" % (m["phase"], m["n_atoms"], m["x_structure"], m["source"]))
        for s in skipped:
            print("SKIPPED  %-12s %s" % (s["phase"], s["reason"]))
    # relax list: solid endmembers, compounds, SQS (bundle order)
    rl = []
    solids = sorted(l for l in lats if l != "LIQUID")
    rl += ["%s/%s" % (l, mk.name(0, x)) for l in solids for x in (0.0, 1.0)] if not args.no_sqs else []
    rl += ["COMPOUNDS/%s" % m["phase"] for m in sorted(made, key=lambda m: m["phase"])]
    rl += ["%s/%s" % (l, mk.name(lev, x)) for l in solids for lev in (1, 2) for x in LEVELS[lev]] if not args.no_sqs else []
    (out / "relax_list.txt").write_text("\n".join(rl) + "\n")
    total = time.time() - T0
    try:
        mscommit = subprocess.run(["git", "-C", str(Path(m_scraps.__file__).resolve().parents[2]), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        mscommit = ""
    manifest = {"system": sysid, "elements": [A, B], "wall_seconds_total": round(total, 2), "cpu_count": os.cpu_count(),
                "platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0],
                "m_scraps": m_scraps.__version__, "m_scraps_commit_if_source_tree": mscommit, "seed": args.seed, "mpi_ranks": args.ranks,
                "orchid_data": str(args.orchid_data), "orchid_data_commit": subprocess.run(["git", "-C", str(args.orchid_data), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                "volumes_A3_per_atom": {A: elem_volume(A), B: elem_volume(B)},
                "n_relaxations": len(rl), "n_liquid": len(LEVELS[0]) + len(LEVELS[1]) + len(LEVELS[2]) if "LIQUID" in lats else 0,
                "timings": timings, "compounds": made, "skipped": skipped}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("total %.1f s on %d CPUs; %d relaxations listed; manifest.json written" % (total, os.cpu_count(), len(rl)))


if __name__ == "__main__":
    main()
