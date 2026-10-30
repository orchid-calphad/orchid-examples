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
  python make_structures.py Cu-Zr --out out/CuZr [--orchid-data $ORCHID_DATA] [--curated DIR]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import m_scraps
import spglib
from pymatgen.analysis import prototypes as pp
from pymatgen.core import Structure

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sqs_structures import SQSMaker, elem_volume, levels, solid_lattices

here = Path(__file__).resolve().parent


# ---------------------------------------------------------------- registry and compounds
def registry_rows(odata: str | Path, sysid: str) -> list[dict]:
    """Rows of the ORCHID phase registry that belong to one system.

    Args:
        odata (str | Path): Root of the orchid-data checkout.
        sysid (str): System identifier such as CU-ZR.

    Returns:
        List of registry rows (dicts keyed by the CSV header).
    """
    with (Path(odata) / "phases/registry.csv").open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["system"] == sysid]


def ideal_composition(row: dict, el_a: str, el_b: str) -> dict | None:
    """Stoichiometry of a registry phase as a dict {el: fraction}, or None for a solution/liquid phase.

    Line compounds carry `stoichiometry`; ordered sublattice phases (D0_24, B2, B19) take the species marked '%'
    (the majority species of the model) on each sublattice, weighted by the site ratios of the PHASE line.

    Args:
        row (dict): Registry row of the phase.
        el_a (str): First element symbol.
        el_b (str): Second element symbol.

    Returns:
        Mole fraction per element, or None if the phase is not a stoichiometric compound of exactly these two elements.
    """
    s = row["stoichiometry"].strip()
    if s:
        d = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in s.split(";")}
        return {k.capitalize(): v for k, v in d.items()}
    name = row["preferred_name"]
    if (
        name in ("FCC_A1", "BCC_A2", "HCP_A3", "LIQUID", "B2_BCC")
        or row["structure_type"] == "st:liquid"
        or not row["structure_type"]
    ):
        return None
    ph, cn = row["tdb_phase_line"].split(), row["tdb_constituent_line"]
    try:
        toks = re.findall(r"[-+0-9.eE]+", " ".join(ph[3 : ph.index("!")]))
        nsub = int(float(toks[0]))
        sites = [float(t) for t in toks[1 : 1 + nsub]]
        subs = cn.split(":")[1 : 1 + nsub]
        comp = {}
        # strict=False on purpose: a malformed constituent line that is shorter than the site list is truncated, not an error
        for site, sub in zip(sites, subs, strict=False):
            sp = sub.replace("!", "").split()
            if "VA" in sp and len(sp) == 1:
                continue
            sp = [t for t in sp if t != "VA"]
            maj = [t.rstrip("%") for t in sp if t.endswith("%")] or ([sp[0]] if len(sp) == 1 else [])
            if len(maj) != 1:
                return None
            comp[maj[0].capitalize()] = comp.get(maj[0].capitalize(), 0) + site
        tot = sum(comp.values())
        return {k: v / tot for k, v in comp.items()} if set(comp) == {el_a, el_b} else None
    except Exception:  # a malformed phase line is not a usable composition
        return None


def load_mp(path: str | Path) -> list[dict]:
    """Materials Project structures keyed like the ORCHID hulls, from a JSON or gzipped JSON file.

    Args:
        path (str | Path): Path of the .json or .json.gz file.

    Returns:
        List of entries (each with id, comp, structure and optional energy fields).
    """
    with gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else Path(path).open("rt", encoding="utf-8") as f:
        d = json.load(f)
    return d["items"] if isinstance(d, dict) else d


def hull_vertices(odata: str | Path, el_a: str, el_b: str) -> dict:
    """Phase ids of the 0 K hull vertices of a binary system and their mole fractions.

    Args:
        odata (str | Path): Root of the orchid-data checkout.
        el_a (str): First element symbol.
        el_b (str): Second element symbol.

    Returns:
        Dict phase id -> mole fraction of the second element; empty if the hull file does not exist.
    """
    p = Path(odata) / "hulls/dft-mp" / el_a.upper() / f"{el_a.upper()}-{el_b.upper()}.json"
    if not p.exists():
        return {}
    h = json.loads(p.read_text(encoding="utf-8"))
    v = {}
    for iso in h["isotherms"][:1]:  # the 0 K hull
        for iv in iso["intervals"]:
            for side in ("left", "right"):
                v[iv[side]["phase"]] = iv[side]["x"]
    return v


def sg_number(struct: Structure, symprec: float = 0.1) -> int | None:
    """Space-group number of a structure.

    Args:
        struct (Structure): pymatgen structure.
        symprec (float): Symmetry tolerance in A.

    Returns:
        The space-group number, or None if spglib finds no symmetry dataset.
    """
    cell = (struct.lattice.matrix, struct.frac_coords, [s.specie.Z for s in struct])
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec)
    return ds.number if ds else None


def structure_from_mp(
    row: dict,
    x_target: float,
    el_b: str,
    mp_items: list[dict],
    hull_v: dict,
    *,
    el_a: str,
    other_sg: bool = False,
) -> tuple[Structure, dict] | None:
    """Structure of a compound taken from the Materials Project entries that match its composition.

    Args:
        row (dict): Registry row of the phase.
        x_target (float): Mole fraction of el_b of the phase.
        el_b (str): Second element symbol.
        mp_items (list[dict]): Materials Project entries.
        hull_v (dict): Hull vertex phase ids (preferred over other entries).
        el_a (str): First element symbol.
        other_sg (bool): Also accept entries whose space group differs from the registry's, if they are hull vertices.

    Returns:
        Tuple (structure, provenance dict), or None if no entry matches.
    """
    sgn = int(row["space_group_number"]) if row["space_group_number"] else None
    best = {True: None, False: None}  # keyed by "space group equals the registry's"
    for e in mp_items:
        c = e.get("comp", {})
        if set(c) != {el_a, el_b} or abs(c[el_b] / sum(c.values()) - x_target) > 2e-3:
            continue
        s = Structure.from_dict(e["structure"])
        same = (sgn is None) or sg_number(s) == sgn
        if not same and (not other_sg or e["id"] not in hull_v):  # a different space group: last resort, hull vertices only
            continue
        key = (e["id"] not in hull_v, e.get("dft_ef_eV", 0.0))  # hull vertices first, then lowest formation energy
        if best[same] is None or key < best[same][0]:
            best[same] = (key, e, s)
    pick = best[True] or (best[False] if other_sg else None)
    if pick is None:
        return None
    key, e, s = pick
    note = (
        "" if best[True] else f"; space group {sg_number(s)} differs from the registry's {sgn} (ordered sub/supergroup variant)"
    )
    kind = "entry, not a vertex" if key[0] else "vertex"
    return s, {
        "source": f"Materials Project structure {e['id']} (ORCHID DFT hull {kind}){note}",
        "material_id": e["id"],
        "nsites": e.get("nsites"),
    }


def structure_from_aflow(row: dict, comp: dict, el_a: str, el_b: str) -> tuple[Structure, dict] | None:
    """Structure of a compound taken from the pymatgen AFLOW prototype library, scaled to the Vegard volume.

    Args:
        row (dict): Registry row of the phase (Pearson symbol and space-group number are used).
        comp (dict): Mole fraction per element of the phase.
        el_a (str): First element symbol.
        el_b (str): Second element symbol.

    Returns:
        Tuple (structure, provenance dict), or None if no prototype matches the composition.
    """
    f = Path(pp.__file__).parent / "aflow_prototypes.json.gz"
    with gzip.open(f) as fh:
        protos = json.load(fh)
    pearson, sgn = row["pearson"], row["space_group_number"]
    x_b = comp[el_b]
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
        for m in ((els[0], els[1]), (els[1], els[0])):  # try both decorations
            n_b = n[m[1]] / sum(n.values())
            if abs(n_b - x_b) < 2e-3:
                out.append((s, m, t, n[els[0]] == n[els[1]]))
                break
    if not out:
        return None
    s, m, t, amb = out[0]
    s = s.copy()
    s.replace_species({m[0]: el_a, m[1]: el_b})
    v = (1 - x_b) * elem_volume(el_a) + x_b * elem_volume(el_b)
    s.scale_lattice(v * len(s))
    label = t.get("mineral") or t.get("strukturbericht") or ""
    return s, {
        "source": f"pymatgen AFLOW prototype {t['aflow']} ({label}), cell scaled to Vegard volume",
        "decoration_ambiguous": bool(amb),
    }


def compounds(
    args: argparse.Namespace, el_a: str, el_b: str, rows: list[dict], outdir: str | Path
) -> tuple[list[dict], list[dict]]:
    """Write the POSCAR of every line compound that has a starting structure, and list the ones without.

    The source of each structure is tried in order: curated file, Materials Project hull structure of the same space
    group, AFLOW prototype, and finally a hull vertex in another space group.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        el_a (str): First element symbol.
        el_b (str): Second element symbol.
        rows (list[dict]): Registry rows of the system.
        outdir (str | Path): Output root; POSCARs go to <outdir>/COMPOUNDS/<PHASE>/POSCAR.

    Returns:
        Tuple (made, skipped): provenance dicts of the written compounds, and dicts with the reason for the skipped ones.
    """
    mp_items = load_mp(args.mp_structures) if args.mp_structures and Path(args.mp_structures).exists() else []
    hv = hull_vertices(args.orchid_data, el_a, el_b)
    cur = Path(args.curated) if args.curated else None
    made, skipped = [], []
    cdir = Path(outdir) / "COMPOUNDS"
    for r in rows:
        comp = ideal_composition(r, el_a, el_b)
        if (
            comp is None
            or r["status"] != "assigned"
            or r["structure_type"] in ("", "st:a1-cu-cf4", "st:a2-w-ci2", "st:a3-mg-hp2")
        ):
            continue
        name = r["preferred_name"]
        if name in args.exclude_compounds:
            continue
        t0 = time.time()
        s = None
        info = None
        if cur:
            curated = cur / f"{name}.vasp"
            if curated.exists():
                s = Structure.from_file(curated)
                info = {"source": f"curated file {curated}"}
        if s is None and mp_items:
            res = structure_from_mp(r, comp[el_b], el_b, mp_items, hv, el_a=el_a)
            if res:
                s, info = res
        if s is None:
            res = structure_from_aflow(r, comp, el_a, el_b)
            if res:
                s, info = res
        if s is None and mp_items:  # last resort: hull vertex of the right composition, other space group
            res = structure_from_mp(r, comp[el_b], el_b, mp_items, hv, el_a=el_a, other_sg=True)
            if res:
                s, info = res
        if s is None:
            skipped.append(
                {
                    "phase": name,
                    "structure_type": r["structure_type"],
                    "pearson": r["pearson"],
                    "space_group": r["space_group_symbol"],
                    f"x_{el_b}": round(comp[el_b], 6),
                    "reason": "no curated file, no matching Materials Project hull structure (same composition and space group), no AFLOW prototype",
                }
            )
            continue
        s = s.get_primitive_structure(tolerance=0.01)
        n_b = sum(1 for sp in s.species if sp.symbol == el_b)
        d = cdir / name
        d.mkdir(parents=True, exist_ok=True)
        s.to(filename=str(d / "POSCAR"), fmt="poscar")
        made.append(
            dict(
                info,
                phase=name,
                n_atoms=len(s),
                x_target=round(comp[el_b], 6),
                x_structure=round(n_b / len(s), 6),
                composition_matches=abs(n_b / len(s) - comp[el_b]) < 2e-3,
                registry_space_group=r["space_group_symbol"],
                detected_space_group_number=sg_number(s),
                seconds=round(time.time() - t0, 2),
            )
        )
    return made, skipped


def _parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        The parsed arguments.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system", help="binary system, e.g. Cu-Zr")
    ap.add_argument("--out", required=True, help="output directory (created)")
    ap.add_argument("--orchid-data", default=os.environ.get("ORCHID_DATA", str(Path.home() / "projects/orchid-data")))
    ap.add_argument(
        "--mp-structures",
        default=str(here.parents[1] / "mlip-formation-enthalpy/inputs/structures.json.gz"),
        help="local file of Materials Project structures keyed like ORCHID's hulls (default: the formation-enthalpy example's)",
    )
    ap.add_argument("--curated", help="directory of <PHASE>.vasp files that override the automatic compound structures")
    ap.add_argument(
        "--exclude-compounds",
        nargs="*",
        default=[],
        metavar="PHASE",
        help="registry phase names to leave out (the Cu-Zr set of the bundles omits CUZR_B33)",
    )
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ranks", type=int, default=4, help="MPI ranks of m_scraps (4 reproduces the shipped Cu-Zr set)")
    ap.add_argument("--work", help="scratch directory for m_scraps files (default: <out>/../.make_structures_work)")
    ap.add_argument("--no-sqs", action="store_true")
    ap.add_argument("--no-compounds", action="store_true")
    ap.add_argument("--no-liquid", action="store_true")
    return ap.parse_args()


def _sqs_stage(mk: SQSMaker, lats: list[str], out: Path, work: Path) -> list[dict]:
    """Write the endmember, SQS and liquid structures of every lattice.

    Args:
        mk (SQSMaker): Structure maker of the system.
        lats (list[str]): Lattice keys to build.
        out (Path): Output root.
        work (Path): Scratch root for m_scraps.

    Returns:
        Timing records, one per written structure.
    """
    timings = []
    for lat in lats:
        for lev in (0, 1, 2):
            for x in levels[lev]:
                t = time.time()
                d, meta = mk.make(lat, x, out, work)
                timings.append({"item": f"{lat}/{d.name}", "seconds": round(time.time() - t, 2), "n_atoms": meta["n_atoms"]})
                print(f"{lat:<8} {d.name:<36} {meta['n_atoms']:4d} atoms {timings[-1]['seconds']:7.2f} s", flush=True)
    return timings


def _relax_list(mk: SQSMaker, lats: list[str], made: list[dict], no_sqs: bool) -> list[str]:
    """Relaxation list in bundle order: endmembers, compounds, then SQS.

    Args:
        mk (SQSMaker): Structure maker, used for the directory names.
        lats (list[str]): Lattice keys that were built.
        made (list[dict]): Provenance dicts of the written compounds.
        no_sqs (bool): Whether the SQS structures were skipped.

    Returns:
        Relative paths of the structures to relax.
    """
    rl = []
    solids = sorted(lat for lat in lats if lat != "LIQUID")
    rl += [f"{lat}/{mk.name(0, x)}" for lat in solids for x in (0.0, 1.0)] if not no_sqs else []
    rl += [f"COMPOUNDS/{m['phase']}" for m in sorted(made, key=lambda m: m["phase"])]
    rl += [f"{lat}/{mk.name(lev, x)}" for lat in solids for lev in (1, 2) for x in levels[lev]] if not no_sqs else []
    return rl


def main() -> None:
    """Prepare the structure set of the system named on the command line and write manifest.json."""
    args = _parse_args()
    els = sorted(e.strip().capitalize() for e in re.split(r"[-_ ,]", args.system) if e.strip())
    if len(els) != 2:
        sys.exit("need a binary system, e.g. Cu-Zr")
    el_a, el_b = els
    sysid = f"{el_a.upper()}-{el_b.upper()}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work) if args.work else out.resolve().parent / (".make_structures_work_" + sysid)
    rows = registry_rows(args.orchid_data, sysid)
    if not rows:
        sys.exit(f"system {sysid} not in {args.orchid_data}/phases/registry.csv")
    have = {r["preferred_name"] for r in rows}
    lats = [lat for lat in solid_lattices if lat in have] + ([] if args.no_liquid or "LIQUID" not in have else ["LIQUID"])
    for lat in solid_lattices:
        if lat not in have:
            print(f"warning: {lat} not in the registry for {sysid}; skipped")
    t_start = time.time()
    timings = []
    mk = SQSMaker(el_a, el_b, args.seed, args.ranks)
    if not args.no_sqs:
        timings += _sqs_stage(mk, lats, out, work)
    made, skipped = [], []
    if not args.no_compounds:
        tc = time.time()
        made, skipped = compounds(args, el_a, el_b, rows, out)
        timings.append({"item": f"COMPOUNDS ({len(made)} made, {len(skipped)} skipped)", "seconds": round(time.time() - tc, 2)})
        for m in made:
            print(f"compound {m['phase']:<12} {m['n_atoms']:3d} atoms x={m['x_structure']:.4f}  {m['source']}")
        for s in skipped:
            print(f"SKIPPED  {s['phase']:<12} {s['reason']}")
    rl = _relax_list(mk, lats, made, args.no_sqs)
    (out / "relax_list.txt").write_text("\n".join(rl) + "\n", encoding="utf-8")
    total = time.time() - t_start
    try:
        mscommit = subprocess.run(
            ["git", "-C", str(Path(m_scraps.__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except Exception:  # no git, or no source tree of m_scraps: the commit is simply unknown
        mscommit = ""
    manifest = {
        "system": sysid,
        "elements": [el_a, el_b],
        "wall_seconds_total": round(total, 2),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "m_scraps": m_scraps.__version__,
        "m_scraps_commit_if_source_tree": mscommit,
        "seed": args.seed,
        "mpi_ranks": args.ranks,
        "orchid_data": str(args.orchid_data),
        "orchid_data_commit": subprocess.run(
            ["git", "-C", str(args.orchid_data), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "volumes_A3_per_atom": {el_a: elem_volume(el_a), el_b: elem_volume(el_b)},
        "n_relaxations": len(rl),
        "n_liquid": len(levels[0]) + len(levels[1]) + len(levels[2]) if "LIQUID" in lats else 0,
        "timings": timings,
        "compounds": made,
        "skipped": skipped,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"total {total:.1f} s on {os.cpu_count()} CPUs; {len(rl)} relaxations listed; manifest.json written")


if __name__ == "__main__":
    main()
