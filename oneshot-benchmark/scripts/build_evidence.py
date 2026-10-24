# Adapted from orchid-bench-pd scripts/build_evidence.py: release files are read at the pinned commit
# (release.py, common/orchid_data.py); the PhaseForge+ fit-set flags come from inputs/phaseforge_fit_split.json.
"""Build evidence.json: Cu-Zr (T, x_Zr, reference phase assemblage) points for the one-shot phase-diagram track.

Sources (read-only):
  (a) FIELD points: ORCHID Dataset 6, assessed phase assemblages (Okamoto handbook, CU-ZR-p371), released coordinates
      and phase labels, all Cu-Zr records (no confidence filter; `confidence`, `boundary_basis` kept per point).
  (b) BOUNDARY points: ORCHID literature ZPF tables (espei/experimental/CU-ZR/*zpf*) and invariant tables
      (espei/invariants/CU-ZR/*); the bundle holds espei_ready values only. One equilibrium (tie-line or invariant) =
      one ORCHID observation id; duplicates of the same observation id across TDB-named copies are kept once.
      Each vertex of the equilibrium becomes a point at (T, x_vertex) whose reference assemblage is the full phase set
      of the equilibrium. Vertex composition: printed x when given; for a vertex written null because the phase is a
      line compound in the source TDB, the compound's formula composition in that TDB; when a non-stoichiometric
      vertex is null, the dataset's __HYPERPLANE__ composition is used instead (one point). Vertices of one
      equilibrium closer than 0.001 in x are merged (congruent points).
      A one-phase literature value (none for Cu-Zr in this release) would be a field point.
Provenance per point: record / observation id, dataset file + SHA-256, reference, DOI(s), tags; and for the
PhaseForge+ final TDB, whether the point was in its fit set (data/a4d_F10) or the demo's hold-out set.

Usage: python build_evidence.py [-o evidence.json]
"""
import argparse, glob, gzip, hashlib, json, os, re, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_identity as pid

from release import RELEASE, ROOT, orchid_data, rel  # noqa: E402
ASPA_REL = "assessed-phase-assemblages/records.jsonl.gz"
TDBDIRS = [os.path.join(ROOT, "tdb"), os.path.join(ROOT, "tdb", "reference_panel")]
SPLIT = os.path.join(ROOT, "inputs", "phaseforge_fit_split.json")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ids_in(paths):
    s = set()
    for p in paths:
        t = open(p).read()
        s |= set(re.findall(r"ORC-[A-Z0-9]{10}", t)) | set(re.findall(r"aspa-[0-9a-f]{16}", t))
    return s


def stoich_x(tdb_path):
    """{PHASE: x_Zr} for phases that are line compounds (one constituent per non-VA sublattice) in a TDB."""
    from pycalphad import Database
    db = Database(tdb_path)
    out = {}
    for name, ph in db.phases.items():
        subl = [(r, {str(c) for c in cons} - {"VA"}) for r, cons in zip(ph.sublattices, ph.constituents)]
        subl = [(r, c) for r, c in subl if c]
        if all(len(c) == 1 for r, c in subl):
            tot = sum(float(r) for r, c in subl)
            out[name] = sum(float(r) for r, c in subl if "ZR" in c) / tot
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=os.path.join(ROOT, "inputs", "evidence.json"))
    a = ap.parse_args()
    reg = pid.Registry()
    split = json.load(open(SPLIT))
    fit_ids, hold_ids = set(split["fit_ids"]), set(split["holdout_ids"])
    dense_assemblages = {frozenset(a) for a in split["dense_grid_assemblages"]}
    ASPA = str(orchid_data.path(ASPA_REL, RELEASE))

    def split_flag(i, labels=None):
        if i in fit_ids:
            return "fit"
        if labels is not None and frozenset(labels) in dense_assemblages:
            return "fit(dense_assessed_field)"
        if i in hold_ids:
            return "holdout"
        return "not_used"

    points, mapping = [], {}

    def note(source_kind, name, reg_id, basis):
        mapping.setdefault((source_kind, name), {"source": source_kind, "name": name, "registry_id": reg_id,
                                                  "basis": basis, "class_physical": pid.bench_class(reg_id),
                                                  "class_strict": pid.bench_class(reg_id, "strict"), "n_points": 0})
        mapping[(source_kind, name)]["n_points"] += 1

    # (a) Dataset 6 field points
    for line in gzip.open(ASPA, "rt"):
        r = json.loads(line)
        if r["system"] != "CU-ZR":
            continue
        assert r["x_component"] == "Zr"
        labels, regs = [], []
        for ph in r["assemblage"]:
            rid, basis = reg.label(ph["label"], ph.get("phase_id"))
            note("handbook_label", ph["label"], rid, basis)
            labels.append(ph["label"]); regs.append(rid)
        points.append({
            "id": r["id"], "kind": "field", "source_class": "assessed_phase_diagram", "T_K": r["T_K"], "x_Zr": r["x"],
            "phase_labels": labels, "registry_ids": regs,
            "truth_physical": sorted({pid.bench_class(i) for i in regs}),
            "truth_strict": sorted({pid.bench_class(i, "strict") for i in regs}),
            "provenance": {"dataset": "ORCHID Dataset 6 (assessed-phase-assemblages)", "record_id": r["id"],
                           "diagram": r["diagram"], "field": r["field"], "point_method": r["point_method"],
                           "boundary_basis": r["boundary_basis"], "basis": r["basis"], "confidence": r["confidence"],
                           "review_state": r["review_state"], "flags": r["flags"],
                           "coordinate_uncertainty": r["coordinate_uncertainty"],
                           "positional_margin": r["positional_margin"],
                           "citations": [c.get("citation") for c in r["citations"]],
                           "dois": [c.get("doi") for c in r["citations"] if c.get("doi")]},
            "final_tdb_split": split_flag(r["id"], labels),
        })

    # (b) literature boundary points
    rels = [r for r in orchid_data.listdir("espei/experimental/CU-ZR", RELEASE) if "zpf" in r.rsplit("/", 1)[1]] + \
        orchid_data.listdir("espei/invariants/CU-ZR", RELEASE)
    seen, files = {}, [(r, str(orchid_data.path(r, RELEASE))) for r in rels]
    stoich_cache, dup = {}, 0
    for frel, f in files:
        d = json.load(open(f))
        if d.get("output") != "ZPF":
            continue
        tdbname = os.path.basename(f).split(".")[0] + ".tdb"
        if tdbname not in stoich_cache:
            stoich_cache[tdbname] = stoich_x(next(os.path.join(t, tdbname) for t in TDBDIRS if os.path.exists(os.path.join(t, tdbname))))
        sx = stoich_cache[tdbname]
        obs = d["orchid"]["observations"]
        assert len(obs) == len(d["values"]) == len(d["conditions"]["T"]), f
        for k, (T, val, o) in enumerate(zip(d["conditions"]["T"], d["values"], obs)):
            if o["id"] in seen:
                dup += 1
                continue
            seen[o["id"]] = f
            phases = [e for e in val if e[0] != "__HYPERPLANE__"]
            hyp = [e[2][0] for e in val if e[0] == "__HYPERPLANE__"]
            regs = []
            for e in phases:
                rid = reg.tdb_name(tdbname, e[0])
                note(f"tdb:{tdbname}", e[0], rid, "registry_tdb_name" if rid else "unmapped")
                regs.append(rid)
            verts = []
            for e, rid in zip(phases, regs):
                x = e[2][0]
                if x is not None:
                    verts.append((x, e[0], "printed"))
                elif e[0] in sx:
                    verts.append((sx[e[0]], e[0], "line_compound_formula"))
                elif hyp:
                    verts.append((hyp[0], e[0], "hyperplane_for_null_solution_vertex"))
            merged = []
            for x, ph, how in sorted(verts):
                if merged and abs(x - merged[-1][0]) < 1e-3:
                    merged[-1][1].append(ph); merged[-1][2].append(how)
                else:
                    merged.append([x, [ph], [how]])
            kind = "field" if len(phases) == 1 else "boundary"
            for j, (x, vph, how) in enumerate(merged):
                points.append({
                    "id": f"{o['id']}#v{j}", "kind": kind, "source_class": "literature_" + ("invariant" if "invariant_table" in d["tags"] else "zpf"),
                    "T_K": float(T), "x_Zr": float(x), "phase_labels": [e[0] for e in phases], "registry_ids": regs,
                    "vertex_phases": vph, "vertex_x_basis": how,
                    "n_phases_equilibrium": len(phases),
                    "truth_physical": sorted({pid.bench_class(i) for i in regs if i}),
                    "truth_strict": sorted({pid.bench_class(i, "strict") for i in regs if i}),
                    "unmapped_phases": [e[0] for e, i in zip(phases, regs) if not i],
                    "provenance": {"dataset": "ORCHID literature ESPEI bundle (espei_ready values only)",
                                   "observation_id": o["id"], "file": frel, "file_sha256": sha(f),
                                   "value_index": k, "reference": d.get("reference"), "dois": d["orchid"].get("dois"),
                                   "paper": o.get("paper"), "measured_in": o.get("measured_in"),
                                   "reader_agreement": o.get("reader_agreement"), "tags": d["tags"]},
                    "final_tdb_split": split_flag(o["id"]),
                })
    mt = sorted(mapping.values(), key=lambda m: (m["source"], m["name"]))
    out = {
        "track": "ORCHID-bench one-shot phase diagram, Cu-Zr", "system": "CU-ZR", "x_component": "ZR", "P_Pa": 101325,
        "boundary_tolerance": {"T_K": 10.0, "x": 0.01},
        "inputs": {"orchid_data_release": RELEASE, "assessed_records": {"path": ASPA_REL, "sha256": sha(ASPA)},
                   "espei_bundle": "espei/{experimental,invariants}/CU-ZR", "registry": {"path": pid.REG_REL, "sha256": reg.sha},
                   "final_fit_split": {"path": rel(SPLIT), "sha256": sha(SPLIT),
                                       "source": "orchid-phaseforge-demo data/a4d_F10 and data/holdout (release 778cc6bfa)"}},
        "counts": dict(collections.Counter(p["kind"] + ":" + p["source_class"] for p in points)),
        "literature_duplicate_observations_skipped": dup,
        "phase_classes_physical_merges": pid.MERGE,
        "phase_mapping_evidence": mt,
        "points": points,
    }
    json.dump(out, open(a.o, "w"), indent=1, ensure_ascii=False)
    print(a.o, out["counts"], "dups skipped", dup)
    for m in mt:
        print(f"{m['source']:28s} {m['name']:14s} {str(m['registry_id']):28s} {m['class_physical']:12s} {m['basis']} n={m['n_points']}")


if __name__ == "__main__":
    main()
