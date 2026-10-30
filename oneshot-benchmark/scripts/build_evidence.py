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

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

from pycalphad import Database

sys.path.insert(0, str(Path(__file__).absolute().parent))
import phase_identity as pid  # noqa: E402
from release import example_root, file_sha256, orchid_data, rel, release_commit  # noqa: E402

aspa_rel = "assessed-phase-assemblages/records.jsonl.gz"
tdb_dirs = [example_root / "tdb", example_root / "tdb" / "reference_panel"]
split_path = example_root / "inputs" / "phaseforge_fit_split.json"
vertex_merge_x = 1e-3  # mole fraction; vertices of one equilibrium closer than this are one congruent point


class _FitSplit:
    """Membership of evidence points in the fit set of the PhaseForge+ final TDB.

    Args:
        split (dict): Contents of inputs/phaseforge_fit_split.json (fit_ids, holdout_ids, dense_grid_assemblages).
    """

    def __init__(self, split: dict) -> None:
        """Index the fit and hold-out ids and the dense assessed-field assemblages."""
        self.fit_ids = set(split["fit_ids"])
        self.hold_ids = set(split["holdout_ids"])
        self.dense_assemblages = {frozenset(s) for s in split["dense_grid_assemblages"]}

    def flag(self, i: str, labels: list[str] | None = None) -> str:
        """Fit-set flag of one point.

        Args:
            i (str): Record or observation id of the point.
            labels (list[str] | None): Phase labels of a field point; a field point whose assemblage is one of the dense
                assessed-field assemblages counts as fit.

        Returns:
            str: "fit", "fit(dense_assessed_field)", "holdout" or "not_used".
        """
        if i in self.fit_ids:
            return "fit"
        if labels is not None and frozenset(labels) in self.dense_assemblages:
            return "fit(dense_assessed_field)"
        if i in self.hold_ids:
            return "holdout"
        return "not_used"


def stoich_x(tdb_path: str) -> dict[str, float]:
    """Zr mole fraction of the line compounds of a TDB.

    Args:
        tdb_path (str): TDB file to read.

    Returns:
        dict[str, float]: Phase name -> x_Zr, for the phases with one constituent per non-VA sublattice.
    """
    db = Database(tdb_path)
    out = {}
    for name, ph in db.phases.items():
        subl = [(r, {str(c) for c in cons} - {"VA"}) for r, cons in zip(ph.sublattices, ph.constituents, strict=True)]
        subl = [(r, c) for r, c in subl if c]
        if all(len(c) == 1 for _, c in subl):
            tot = sum(float(r) for r, _ in subl)
            out[name] = sum(float(r) for r, c in subl if "ZR" in c) / tot
    return out


def _note(mapping: dict, source_kind: str, name: str, reg_id: str | None, basis: str) -> None:
    """Record one use of a source phase name in the phase-mapping evidence.

    Args:
        mapping (dict): Mapping table, keyed by (source kind, name); updated in place.
        source_kind (str): "handbook_label" or "tdb:<file>".
        name (str): Phase name as written in the source.
        reg_id (str | None): Registry id it maps to.
        basis (str): How the mapping was made.
    """
    entry = mapping.setdefault(
        (source_kind, name),
        {
            "source": source_kind,
            "name": name,
            "registry_id": reg_id,
            "basis": basis,
            "class_physical": pid.bench_class(reg_id),
            "class_strict": pid.bench_class(reg_id, "strict"),
            "n_points": 0,
        },
    )
    entry["n_points"] += 1


def _field_points(reg: pid.Registry, mapping: dict, split: _FitSplit) -> list[dict]:
    """Dataset 6 field points: one per assessed Cu-Zr record.

    Args:
        reg (pid.Registry): Phase registry used to map the handbook labels.
        mapping (dict): Phase-mapping evidence, updated in place.
        split (_FitSplit): Fit-set membership.

    Returns:
        list[dict]: Field evidence points.
    """
    points = []
    aspa = str(orchid_data.path(aspa_rel, release_commit))
    with gzip.open(aspa, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["system"] != "CU-ZR":
                continue
            assert r["x_component"] == "Zr"
            labels, regs = [], []
            for ph in r["assemblage"]:
                rid, basis = reg.label(ph["label"], ph.get("phase_id"))
                _note(mapping, "handbook_label", ph["label"], rid, basis)
                labels.append(ph["label"])
                regs.append(rid)
            points.append(
                {
                    "id": r["id"],
                    "kind": "field",
                    "source_class": "assessed_phase_diagram",
                    "T_K": r["T_K"],
                    "x_Zr": r["x"],
                    "phase_labels": labels,
                    "registry_ids": regs,
                    "truth_physical": sorted({pid.bench_class(i) for i in regs}),
                    "truth_strict": sorted({pid.bench_class(i, "strict") for i in regs}),
                    "provenance": {
                        "dataset": "ORCHID Dataset 6 (assessed-phase-assemblages)",
                        "record_id": r["id"],
                        "diagram": r["diagram"],
                        "field": r["field"],
                        "point_method": r["point_method"],
                        "boundary_basis": r["boundary_basis"],
                        "basis": r["basis"],
                        "confidence": r["confidence"],
                        "review_state": r["review_state"],
                        "flags": r["flags"],
                        "coordinate_uncertainty": r["coordinate_uncertainty"],
                        "positional_margin": r["positional_margin"],
                        "citations": [c.get("citation") for c in r["citations"]],
                        "dois": [c.get("doi") for c in r["citations"] if c.get("doi")],
                    },
                    "final_tdb_split": split.flag(r["id"], labels),
                }
            )
    return points


def _tdb_path(tdbname: str) -> str:
    """Path of a TDB by its file name, searched in the TDB folders.

    Args:
        tdbname (str): TDB file name.

    Returns:
        str: Path of the first folder that holds the file.
    """
    return str(next(t / tdbname for t in tdb_dirs if (t / tdbname).exists()))


def _merged_vertices(phases: list, hyp: list, sx: dict[str, float]) -> list[list]:
    """Composition vertices of one literature equilibrium, merged when congruent.

    Args:
        phases (list): Phase entries of the equilibrium, each [name, _, [x, ...]].
        hyp (list): Hyperplane composition of the equilibrium, empty when it has none.
        sx (dict[str, float]): Line-compound compositions of the source TDB.

    Returns:
        list[list]: [x_Zr, [phase names], [vertex bases]] per vertex, sorted by x_Zr.
    """
    verts = []
    for e in phases:
        x = e[2][0]
        if x is not None:
            verts.append((x, e[0], "printed"))
        elif e[0] in sx:
            verts.append((sx[e[0]], e[0], "line_compound_formula"))
        elif hyp:
            verts.append((hyp[0], e[0], "hyperplane_for_null_solution_vertex"))
    merged = []
    for x, ph, how in sorted(verts):
        if merged and abs(x - merged[-1][0]) < vertex_merge_x:
            merged[-1][1].append(ph)
            merged[-1][2].append(how)
        else:
            merged.append([x, [ph], [how]])
    return merged


def _literature_points(reg: pid.Registry, mapping: dict, split: _FitSplit) -> tuple[list[dict], int]:
    """Boundary points from the literature ZPF and invariant tables of the ESPEI bundle.

    Args:
        reg (pid.Registry): Phase registry used to map the TDB phase names.
        mapping (dict): Phase-mapping evidence, updated in place.
        split (_FitSplit): Fit-set membership.

    Returns:
        tuple[list[dict], int]: Boundary evidence points, and the number of duplicate observations skipped.
    """
    points = []
    rels = [r for r in orchid_data.listdir("espei/experimental/CU-ZR", release_commit) if "zpf" in r.rsplit("/", 1)[1]]
    rels = rels + orchid_data.listdir("espei/invariants/CU-ZR", release_commit)
    seen, files = {}, [(r, str(orchid_data.path(r, release_commit))) for r in rels]
    stoich_cache, dup = {}, 0
    for frel, f in files:
        d = json.loads(Path(f).read_text())
        if d.get("output") != "ZPF":
            continue
        tdbname = Path(f).name.split(".")[0] + ".tdb"
        if tdbname not in stoich_cache:
            stoich_cache[tdbname] = stoich_x(_tdb_path(tdbname))
        sx = stoich_cache[tdbname]
        obs = d["orchid"]["observations"]
        assert len(obs) == len(d["values"]) == len(d["conditions"]["T"]), f
        for k, (temp, val, o) in enumerate(zip(d["conditions"]["T"], d["values"], obs, strict=True)):
            if o["id"] in seen:
                dup += 1
                continue
            seen[o["id"]] = f
            phases = [e for e in val if e[0] != "__HYPERPLANE__"]
            hyp = [e[2][0] for e in val if e[0] == "__HYPERPLANE__"]
            regs = []
            for e in phases:
                rid = reg.tdb_name(tdbname, e[0])
                _note(mapping, f"tdb:{tdbname}", e[0], rid, "registry_tdb_name" if rid else "unmapped")
                regs.append(rid)
            merged = _merged_vertices(phases, hyp, sx)
            kind = "field" if len(phases) == 1 else "boundary"
            for j, (x, vph, how) in enumerate(merged):
                points.append(
                    {
                        "id": f"{o['id']}#v{j}",
                        "kind": kind,
                        "source_class": "literature_" + ("invariant" if "invariant_table" in d["tags"] else "zpf"),
                        "T_K": float(temp),
                        "x_Zr": float(x),
                        "phase_labels": [e[0] for e in phases],
                        "registry_ids": regs,
                        "vertex_phases": vph,
                        "vertex_x_basis": how,
                        "n_phases_equilibrium": len(phases),
                        "truth_physical": sorted({pid.bench_class(i) for i in regs if i}),
                        "truth_strict": sorted({pid.bench_class(i, "strict") for i in regs if i}),
                        "unmapped_phases": [e[0] for e, i in zip(phases, regs, strict=True) if not i],
                        "provenance": {
                            "dataset": "ORCHID literature ESPEI bundle (espei_ready values only)",
                            "observation_id": o["id"],
                            "file": frel,
                            "file_sha256": file_sha256(f),
                            "value_index": k,
                            "reference": d.get("reference"),
                            "dois": d["orchid"].get("dois"),
                            "paper": o.get("paper"),
                            "measured_in": o.get("measured_in"),
                            "reader_agreement": o.get("reader_agreement"),
                            "tags": d["tags"],
                        },
                        "final_tdb_split": split.flag(o["id"]),
                    }
                )
    return points, dup


def main() -> None:
    """Build the evidence file from the pinned release and print the phase-mapping summary."""
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", default=str(example_root / "inputs" / "evidence.json"))
    a = ap.parse_args()
    reg = pid.Registry()
    split = _FitSplit(json.loads(Path(split_path).read_text()))
    mapping = {}
    field = _field_points(reg, mapping, split)
    literature, dup = _literature_points(reg, mapping, split)
    points = field + literature
    mt = sorted(mapping.values(), key=lambda m: (m["source"], m["name"]))
    out = {
        "track": "ORCHID-bench one-shot phase diagram, Cu-Zr",
        "system": "CU-ZR",
        "x_component": "ZR",
        "P_Pa": 101325,
        "boundary_tolerance": {"T_K": 10.0, "x": 0.01},
        "inputs": {
            "orchid_data_release": release_commit,
            "assessed_records": {"path": aspa_rel, "sha256": file_sha256(str(orchid_data.path(aspa_rel, release_commit)))},
            "espei_bundle": "espei/{experimental,invariants}/CU-ZR",
            "registry": {"path": pid.reg_rel, "sha256": reg.sha},
            "final_fit_split": {
                "path": rel(split_path),
                "sha256": file_sha256(split_path),
                "source": "orchid-phaseforge-demo data/a4d_F10 and data/holdout (release 778cc6bfa)",
            },
        },
        "counts": dict(collections.Counter(p["kind"] + ":" + p["source_class"] for p in points)),
        "literature_duplicate_observations_skipped": dup,
        "phase_classes_physical_merges": pid.merge_rules,
        "phase_mapping_evidence": mt,
        "points": points,
    }
    with Path(a.o).open("w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(a.o, out["counts"], "dups skipped", dup)
    for m in mt:
        print(
            f"{m['source']:28s} {m['name']:14s} {str(m['registry_id']):28s} {m['class_physical']:12s} {m['basis']} n={m['n_points']}"
        )


if __name__ == "__main__":
    main()
