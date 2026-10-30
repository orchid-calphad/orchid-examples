# Copied from orchid-bench-pd scripts/phase_identity.py; the registry path now comes from the pinned release.
"""Phase identity for the Cu-Zr one-shot phase-diagram track.

Every phase name (handbook label, literature/ESPEI TDB name, model TDB name) is mapped in two steps:

1. name -> ORCHID phase-registry id (orchid-data phases/systems/CU-ZR.json at the pinned release, release.py): through the registry's own
   `names` list ('tdb:<file>:<NAME>', 'paper:<doi>:<label>'), or the assessed record's `phase_id` when it has one.
2. registry id -> benchmark class. Two modes:
   - `strict`: the class is the registry id itself (the convention of the ORCHID release, e.g. the Dataset 6 SGTE
     cross-check, which counts CU9ZR2_C15B vs CU5ZR_C15B as a disagreement).
   - `physical` (primary): registry ids that name the same physical phase under different assessments are merged,
     by the explicit rules in merge_rules below. Nothing else is merged.

Order/disorder, composition sets and liquid are handled in `model_phase_class` (model side) and here (evidence side):
   - liquid: LIQUID, L, L#2, Liquid, liquid -> phase:cu-zr:liquid; a miscibility gap (two LIQUID composition sets)
     counts as one phase (presence metric).
   - composition sets: 'BCC_A2#2', 'δ#2' -> the phase itself; presence is a set, multiplicity is not scored.
   - order/disorder: a phase with an ordered model (B2_BCC with a disordered part, 22Zho) is classified from its
     site fractions: ordered (|y_ZR(s1) - y_ZR(s2)| >= order_tol) -> its ordered registry id (bcc-b2 -> class CUZR_B2
     in the physical mode); disordered -> BCC_A2. Stoichiometric CUZR_B2 (94Zen, PhaseForge) is never confused with
     BCC_A2: the two are separate TDB phases and map to separate registry ids.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent))
from release import orchid_data, release_commit  # noqa: E402

reg_rel = "phases/systems/CU-ZR.json"
order_tol = 0.01  # site-fraction difference between the two sublattices above which an order/disorder phase is ordered

# Benchmark classes, physical mode. key = registry id, value = class label.
physical_class = {
    "phase:cu-zr:liquid": "LIQUID",
    "phase:cu-zr:fcc-a1": "FCC_A1",
    "phase:cu-zr:bcc-a2": "BCC_A2",
    "phase:cu-zr:hcp-a3": "HCP_A3",
    "phase:cu-zr:cu5zr-c15b": "C15B_CU5ZR",
    "phase:cu-zr:cu9zr2-c15b": "C15B_CU5ZR",
    "phase:cu-zr:cu51zr14": "CU51ZR14",
    "phase:cu-zr:cu8zr3": "CU8ZR3",
    "phase:cu-zr:cu10zr7": "CU10ZR7",
    "phase:cu-zr:cuzr-b2": "CUZR_B2",
    "phase:cu-zr:bcc-b2": "CUZR_B2",
    "phase:cu-zr:cuzr2": "CUZR2",
    "phase:cu-zr:cuzr2-ht": "CUZR2",
    "phase:cu-zr:cuzr2-lt": "CUZR2",
    "label:αCuZr2": "CUZR2",
    "phase:cu-zr:cu24zr13": "CU24ZR13",
    "phase:cu-zr:cu2zr": "CU2ZR",
    "phase:cu-zr:cu5zr8": "CU5ZR8",
    "phase:cu-zr:cu5zr7": "CU5ZR7",
    "phase:cu-zr:cu4zr": "CU4ZR",
    "phase:cu-zr:cu5zr2": "CU5ZR2",
    "phase:cu-zr:cuzr-b19": "CUZR_B19",
    "phase:cu-zr:cuzr-b33": "CUZR_B33",
}
merge_rules = [
    {
        "class": "C15B_CU5ZR",
        "members": ["phase:cu-zr:cu5zr-c15b", "phase:cu-zr:cu9zr2-c15b"],
        "why": "the most Cu-rich compound, AuBe5-type (C15b, cF24) in both: Hallstedt 2025 mmc2 lists CU9ZR2_C15B as AuBe5 cF24 "
        "and CU5ZR_C15B is C15b by name; the assessments disagree on its stoichiometry (Cu5Zr x=0.167 vs Cu9Zr2 x=0.182), "
        "not on the phase. The registry keeps them apart (assessment_declared_blocked_by:stoichiometry_contradicts_C15_b).",
    },
    {
        "class": "CUZR2",
        "members": ["phase:cu-zr:cuzr2", "phase:cu-zr:cuzr2-ht", "phase:cu-zr:cuzr2-lt", "label:αCuZr2"],
        "why": "CuZr2 (MoSi2-type tI6 in the registry) and the high/low-temperature variants of Wang 2006 and the handbook "
        "(αCuZr2 / βCuZr2): one composition, polymorph family (registry CONFLICTS: polymorph_family). Models carry one CuZr2.",
    },
    {
        "class": "CUZR_B2",
        "members": ["phase:cu-zr:cuzr-b2", "phase:cu-zr:bcc-b2"],
        "why": "B2 (CsCl) CuZr, modelled stoichiometrically (CUZR_B2) or as the ordered part of a BCC order/disorder model "
        "(B2_BCC, 22Zho; only when ordered, see ORDER_TOL).",
    },
]


class Registry:
    """Phase-name lookup into the pinned ORCHID phase registry.

    Args:
        path (str | Path | None): Registry JSON file. Defaults to the phase registry of the pinned release.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        """Load the registry and index its TDB and paper names."""
        path = path or orchid_data.path(reg_rel, release_commit)
        d = json.loads(Path(path).read_text())
        self.phases = {p["id"]: p for p in d["phases"]}
        self.by_tdb, self.by_label = {}, {}
        for p in d["phases"]:
            for entry in p.get("names", []):
                name = entry.get("name") if isinstance(entry, dict) else entry
                parts = name.split(":")
                if parts[0] == "tdb":
                    self.by_tdb[(parts[1], ":".join(parts[2:]))] = p["id"]
                elif parts[0] == "paper":
                    lab = name.split(":", 2)[2]
                    self.by_label.setdefault(lab, set()).add(p["id"])
            for t in p.get("tdb_names", []) or []:
                self.by_label.setdefault(t, set()).add(p["id"])
        self.sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def tdb_name(self, tdb_file: str | Path, name: str) -> str | None:
        """Map a TDB phase name to its registry id.

        Args:
            tdb_file (str | Path): TDB file the name comes from; only its file name is used.
            name (str): Phase name as written in the TDB. A composition-set suffix '#n' is dropped.

        Returns:
            str | None: Registry id, or None when the registry does not know the name.
        """
        name = re.sub(r"#\d+$", "", name.upper())
        return self.by_tdb.get((Path(tdb_file).name, name))

    def label(self, label: str, phase_id: str | None = None) -> tuple[str, str]:
        """Map a handbook or paper label to its registry id.

        Args:
            label (str): Phase label as printed in the handbook or paper.
            phase_id (str | None): Registry id already given by the assessed record; takes precedence when present.

        Returns:
            tuple[str, str]: Registry id (or "label:<name>" when unmapped) and the basis of the mapping.
        """
        if phase_id:
            return phase_id, "assessed_record_phase_id"
        base = re.sub(r"#\d+$", "", label)
        if base in ("L", "Liquid", "liquid", "LIQUID"):
            return "phase:cu-zr:liquid", "liquid_label"
        ids = self.by_label.get(base, set())
        if len(ids) == 1:
            return next(iter(ids)), "registry_names_unique_label"
        if base.startswith("β") and base[1:] in self.by_label and len(self.by_label[base[1:]]) == 1:
            return next(iter(self.by_label[base[1:]])), "registry_names_label_without_polymorph_prefix"
        return f"label:{base}", ("ambiguous_label:" + ";".join(sorted(ids))) if ids else "unmapped_label"


def bench_class(reg_id: str | None, mode: str = "physical") -> str | None:
    """Benchmark class of a registry id.

    Args:
        reg_id (str | None): Registry id; None passes through.
        mode (str): "physical" merges the ids listed in merge_rules; "strict" keeps the registry id.

    Returns:
        str | None: Benchmark class label, or None when reg_id is None.
    """
    if reg_id is None:
        return None
    if mode == "strict":
        return reg_id
    return physical_class.get(reg_id, reg_id)
