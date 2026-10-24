# Copied from orchid-bench-pd scripts/phase_identity.py; the registry path now comes from the pinned release.
"""Phase identity for the Cu-Zr one-shot phase-diagram track.

Every phase name (handbook label, literature/ESPEI TDB name, model TDB name) is mapped in two steps:

1. name -> ORCHID phase-registry id (orchid-data phases/systems/CU-ZR.json at the pinned release, release.py): through the registry's own
   `names` list ('tdb:<file>:<NAME>', 'paper:<doi>:<label>'), or the assessed record's `phase_id` when it has one.
2. registry id -> benchmark class. Two modes:
   - `strict`: the class is the registry id itself (the convention of the ORCHID release, e.g. the Dataset 6 SGTE
     cross-check, which counts CU9ZR2_C15B vs CU5ZR_C15B as a disagreement).
   - `physical` (primary): registry ids that name the same physical phase under different assessments are merged,
     by the explicit rules in MERGE below. Nothing else is merged.

Order/disorder, composition sets and liquid are handled in `model_phase_class` (model side) and here (evidence side):
   - liquid: LIQUID, L, L#2, Liquid, liquid -> phase:cu-zr:liquid; a miscibility gap (two LIQUID composition sets)
     counts as one phase (presence metric).
   - composition sets: 'BCC_A2#2', 'δ#2' -> the phase itself; presence is a set, multiplicity is not scored.
   - order/disorder: a phase with an ordered model (B2_BCC with a disordered part, 22Zho) is classified from its
     site fractions: ordered (|y_ZR(s1) - y_ZR(s2)| >= ORDER_TOL) -> its ordered registry id (bcc-b2 -> class CUZR_B2
     in the physical mode); disordered -> BCC_A2. Stoichiometric CUZR_B2 (94Zen, PhaseForge) is never confused with
     BCC_A2: the two are separate TDB phases and map to separate registry ids.
"""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from release import RELEASE, orchid_data  # noqa: E402

REG_REL = "phases/systems/CU-ZR.json"
ORDER_TOL = 0.01

# Benchmark classes, physical mode. key = registry id, value = class label.
CLASS = {
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
MERGE = [
    {"class": "C15B_CU5ZR", "members": ["phase:cu-zr:cu5zr-c15b", "phase:cu-zr:cu9zr2-c15b"],
     "why": "the most Cu-rich compound, AuBe5-type (C15b, cF24) in both: Hallstedt 2025 mmc2 lists CU9ZR2_C15B as AuBe5 cF24 "
            "and CU5ZR_C15B is C15b by name; the assessments disagree on its stoichiometry (Cu5Zr x=0.167 vs Cu9Zr2 x=0.182), "
            "not on the phase. The registry keeps them apart (assessment_declared_blocked_by:stoichiometry_contradicts_C15_b)."},
    {"class": "CUZR2", "members": ["phase:cu-zr:cuzr2", "phase:cu-zr:cuzr2-ht", "phase:cu-zr:cuzr2-lt", "label:αCuZr2"],
     "why": "CuZr2 (MoSi2-type tI6 in the registry) and the high/low-temperature variants of Wang 2006 and the handbook "
            "(αCuZr2 / βCuZr2): one composition, polymorph family (registry CONFLICTS: polymorph_family). Models carry one CuZr2."},
    {"class": "CUZR_B2", "members": ["phase:cu-zr:cuzr-b2", "phase:cu-zr:bcc-b2"],
     "why": "B2 (CsCl) CuZr, modelled stoichiometrically (CUZR_B2) or as the ordered part of a BCC order/disorder model "
            "(B2_BCC, 22Zho; only when ordered, see ORDER_TOL)."},
]


class Registry:
    def __init__(self, path=None):
        path = path or orchid_data.path(REG_REL, RELEASE)
        d = json.load(open(path))
        self.phases = {p["id"]: p for p in d["phases"]}
        self.by_tdb, self.by_label = {}, {}
        for p in d["phases"]:
            for n in p.get("names", []):
                n = n.get("name") if isinstance(n, dict) else n
                parts = n.split(":")
                if parts[0] == "tdb":
                    self.by_tdb[(parts[1], ":".join(parts[2:]))] = p["id"]
                elif parts[0] == "paper":
                    lab = n.split(":", 2)[2]
                    self.by_label.setdefault(lab, set()).add(p["id"])
            for t in p.get("tdb_names", []) or []:
                self.by_label.setdefault(t, set()).add(p["id"])
        self.sha = __import__("hashlib").sha256(open(path, "rb").read()).hexdigest()

    def tdb_name(self, tdb_file, name):
        """TDB phase name -> registry id; composition-set suffix '#n' dropped."""
        name = re.sub(r"#\d+$", "", name.upper())
        return self.by_tdb.get((os.path.basename(tdb_file), name))

    def label(self, label, phase_id=None):
        """Handbook/paper label -> (registry id, basis)."""
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


def bench_class(reg_id, mode="physical"):
    if reg_id is None:
        return None
    if mode == "strict":
        return reg_id
    return CLASS.get(reg_id, reg_id)
