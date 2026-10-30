# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/printed_invariants.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Match the printed ORCHID invariant temperatures of Cu-Zr to the model reactions.

Copied from orchid-espei-demo scripts/v2/printed_invariants.py, paths adapted. The fit and hold-out sets of the v2
data are matched to the model reactions (invariants.reactions, in scripts/model.py) by phase set, using CuZr-94Zen-LB
phase names. Writes results/printed_invariants.json. Usage: python scripts/full/printed_invariants.py (needs ORCHID_DATA).
"""

import collections
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import orchid, root

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))  # model.py, as set up by _paths.py
import model as iv

default_u_k = 5.0  # K: temperature uncertainty used when the publication prints none
pub = {}
with gzip.open(orchid("literature/orchid-literature.jsonl.gz"), "rt") as fh:
    for line in fh:
        r = json.loads(line)
        if r.get("system") == "CU-ZR":
            pub[r["id"]] = r
rows = []
for split in ("fit", "holdout"):
    for f in sorted(str(p) for p in (root / "data" / split).glob("*.json")):
        b = Path(f).name
        if not (b.startswith("experimental__") or b.startswith("invariants__")):
            continue
        d = json.loads(Path(f).read_text())
        if d["output"] != "ZPF":
            continue
        temps = d["conditions"]["T"] if isinstance(d["conditions"]["T"], list) else [d["conditions"]["T"]] * len(d["values"])
        for T, reg, ob in zip(temps, d["values"], d["orchid"]["observations"], strict=False):
            r = pub.get(ob["id"], {})
            if r.get("observation_type") != "invariant_reaction":
                continue
            ph = {v[0] for v in reg if v[0] != "__HYPERPLANE__"}
            rx = [x[0] for x in iv.reactions if x[4] == ph]
            tu = r.get("temperature_uncertainty") or {}
            u = tu["value"] if tu.get("basis") in ("printed", "printed_range") and tu.get("value") else None
            rows.append(
                {
                    "reaction": rx[0] if rx else None,
                    "phases": sorted(ph),
                    "T_K": T,
                    "u_K": u,
                    "u_used_K": u or default_u_k,
                    "split": split,
                    "id": ob["id"],
                    "also_reported_as": [a.get("id") for a in ob.get("also_reported_as", [])],
                    "printed": (r.get("reaction") or {}).get("printed"),
                    "reference": d.get("reference"),
                    "file": b,
                }
            )
(root / "results").mkdir(parents=True, exist_ok=True)
(root / "results" / "printed_invariants.json").write_text(json.dumps(rows, indent=1))
c = collections.defaultdict(list)
for r in rows:
    c[(r["reaction"], r["split"])].append(r["T_K"])
for k in sorted(c, key=str):
    print(k, sorted(c[k]))
