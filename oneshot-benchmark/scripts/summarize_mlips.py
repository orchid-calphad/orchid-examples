"""Table of the one-shot Cu-Zr scores of the MLIP-only descriptions (results/mlips/cuzr_*.json) next to the references.
Usage: python summarize_mlips.py   -> results/mlips_summary.md (markdown table, printed too)."""
import json, os, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = {"grace-2l-omat": "GRACE-2L-OMAT", "mace-mpa-0": "MACE-MPA-0", "mace-omat-0": "MACE-OMAT-0", "sevennet-omni": "SevenNet-Omni",
         "orb-v3-omat": "ORB-v3 OMat", "orb-v3-mpa": "ORB-v3 MPA", "mattersim-5m": "MatterSim 5M", "chgnet-0.3.0": "CHGNet 0.3.0"}


def cell(d, kind):
    b = d["metrics"]["physical"][kind]["all"]; w, c = b["weighted"], b["ci95"]["recall"]
    return f"{w['recall']:.2f} [{c[0]:.2f}, {c[1]:.2f}]", f"{w['precision']:.2f}", f"{w['f1']:.2f}"


rows = []
for f in glob.glob(os.path.join(ROOT, "results/mlips/cuzr_*.json")):
    d = json.load(open(f)); rows.append((d["metrics"]["physical"]["field"]["all"]["weighted"]["recall"], NAMES[d["label"][5:]], d))
rows.sort(key=lambda r: -r[0])
ref = [("SGTE 94Zen (reference)", "results/cuzr_sgte_94Zen.json"), ("PhaseForge+ final (in-sample)", "results/cuzr_phaseforgeplus_final.json")]
out = ["| model (MLIP-only, no fitting) | field S [95% CI] | P | F1 | boundary S [95% CI] | P | F1 |", "|---|---|---|---|---|---|---|"]
for _, n, d in rows + [(0, a, json.load(open(os.path.join(ROOT, p)))) for a, p in ref]:
    out.append("| " + " | ".join([n, *cell(d, "field"), *cell(d, "boundary")]) + " |")
s = "\n".join(out)
open(os.path.join(ROOT, "results/mlips_summary.md"), "w").write(s + "\n")
print(s)
