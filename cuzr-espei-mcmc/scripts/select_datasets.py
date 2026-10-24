"""Choose ORCHID evidence for a classic ESPEI MCMC refinement of Cu-Zr and write a ready-to-run ESPEI input.

1. Gather the Cu-Zr ESPEI datasets of the pinned ORCHID release (RELEASE below) from the folders of the chosen
   evidence classes. Each folder holds one evidence class; datasets are written in the phase names of the assessment
   (TDB) they are named after, so a dataset is kept only when every phase it names maps, in the ORCHID phase registry,
   to the same registry phase in its own TDB and in CuZr-94Zen-LB.tdb (the model refined here).
2. Load them with ESPEI's `load_datasets` and choose evidence with `apply_tags` (the same tag operations are written
   into espei-in.yaml under `system: tags`, so `espei --input` applies them identically).
3. Copy the kept files to work/datasets/ and write work/espei-in.yaml (ESPEI 0.9.1 schema; mcmc of 8 coefficients:
   liquid 0L, 1L (a + bT) and the constants of CU51ZR14, CU10ZR7, CUZR_B2, CUZR2; normal priors centred on the SGTE
   values, sd = |value|; 2 chains per parameter).

  python scripts/select_datasets.py [--iterations 1500] [--evidence measured|measured+constraints]
                                    [--drop-single-reader] [--scheduler dask|none] [--cores N]

Evidence classes: `measured` = experimental/ (activities, formation enthalpies and entropies, phase compositions) +
invariants/ (printed invariant tables) + dft/literature/ for CuZr-94Zen-LB (calibrated DFT sigma);
`measured+constraints` (default) adds the optional phase constraints: derived invariant brackets, derived
liquid-stability points and the assessed phase-assemblage points (Dataset 6) written for CuZr-94Zen-LB.
"""
import argparse, collections, json, os, shutil, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(ROOT), "common"))
import orchid_data  # noqa: E402

RELEASE = "778cc6bfa67db292e3f917f5ac6d691e0f37fc1a"   # orchid-data commit (same as cuzr-phaseforgeplus)
TARGET = "CuZr-94Zen-LB"
FOLDERS = {"measured": ["espei/experimental/CU-ZR", "espei/invariants/CU-ZR", "espei/dft/literature/CU-ZR"],
           "constraints": ["espei/derived/invariant-brackets/CU-ZR", "espei/derived/liquid-stability/CU-ZR", "espei/assessed/CU-ZR"]}
ONLY_TARGET = ("espei/dft/literature/CU-ZR", "espei/derived/liquid-stability/CU-ZR", "espei/assessed/CU-ZR")


def registry_map():
    reg = json.load(open(orchid_data.path("phases/systems/CU-ZR.json", RELEASE)))
    pid = collections.defaultdict(dict)
    for p in reg["phases"]:
        for m in p.get("models") or []:
            pid[m["tdb"][:-4]][m["phase_name"]] = p["id"]
    return pid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--evidence", choices=("measured", "measured+constraints"), default="measured+constraints")
    ap.add_argument("--drop-single-reader", action="store_true", help="disable datasets tagged single_reader (apply_tags)")
    ap.add_argument("--scheduler", default="dask", choices=("dask", "none"))
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--work", default=os.path.join(ROOT, "work"))
    a = ap.parse_args()
    from espei.datasets import load_datasets, apply_tags

    pid = registry_map()
    folders = FOLDERS["measured"] + (FOLDERS["constraints"] if a.evidence == "measured+constraints" else [])
    keep, skipped = [], collections.Counter()
    for folder in folders:
        for rel in orchid_data.listdir(folder, RELEASE):
            stem = os.path.basename(rel).split(".")[0]
            if folder in ONLY_TARGET and stem != TARGET:
                skipped["other TDB (folder used for the target TDB only)"] += 1; continue
            d = json.load(open(orchid_data.path(rel, RELEASE)))
            ok = all(pid.get(stem, {}).get(ph) is not None and pid[stem][ph] == pid[TARGET].get(ph) for ph in d["phases"])
            if not ok:
                skipped["phase not in CuZr-94Zen-LB (registry)"] += 1; continue
            keep.append(rel)
    ds_dir = os.path.join(a.work, "datasets")
    shutil.rmtree(ds_dir, ignore_errors=True); os.makedirs(ds_dir)
    for rel in keep:   # flat copy; the name keeps folder and file (one evidence class per folder in the release)
        shutil.copy(orchid_data.path(rel, RELEASE), os.path.join(ds_dir, rel[len("espei/"):].replace("/", "__")))
    files = sorted(os.path.join(ds_dir, f) for f in os.listdir(ds_dir))

    tags = {"model": {"disabled": True}}                 # model-calculated values never enter (none are in these folders)
    if a.drop_single_reader:
        tags["single_reader"] = {"disabled": True}
    ds = load_datasets(files, include_disabled=True)
    apply_tags(ds, tags)
    on = [d for d in ds.all() if not d.get("disabled", False)]
    by = collections.Counter(d["output"].split("_")[0] if d["output"].startswith("ACR") else d["output"] for d in on)
    print(f"release {RELEASE[:10]}: {len(keep)} datasets kept from {len(folders)} folders, skipped {dict(skipped)}")
    print(f"after apply_tags({tags}): {len(on)} enabled datasets; by output: {dict(sorted(by.items()))}")

    params = json.load(open(os.path.join(ROOT, "tdb", "params.json")))
    prior = [{"name": "normal", "loc_absolute": p["start"], "scale_absolute": abs(p["start"])} for p in params]
    yml = ["system:", f"  phase_models: {os.path.relpath(os.path.join(ROOT, 'inputs', 'phases.json'), a.work)}",
           "  datasets: datasets", "  tags:"] + [f"    {k}: {{disabled: true}}" for k in tags] + [
           "output:", "  verbosity: 1", "  output_db: refined_REVIEW_ONLY.tdb", "  tracefile: trace.npy", "  probfile: lnprob.npy",
           "mcmc:", f"  iterations: {a.iterations}",
           f"  input_db: {os.path.relpath(os.path.join(ROOT, 'tdb', 'CuZr-94Zen-LB_param.tdb'), a.work)}",
           "  save_interval: 1", f"  scheduler: {'dask' if a.scheduler == 'dask' else 'null'}"] + (
           [f"  cores: {a.cores}"] if a.scheduler == "dask" else []) + [
           "  chains_per_parameter: 2", "  chain_std_deviation: 0.05", "  deterministic: true", "  prior:"] + sum(
           ([f"  - name: normal", f"    loc_absolute: {p['loc_absolute']}", f"    scale_absolute: {p['scale_absolute']}"] for p in prior), [])
    open(os.path.join(a.work, "espei-in.yaml"), "w").write("\n".join(yml) + "\n")
    json.dump({"orchid_data_release": RELEASE, "evidence": a.evidence, "tags": tags, "kept": keep, "skipped": dict(skipped),
               "enabled_by_output": dict(by)}, open(os.path.join(a.work, "selection.json"), "w"), indent=1)
    print(f"wrote {os.path.relpath(a.work, ROOT)}/espei-in.yaml; run:  cd {os.path.relpath(a.work, ROOT)} && espei --input espei-in.yaml")


if __name__ == "__main__":
    main()
