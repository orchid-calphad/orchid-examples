"""Choose ORCHID evidence for a classic ESPEI MCMC refinement of Cu-Zr and write a ready-to-run ESPEI input.

1. Gather the Cu-Zr ESPEI datasets of the pinned ORCHID release (release below) from the folders of the chosen
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

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
from pathlib import Path

root = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(root.parent / "common"))
import orchid_data  # noqa: E402

release = "778cc6bfa67db292e3f917f5ac6d691e0f37fc1a"  # orchid-data commit (same as cuzr-phaseforgeplus)
target = "CuZr-94Zen-LB"
evidence_folders = {
    "measured": ["espei/experimental/CU-ZR", "espei/invariants/CU-ZR", "espei/dft/literature/CU-ZR"],
    "constraints": [
        "espei/derived/invariant-brackets/CU-ZR",
        "espei/derived/liquid-stability/CU-ZR",
        "espei/assessed/CU-ZR",
    ],
}
only_target = ("espei/dft/literature/CU-ZR", "espei/derived/liquid-stability/CU-ZR", "espei/assessed/CU-ZR")


def registry_map() -> dict[str, dict[str, str]]:
    """Map each TDB name and its phase names to the ORCHID registry phase id.

    Returns:
        dict[str, dict[str, str]]: TDB stem to {TDB phase name: registry phase id}.
    """
    reg = json.loads(orchid_data.path("phases/systems/CU-ZR.json", release).read_text(encoding="utf-8"))
    pid: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for p in reg["phases"]:
        for m in p.get("models") or []:
            pid[m["tdb"][:-4]][m["phase_name"]] = p["id"]
    return pid


def _espei_yaml_lines(args: argparse.Namespace, tags: dict, prior: list[dict]) -> list[str]:
    """Lines of the ESPEI 0.9.1 input file (espei-in.yaml) for this selection.

    Args:
        args (argparse.Namespace): Parsed command-line options (iterations, scheduler, cores, work).
        tags (dict): Tag operations written under system: tags, as applied by apply_tags.
        prior (list[dict]): Normal priors, one per coefficient, with loc_absolute and scale_absolute.

    Returns:
        list[str]: The YAML lines, without the trailing newline.
    """
    yml = [
        "system:",
        f"  phase_models: {Path(os.path.relpath(root / 'inputs' / 'phases.json', args.work)).as_posix()}",
        "  datasets: datasets",
        "  tags:",
    ]
    yml += [f"    {k}: {{disabled: true}}" for k in tags]
    yml += [
        "output:",
        "  verbosity: 1",
        "  output_db: refined_REVIEW_ONLY.tdb",
        "  tracefile: trace.npy",
        "  probfile: lnprob.npy",
        "mcmc:",
        f"  iterations: {args.iterations}",
        f"  input_db: {Path(os.path.relpath(root / 'tdb' / 'CuZr-94Zen-LB_param.tdb', args.work)).as_posix()}",
        "  save_interval: 1",
        f"  scheduler: {'dask' if args.scheduler == 'dask' else 'null'}",
    ]
    if args.scheduler == "dask":
        yml.append(f"  cores: {args.cores}")
    yml += ["  chains_per_parameter: 2", "  chain_std_deviation: 0.05", "  deterministic: true", "  prior:"]
    for p in prior:
        yml += [
            "  - name: normal",
            f"    loc_absolute: {p['loc_absolute']}",
            f"    scale_absolute: {p['scale_absolute']}",
        ]
    return yml


def main() -> None:
    """Select the datasets, write them with the tags, and write the ESPEI input and selection record."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--evidence", choices=("measured", "measured+constraints"), default="measured+constraints")
    ap.add_argument("--drop-single-reader", action="store_true", help="disable datasets tagged single_reader (apply_tags)")
    ap.add_argument("--scheduler", default="dask", choices=("dask", "none"))
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--work", default=str(root / "work"))
    a = ap.parse_args()
    from espei.datasets import apply_tags, load_datasets  # noqa: PLC0415  lazy: ESPEI is only needed for this step

    pid = registry_map()
    folders = evidence_folders["measured"] + (evidence_folders["constraints"] if a.evidence == "measured+constraints" else [])
    keep, skipped = [], collections.Counter()
    for folder in folders:
        for rel in orchid_data.listdir(folder, release):
            stem = Path(rel).name.split(".")[0]
            if folder in only_target and stem != target:
                skipped["other TDB (folder used for the target TDB only)"] += 1
                continue
            d = json.loads(orchid_data.path(rel, release).read_text(encoding="utf-8"))
            ok = all(pid.get(stem, {}).get(ph) is not None and pid[stem][ph] == pid[target].get(ph) for ph in d["phases"])
            if not ok:
                skipped["phase not in CuZr-94Zen-LB (registry)"] += 1
                continue
            keep.append(rel)
    ds_dir = Path(a.work) / "datasets"
    shutil.rmtree(ds_dir, ignore_errors=True)
    ds_dir.mkdir(parents=True)
    for rel in keep:  # flat copy; the name keeps folder and file (one evidence class per folder in the release)
        shutil.copy(orchid_data.path(rel, release), ds_dir / rel[len("espei/") :].replace("/", "__"))
    files = sorted(str(p) for p in ds_dir.iterdir())

    tags = {"model": {"disabled": True}}  # model-calculated values never enter (none are in these folders)
    if a.drop_single_reader:
        tags["single_reader"] = {"disabled": True}
    ds = load_datasets(files, include_disabled=True)
    apply_tags(ds, tags)
    on = [d for d in ds.all() if not d.get("disabled", False)]
    by = collections.Counter(d["output"].split("_")[0] if d["output"].startswith("ACR") else d["output"] for d in on)
    print(f"release {release[:10]}: {len(keep)} datasets kept from {len(folders)} folders, skipped {dict(skipped)}")
    print(f"after apply_tags({tags}): {len(on)} enabled datasets; by output: {dict(sorted(by.items()))}")

    params = json.loads((root / "tdb" / "params.json").read_text(encoding="utf-8"))
    prior = [{"name": "normal", "loc_absolute": p["start"], "scale_absolute": abs(p["start"])} for p in params]
    (Path(a.work) / "espei-in.yaml").write_text("\n".join(_espei_yaml_lines(a, tags, prior)) + "\n", encoding="utf-8")
    selection = {
        "orchid_data_release": release,
        "evidence": a.evidence,
        "tags": tags,
        "kept": keep,
        "skipped": dict(skipped),
        "enabled_by_output": dict(by),
    }
    (Path(a.work) / "selection.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")
    work_rel = Path(os.path.relpath(a.work, root)).as_posix()
    print(f"wrote {work_rel}/espei-in.yaml; run:  cd {work_rel} && espei --input espei-in.yaml")


if __name__ == "__main__":
    main()
