#!/usr/bin/env python3
"""Plan, verify, or execute the Cu-Zr PhaseForge workflow with hashed inputs and stage receipts."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

example = Path(__file__).resolve().parents[1]
_chunk_bytes = 1024 * 1024  # bytes read per step when hashing a file (1 MiB)


def sha(path) -> str:
    """SHA-256 hex digest of a file, read in chunks.

    Args:
        path (str | Path): File to hash.

    Returns:
        str: Lowercase hex digest.
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(_chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def expand(text: str, context: dict) -> str:
    """Replace each {key} placeholder with str(value); arguments are never shell-expanded.

    Args:
        text (str): Template text, possibly containing {key} placeholders.
        context (dict): Placeholder values.

    Returns:
        str: Text with every known placeholder replaced.
    """
    for k, v in context.items():
        text = text.replace("{" + k + "}", str(v))
    return text


def order(manifest: dict, targets: list[str]) -> list[dict]:
    """Order the stages needed for the targets so that every stage follows its dependencies.

    Args:
        manifest (dict): Parsed workflow manifest with a "stages" list.
        targets (list[str]): Stage ids to run.

    Returns:
        list[dict]: Stage records in execution order.

    Raises:
        ValueError: On a dependency cycle or an unknown stage id.
    """
    stages = {s["id"]: s for s in manifest["stages"]}
    done: set[str] = set()
    active: set[str] = set()
    out: list[dict] = []

    def visit(name: str) -> None:
        """Append a stage after its dependencies, rejecting cycles and unknown ids."""
        if name in active:
            raise ValueError("Cyclic dependency: " + name)
        if name in done:
            return
        if name not in stages:
            raise ValueError("Unknown stage: " + name)
        active.add(name)
        for dep in stages[name]["depends_on"]:
            visit(dep)
        active.remove(name)
        done.add(name)
        out.append(stages[name])

    for target in targets:
        visit(target)
    return out


def verify(manifest: dict, root=example, potential_dir=None) -> list[str]:
    """Check locked inputs, TDB provenance records and the stage graph against their recorded hashes.

    Args:
        manifest (dict): Parsed workflow manifest.
        root (Path): Example directory that the manifest paths are relative to.
        potential_dir (Path | None): MLIP checkpoint cache directory, or None to skip checkpoint hashes.

    Returns:
        list[str]: Sorted, de-duplicated error messages; empty when everything verifies.
    """
    errors = []
    for rel, digest in manifest["input_sha256"].items():
        p = root / rel
        if not p.is_file() or sha(p) != digest:
            errors.append("Input changed or missing: " + rel)
    provenance = json.loads((root / manifest["potential_manifest"]).read_text(encoding="utf-8"))
    seen = set()
    for rec in provenance["records"]:
        if rec["id"] != "tdb:sha256:" + rec["sha256"]:
            errors.append("ID/hash mismatch: " + rec["path"])
        if rec["id"] in seen:
            errors.append("Duplicate TDB ID: " + rec["id"])
        seen.add(rec["id"])
        if sha(root / rec["path"]) != rec["sha256"]:
            errors.append("TDB hash mismatch: " + rec["path"])
        if rec["generation_method"] not in rec["tags"]:
            errors.append("Missing method tag: " + rec["path"])
        for parent in rec["parents"]:
            if sha(root / parent["path"]) != parent["sha256"]:
                errors.append("Parent TDB changed: " + parent["path"])
        if potential_dir and rec["potential"]:
            for entry in rec["potential"]["checkpoint"]["files"]:
                p = potential_dir / entry["path"]
                if not p.is_file() or sha(p) != entry["sha256"]:
                    errors.append("Potential changed: " + entry["path"])
    order(manifest, [s["id"] for s in manifest["stages"]])
    return sorted(set(errors))


def snapshot(patterns: list[str], context: dict, required: bool = True) -> dict[str, str]:
    """Hash every file matched by the output or input patterns.

    Args:
        patterns (list[str]): Glob patterns with {key} placeholders from context.
        context (dict): Placeholder values, including "run" and "example".
        required (bool): Raise when a pattern matches no file.

    Returns:
        dict[str, str]: Matched file path (as a string) to its SHA-256 digest.

    Raises:
        ValueError: If required is True and a pattern matches no file.
    """
    result = {}
    for pattern in patterns:
        paths = sorted(glob.glob(expand(pattern, context), recursive=True))  # noqa: PTH207  absolute glob pattern; Path.glob cannot take one
        files = [Path(p) for p in paths if Path(p).is_file()]
        if required and not files:
            raise ValueError("No artifact matches " + expand(pattern, context))
        for p in files:
            result[str(p)] = sha(p)
    return result


def environment(python: str, requirements: Path) -> dict:
    """Record the interpreter's package versions and check them against a requirements file.

    Package versions and VCS commits are captured without writing environment variables or credential-bearing URLs.

    Args:
        python (str): Interpreter to query.
        requirements (Path): requirements file with pinned "==" and " @ git+" lines.

    Returns:
        dict: Interpreter, platform and package info, plus "requirement_mismatches" (list of messages).
    """
    code = """import importlib.metadata as m,json,sys,platform
out={}
for d in m.distributions():
 try: direct=json.loads(d.read_text('direct_url.json') or '{}')
 except Exception: direct={}
 out[d.metadata['Name'].lower().replace('_','-')]={'version':d.version,'commit':direct.get('vcs_info',{}).get('commit_id')}
print(json.dumps({'python':sys.version,'platform':platform.platform(),'packages':out}))"""
    info = json.loads(subprocess.check_output([python, "-c", code], text=True))
    errors = []
    for raw in requirements.read_text(encoding="utf-8").splitlines():
        spec = raw.split(" #", 1)[0].strip()
        if not spec or spec.startswith("#"):
            continue
        if "==" in spec:
            name, version = spec.split("==", 1)
            actual = info["packages"].get(name.lower().replace("_", "-"), {})
            if actual.get("version") != version:
                errors.append(name + " expected " + version + " found " + str(actual.get("version")))
        elif " @ git+" in spec:
            name, url = spec.split(" @ git+", 1)
            name = name.split("[")[0].lower().replace("_", "-")
            commit = url.rsplit("@", 1)[-1]
            if info["packages"].get(name, {}).get("commit") != commit:
                errors.append(name + " VCS commit not verified as " + commit)
    info["requirement_mismatches"] = errors
    return info


def check_external(manifest: dict) -> dict:
    """Verify the pinned ORCHID data checkout and the dense-grid file before any stage that needs them.

    Args:
        manifest (dict): Parsed workflow manifest with data_release, dense_path and dense_sha256.

    Returns:
        dict: "data_commit" (checkout HEAD), "dense_file" (path) and "dense_sha256".

    Raises:
        ValueError: If ORCHID_DATA is missing, not the pinned release, dirty, or the dense grid does not match.
    """
    root = Path(os.environ.get("ORCHID_DATA", "__missing__")).expanduser()
    if not root.is_dir():
        raise ValueError("Set ORCHID_DATA to the historical pinned checkout")
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != manifest["data_release"]:
        raise ValueError("ORCHID_DATA is not the pinned historical release; new fits need a new manifest")
    dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if dirty:
        raise ValueError("Pinned ORCHID_DATA has tracked local changes")
    dense = Path(os.environ.get("ORCHID_DENSE_DATA", str(root))).expanduser() / manifest["dense_path"]
    if os.environ.get("ORCHID_DENSE_GRID"):
        raise ValueError("Unverified ORCHID_DENSE_GRID overrides are not allowed in this workflow")
    if not dense.is_file() or sha(dense) != manifest["dense_sha256"]:
        raise ValueError("Dense-grid file does not match the historical pin")
    return {"data_commit": head, "dense_file": str(dense), "dense_sha256": sha(dense)}


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser; flags, defaults and the description are the reproduction interface.

    Returns:
        argparse.ArgumentParser: Parser for the plan, verify and run actions.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["plan", "verify", "run"])
    ap.add_argument("--stage", action="append")
    ap.add_argument("--manifest", type=Path, default=example / "workflow/cuzr.json")
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--resume", action="store_true")
    for env in ["quick", "mlip", "fit"]:
        ap.add_argument("--" + env + "-python", default=sys.executable)
    ap.add_argument("--potential-dir", type=Path)
    ap.add_argument("--md-seed", type=int, default=2026)
    ap.add_argument(
        "--allow-env-drift",
        action="store_true",
        help="Record an explicitly changed environment; this is not exact reproduction",
    )
    return ap


def _resolve_run_dir(run_dir: Path, resume: bool) -> Path:
    """Resolve the run directory and reject locations that would overwrite sources or archived results.

    Args:
        run_dir (Path): Requested run directory.
        resume (bool): Whether an existing run directory may be reused.

    Returns:
        Path: Absolute, resolved run directory.

    Raises:
        ValueError: If the run directory contains the example, lies inside it outside work/, or exists without resume.
    """
    run = run_dir.expanduser().resolve()
    if run == example or example.is_relative_to(run):
        raise ValueError("Run directory cannot contain the example source tree")
    if run.is_relative_to(example) and not run.is_relative_to(example / "work"):
        raise ValueError("Inside the example, outputs must be under work/")
    if run.exists() and not resume:
        raise ValueError("Run directory exists; use a new directory or explicit --resume")
    return run


def _run_header(m: dict, a: argparse.Namespace, stages: list[dict]) -> dict:
    """Build the run header recorded in workflow-run.json, checking environments and external inputs on the way.

    Args:
        m (dict): Parsed workflow manifest.
        a (argparse.Namespace): Parsed command-line arguments.
        stages (list[dict]): Stages of this run, in execution order.

    Returns:
        dict: Header with the manifest hash, seed, per-environment info, potential directory and external inputs.

    Raises:
        ValueError: On an environment mismatch (unless --allow-env-drift), a missing or wrong potential directory,
            or failed external-input checks.
    """
    header = {
        "workflow": m["id"],
        "manifest_sha256": sha(a.manifest),
        "md_seed_for_new_run": a.md_seed,
        "environments": {},
        "external_inputs": None,
        "potential_directory": str(a.potential_dir.resolve()) if a.potential_dir else None,
    }
    for env in sorted({s["environment"] for s in stages}):
        info = environment(getattr(a, env + "_python"), example / m["environments"][env])
        header["environments"][env] = info
        if info["requirement_mismatches"] and not a.allow_env_drift:
            raise ValueError("Environment mismatch in " + env + ": " + str(info["requirement_mismatches"]))
    if any(s["requires_potential"] for s in stages) and not a.potential_dir:
        raise ValueError("MLIP stages require --potential-dir for checkpoint verification")
    if a.potential_dir and a.potential_dir.name != "GRACE-2L-OMAT":
        raise ValueError("Potential directory must be the GRACE-2L-OMAT cache directory")
    if any(s["requires_orchid"] for s in stages):
        header["external_inputs"] = check_external(m)
    return header


def _stage_commands(stage: dict, context: dict) -> list[list[str]]:
    """Expand a stage's command templates into argument lists, one per run (per file for foreach commands).

    Args:
        stage (dict): Stage record with a "commands" list.
        context (dict): Placeholder values for expand().

    Returns:
        list[list[str]]: Argument vectors to execute in order.

    Raises:
        ValueError: If a foreach command matches no input files.
    """
    commands = []
    for c in stage["commands"]:
        if isinstance(c, list):
            commands.append([expand(x, context) for x in c])
            continue
        items = sorted({p for pattern in c["foreach"] for p in glob.glob(expand(pattern, context))})  # noqa: PTH207  absolute glob pattern; Path.glob cannot take one
        if not items:
            raise ValueError("No inputs for " + stage["id"])
        commands.extend([[expand(x, dict(context, item=item)) for x in c["argv"]] for item in items])
    return commands


def _check_mlip_results(stage: dict, outputs: dict[str, str]) -> None:
    """Check the MLIP relaxation and MD result files of a stage.

    Args:
        stage (dict): Stage record; only the "relax" and "liquid_md" stages carry such results.
        outputs (dict[str, str]): Output file path to digest of the stage.

    Raises:
        ValueError: On a non-finite energy, an unconverged relaxation, or a missing MD seed.
    """
    for path in outputs:
        if not path.endswith(("relax.json", "md.json")):
            continue
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        if not math.isfinite(result["energy_eV_atom"]):
            raise ValueError("Non-finite MLIP energy: " + path)
        if stage["id"] == "relax" and not result["converged"]:
            raise ValueError("Unconverged relaxation: " + path)
        if stage["id"] == "liquid_md" and result.get("random_seed") is None:
            raise ValueError("MD seed not recorded: " + path)


def _write_provenance(stage: dict, generated: list[Path], run: Path, m: dict, a: argparse.Namespace) -> Path:
    """Write the provenance file for the TDBs a stage generated, using the archived record as the template.

    The stage receipt, recorded as evidence, is run/receipts/<stage id>.json.

    Args:
        stage (dict): Stage that produced the TDB files.
        generated (list[Path]): Generated TDB files.
        run (Path): Run directory.
        m (dict): Parsed workflow manifest.
        a (argparse.Namespace): Parsed command-line arguments.

    Returns:
        Path: Written provenance file under the run directory.
    """
    templates = json.loads((example / m["potential_manifest"]).read_text(encoding="utf-8"))["records"]
    template = next(r for r in templates if r["generation_method"] == "phaseforge")
    records = []
    for tdb in generated:
        method = "phaseforge_plus" if stage["id"].startswith("fit_") else "phaseforge"
        record = json.loads(json.dumps(template))
        record.update(
            id="tdb:sha256:" + sha(tdb),
            path=str(tdb),
            sha256=sha(tdb),
            generation_method=method,
            workflow_history=["phaseforge", "phaseforge_plus"] if method == "phaseforge_plus" else ["phaseforge"],
            tags=[method, "potential:GRACE-2L-OMAT"],
            variant=stage["id"],
            workflow=str(a.manifest),
            evidence=[str(run / "workflow-run.json"), str(run / "receipts" / (stage["id"] + ".json"))],
            parents=[],
            fit=None,
        )
        if method == "phaseforge_plus":
            parent = run / "tdb/CuZr-PhaseForge-MLIP-B2b.tdb"
            record["parents"] = [{"path": str(parent), "sha256": sha(parent)}]
            record["fit"] = {
                "source_run": stage["id"],
                "selection": "not_selected",
                "data_commit": m["data_release"],
                "weighting_policy": m["weighting_policy"],
                "validation": "pending downstream evaluation",
            }
        else:
            parent = run / "tdb/CuZr-PhaseForge-MLIP.tdb" if stage["id"] == "b2_variant" else example / "tdb/CuZr-94Zen-LB.tdb"
            record["parents"] = [{"path": str(parent), "sha256": sha(parent)}]
        records.append(record)
    provenance = run / "provenance" / (stage["id"] + ".json")
    provenance.parent.mkdir(exist_ok=True)
    provenance.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "scope": "New workflow run; demonstration models pending review.",
                "artifact_root": {"repository": None, "path": str(run)},
                "records": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return provenance


def _run_stage(stage: dict, m: dict, a: argparse.Namespace, run: Path) -> None:
    """Execute one stage, or skip it when its receipt still matches, and write its receipt.

    Args:
        stage (dict): Stage record.
        m (dict): Parsed workflow manifest.
        a (argparse.Namespace): Parsed command-line arguments.
        run (Path): Run directory.

    Raises:
        ValueError: If a dependency, input or output changed, or a result check fails.
    """
    receipt = run / "receipts" / (stage["id"] + ".json")
    context = {"example": example, "run": run, "python": getattr(a, stage["environment"] + "_python")}
    inputs = snapshot(stage["inputs"], context)
    for dep in stage["depends_on"]:
        prior = json.loads((run / "receipts" / (dep + ".json")).read_text(encoding="utf-8"))
        inputs.update(prior["outputs"])
    for path, digest in inputs.items():
        if not Path(path).is_file() or sha(path) != digest:
            raise ValueError("Dependency changed: " + path)
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        if saved["inputs"] != inputs or any(not Path(p).is_file() or sha(p) != v for p, v in saved["outputs"].items()):
            raise ValueError("Stage inputs/outputs changed; use a new run directory")
        return
    commands = _stage_commands(stage, context)
    env = os.environ.copy()
    env.update(ORCHID_WORK=str(run), ORCHID_SEED_CACHED="0")
    if a.potential_dir:
        env["GRACE_CACHE"] = str(a.potential_dir.resolve().parent)
    started = datetime.now(timezone.utc).isoformat()
    for i, argv in enumerate(commands):
        env["ORCHID_MD_SEED"] = str(a.md_seed + i)
        with (run / "logs" / f"{stage['id']}-{i}.log").open("w", encoding="utf-8") as log:
            subprocess.run(argv, cwd=example, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    outputs = snapshot(stage["outputs"], context)
    if stage["requires_potential"]:
        changed = verify(m, potential_dir=a.potential_dir)
        if changed:
            raise ValueError("Inputs changed during execution: " + str(changed))
    if stage["id"] in ("relax", "liquid_md"):
        _check_mlip_results(stage, outputs)
    generated = [Path(path) for path in outputs if path.endswith(".tdb")]
    if generated:
        provenance = _write_provenance(stage, generated, run, m, a)
        outputs[str(provenance)] = sha(provenance)
    data = {
        "stage": stage["id"],
        "commands": commands,
        "inputs": inputs,
        "outputs": outputs,
        "started": started,
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    receipt.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print("Completed " + stage["id"], flush=True)


def main() -> int:
    """Plan, verify or run the workflow as selected on the command line.

    Returns:
        int: Exit status: 0 on success, 1 when input or potential verification reports errors.

    Raises:
        ValueError: On run-directory, environment or stage-consistency failures.
    """
    ap = _build_parser()
    a = ap.parse_args()
    m = json.loads(a.manifest.read_text(encoding="utf-8"))
    targets = a.stage or ["cached_gate"]
    stages = order(m, targets)
    if a.action == "plan":
        print(json.dumps({"workflow": m["id"], "stages": stages, "limits": m["reproducibility_limits"]}, indent=2))
        return 0
    errors = verify(m, potential_dir=a.potential_dir)
    if errors:
        print(json.dumps({"errors": errors}, indent=2))
        return 1
    if a.action == "verify":
        summary = {
            "input_files": len(m["input_sha256"]),
            "tdb_records": 4,
            "potential_verified": bool(a.potential_dir),
            "errors": [],
        }
        print(json.dumps(summary, indent=2))
        return 0
    if not a.run_dir:
        ap.error("run requires a new --run-dir (or --resume of a recorded run)")
    run = _resolve_run_dir(a.run_dir, a.resume)
    header = _run_header(m, a, stages)
    if a.resume:
        previous = json.loads((run / "workflow-run.json").read_text(encoding="utf-8"))
        if previous != header:
            raise ValueError("Run manifest/environment changed; use a new run directory")
    else:
        run.mkdir(parents=True)
        (run / "workflow-run.json").write_text(json.dumps(header, indent=2) + "\n", encoding="utf-8")
    (run / "receipts").mkdir(exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)
    for stage in stages:
        _run_stage(stage, m, a, run)
    return 0


if __name__ == "__main__":
    try:
        status = main()
    except (ValueError, subprocess.CalledProcessError) as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1) from e
    raise SystemExit(status)
