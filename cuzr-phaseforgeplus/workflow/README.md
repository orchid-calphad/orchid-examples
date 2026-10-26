# Reproducible Cu–Zr workflow

`cuzr.json` codifies the documented PhaseForge → PhaseForge+ calculation as a dependency graph. It records the input-file hashes, locked environment files, commands, expected outputs, two ORCHID data commits, and historical weighting choices. The runner uses an isolated output directory and writes a receipt for every completed stage. Each receipt records the exact commands, dependency hashes, output hashes, and execution times; the run manifest records software versions and the MD seed used by new runs.

To inspect the workflow or verify its inputs without executing a calculation:

```sh
python scripts/workflow.py plan --stage evaluate_s3
python scripts/workflow.py verify --potential-dir ~/.cache/grace/GRACE-2L-OMAT
```

To recalculate the cached-model invariant gate in a new directory:

```sh
python scripts/workflow.py run --stage cached_gate --run-dir /path/to/new-run --quick-python /path/to/quick-venv/bin/python
```

`cached_figures` adds the overlay figure. Neither cached stage performs MLIP calculations or a fit. The gate is an in-sample check for the tuned model, and its 9/13 result must not be described as independent predictive validation.

The full path is `sqs` → relaxation and liquid MD → `assemble_mlip_tdb` → `b2_variant`, followed by data assembly, MLIP anchors, the five `fit_s0`–`fit_s4` starts, and their `evaluate_s0`–`evaluate_s4` checks. Dependencies run automatically. For example:

```sh
export ORCHID_DATA=/path/to/checkout-at-778cc6bfa67db292e3f917f5ac6d691e0f37fc1a
export ORCHID_DENSE_DATA=/path/to/checkout-containing-the-pinned-dense-grid
python scripts/workflow.py run --stage evaluate_s0 --stage evaluate_s1 --stage evaluate_s2 --stage evaluate_s3 --stage evaluate_s4 \
  --run-dir /path/to/new-full-run \
  --quick-python /path/to/quick-venv/bin/python \
  --mlip-python /path/to/mlip-venv/bin/python \
  --fit-python /path/to/fit-venv/bin/python \
  --potential-dir ~/.cache/grace/GRACE-2L-OMAT --md-seed 2026
```

The commands run sequentially; the historical calculation used parallel starts. Full execution can take many hours. The runner verifies the checkpoint files, points the calculator at that verified GRACE cache, checks the pinned ORCHID checkout and dense-grid digest, and rejects unverified dense-grid overrides. Environment mismatches stop execution unless `--allow-env-drift` is supplied explicitly; the mismatches remain in the run manifest and prevent a claim of exact environment reproduction. No dependencies or checkpoints are installed or downloaded by the runner itself.

A successful stage is reused only with `--resume`, an unchanged manifest/environment, and matching input and output hashes. A failed stage has no success receipt. Use a new run directory when inputs or code change. The original TDBs and cached results remain unchanged. `_paths.py` now supports `ORCHID_WORK`; the runner sets `ORCHID_SEED_CACHED=0` so an uncomputed intermediate cannot silently be replaced by a cached result. Direct use of the original scripts retains their documented cache-seeding behavior.

## Model provenance

`tdb/model-provenance.json` and its JSON Schema record `generation_method`, `workflow_history`, filtering `tags`, the exact TDB digest, parents, the potential, and supporting evidence. `phaseforge` denotes the MLIP-derived model, while `phaseforge_plus` denotes the subsequent fit. The B2 temperature-term variant remains a PhaseForge model variant until a fit is performed. GRACE-2L-OMAT is identified by the six archived checkpoint digests in `grace_2L_OMAT_files.sha256`; no undocumented semantic version is supplied. Generated TDBs receive matching provenance records under the new run's `provenance/` directory. These tags describe a model's origin and are not ESPEI evidence or weighting tags.

## Historical reproduction and new fits

The codified full path reproduces the historical F10b protocol: per-type normalization, formation-enthalpy uncertainty floors, and ZPF weights multiplied by ten. It does not silently substitute the revised ORCHID bundle. A new fit with the individual-uncertainty export requires a new manifest and a review of those historical overrides. Check that bundle with `python espei/check_weights.py --bundle espei`, retain the exported weights unless an override is explicitly documented, and do not load removed parent records alongside their replacements.

The historical liquid-MD random seed was not recorded. New runner executions set and record a seed (default 2026, incremented across liquid compositions), but this does not recover the historical trajectories or guarantee bitwise agreement across hardware. The archived models are identified by their original hashes. The script import checks, provenance verification, and cached gate can be validated without rerunning the expensive MLIP/fit stages; passing these checks is not proof that the full numerical workflow has been rerun. All five fitted starts are preserved for review. The runner does not automatically select a new winner using the historical post-hoc choice of start 3. NUTS scripts remain available as an optional follow-on rather than part of the core generation-and-fit graph.
