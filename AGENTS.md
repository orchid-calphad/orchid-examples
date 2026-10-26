# Working with ORCHID examples

Use the data commit pinned by each example. `ORCHID_DATA` selects an explicit checkout;
it does not authorize changing the scientific baseline. Record the data commit,
model hashes, environment, seeds, and generated outputs when running a new analysis.

Run the cached quick path first: `sh EXAMPLE/run_quick.sh`. Use that example's
`requirements.txt` in a dedicated environment. From the repository root,
`python tests/check_quick_paths.py EXAMPLE` checks regenerated outputs and
`python tests/check_hygiene.py` checks files tracked by Git. The latter does not
validate untracked additions until they are staged. CI also validates CITATION.cff.

Full Cu–Zr runs use `cuzr-phaseforgeplus/scripts/workflow.py`; inspect its `--help`
and `workflow/README.md` before execution. Verify the locked inputs and model
provenance. Use a separate run directory and retain stage receipts. GPU/MD runs
are distinct from reproduction of cached figures. Ni–Al is optional, unfinished
research and is not a release requirement.

Extraction acceptance is not fitting eligibility. Respect `espei_ready`,
`espei_certified`, gate failures, and specification gaps. Records with
`human_review_correction_requires_downstream_revalidation` remain excluded from
fitting until revalidated. Preserve correction before/after values and source
locators. AI-assisted source adjudication is not independent human re-review.

Count a shared observation once: follow `duplicate_of` and
`evidence_identity.canonical_id`, and retain `merged_ids` / `merged_pids` as aliases.
Keep alternate-unit representations and duplicates together in evaluation splits.
Use the released `fold_lso` / `fold_lpo` for system/paper holdouts and audit cross-paper
aliases for leakage. Keep experimental, model, DFT, and assessed-diagram evidence
separate; do not average Materials Project and OQMD energy references.

MCP access is provided by the separate `orchid-mcp` package. Set
`ORCHID_DATA_DIR` and `ORCHID_SITE_DATA_DIR` for local data. Request full records
when tracing provenance. The cached `quickstart/mcp_exchange.json` is a historical
example, not a live validation; `${...}` paths in shared snapshots are placeholders.

Preserve in-sample and negative-result qualifications. Cite the data release and
original sources; placeholder Zenodo DOIs are not published identifiers. Do not
commit publisher PDFs, credentials, model checkpoints, private paths, or large MD
trajectories. Preserve concurrent edits and stage only reviewed files.
