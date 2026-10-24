# Contributing

- **Data** (a wrong value, a missing paper, new values, TDB files): contribute to the ORCHID data repository,
  https://github.com/orchid-calphad/orchid-data (see its `CONTRIBUTING.md`; every value page has a "report" link).
  The examples here only read released data at pinned commits and are not the place to fix values.
- **Examples** (a bug, a clearer explanation, a new workflow, scores of another MLIP for `oneshot-benchmark/`): open an
  issue or a pull request here. A new example needs a README (what it shows; quick path from cached outputs; full
  path with compute requirements), pinned requirements, a `run_quick.sh` that regenerates its figures in minutes
  without a GPU, `figures/SHA256SUMS`, and an entry in `tests/examples.json`; `python tests/check_quick_paths.py`
  must pass. Keep files under 50 MB and the repository under 100 MB (large outputs go to Zenodo,
  `ZENODO_LARGE_FILES.md`), use repository-relative paths only, and add no restricted material (publisher PDFs, page
  images, handbook drawings or text).
- Code contributions are MIT-licensed; sign off your commits (`git commit -s`, Developer Certificate of Origin), as in
  the other ORCHID repositories.
