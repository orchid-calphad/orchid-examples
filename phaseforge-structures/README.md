# PhaseForge input structures from m_scraps and the ORCHID registry (CPU only)

`scripts/make_structures.py <System>` prepares, on any CPU machine, the structure set that the MLIP step of a one-shot
PhaseForge build consumes (energies and liquid MD need a GPU; this step does not):

| part | content | generator |
|---|---|---|
| endmembers | FCC_A1, BCC_A2, HCP_A3 pure elements (primitive cells) | ASE |
| SQS | FCC_A1, BCC_A2, HCP_A3 at x = 0.25, 0.5, 0.75, 64 atoms | m_scraps (decorated pair/triplet/quadruplet correlations, seed 2026, 4 MPI ranks) |
| liquid starts | 108-atom fcc SQS at x = 0, 0.25, 0.5, 0.75, 1 | m_scraps |
| compounds | line compounds of the ORCHID phase registry (`phases/registry.csv`, status `assigned`) | curated file, else Materials Project hull structure, else AFLOW prototype (see `--help`) |

The layout is that of the Vision/Spark bundles' `inputs/structures/` (plus `relax_list.txt` and a `manifest.json` with
timings, versions, provenance and the compounds that could not be built). A compound with no structure source is listed
under `skipped`; nothing is invented.

```sh
# environment: see requirements.txt (m_scraps built from the pinned commit)
python scripts/make_structures.py Cu-Zr --out out/CuZr \
    --curated ../../orchid-phaseforge-demo/structures --exclude-compounds CUZR_B33   # reproduces the shipped 21-structure set
python scripts/make_structures.py Ni-Ti --out out/NiTi                               # any other binary of the registry
```

Inputs: `--orchid-data` (a checkout of orchid-data, set with `$ORCHID_DATA` or pass explicitly) and `--mp-structures` (default the
Materials Project structures of `../mlip-formation-enthalpy/inputs/structures.json.gz`, which cover only the systems with
experimental formation enthalpies; the hull files carry no coordinates). OQMD structures are not available offline.

## Reproducibility, measured 2026-10-24 on the Mac Studio (M1 Max, 10 CPUs, no GPU)
* Cu-Zr, curated compounds: all 20 SQS/endmember/liquid structures are bit-identical to the ones shipped in
  `orchid-phaseforge-demo/runs/phaseforge` and in the Vision/Spark bundles (m_scraps is deterministic at a fixed seed
  **and fixed MPI rank count**: with `--ranks 8` the objective values agree but 12 of 20 decorations differ), and the 6
  compounds are the same files. Wall time for the whole set about 16 s (SQS 1-2.4 s each, 0.7-1 s for the liquid cells).
* Cu-Zr, compounds built automatically (no `--curated`): 6/6 match the shipped compounds under pymatgen's
  `StructureMatcher` (same topology, starting volumes differ by 1-3 %); Cu51Zr14 is the MP P-6 65-atom model (the demo's
  half-occupied 6j ring), found in space group 174 rather than the registry's 175.
* Ni-Ti: 15 SQS/endmember + 5 liquid cells + 4 of 6 registry compounds (Ni3Ti, NiTi B19 and B19', NiTi B2); Ni3Ti2 and
  Ni4Ti3 have no structure offline. NiTi B19 and B19' receive the same MP structure (the registry assigns both mP4).
