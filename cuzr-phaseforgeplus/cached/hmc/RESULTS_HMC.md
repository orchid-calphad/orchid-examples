# Gradient-informed HMC (NUTS) for the final PhaseForge+ Cu-Zr model (PLAN.md Amendment 5, post-hoc, PI-directed)

Protocol: C. Kunselman, B. Bocklund, R. Otis, R. Arróyave, "Gradient-informed Hamiltonian Monte Carlo for multicomponent
CALPHAD model optimization and uncertainty quantification", Acta Mater. 315 (2026) 122412,
doi:10.1016/j.actamat.2026.122412. NUTS from C. Kunselman's littlemcmc fork, github.com/cjkunselman18/littlemcmc @
2917beed02f7ddee7258aa7b987e87c51391b584 ("Implemented callback"; fork of eigenfoo/littlemcmc 0.2.2), installed in
the fit environment (requirements-fit.txt) with ESPEI PR #268 9039ea2 + pycalphad 9a6939d (gradients). Her repositories (littlemcmc fork; ESPEI
branches Gradient-based, Gradient-based_update, gradients) contain no HMC driver or example, so the driver is ours
(`scripts/hmc_nuts.py`); it uses the fork's callback (called after each NUTS iteration with a Draw record) for logging and
wall-time stops.

Target: the final model, runs/final2/F10b_s3 = tdb/CuZr-PhaseForge-final.tdb (release 778cc6bfa, data/a4d_F10 +
data/mlip_anchor, F10b configuration, 23 parameters, L-BFGS-B lnL -5736.17, reproduced exactly by the driver).
logp = ESPEI lnL (the fit's objective) with the paper's improper uniform prior p(theta) = 1; parameters in the TDB's VV
units (constants kJ/mol = the paper's 1/1000 scaling; T terms J/mol/K). NUTS: target acceptance 0.8, max tree depth 6,
diagonal mass matrix with active adaptation. Cost: 3.2-3.9 s per lnL + gradient evaluation (5-7 processes on the Mac
Studio); the value alone costs 3.5 s, so the gradient adds about 5 %.
Earlier runs on runs/final/F10b_s1 (runs/hmc_test_F10b_s1/) were a test of the pipeline only (superseded by the refit).

## (a) Exploration: four NUTS chains + BFGS (paper's Cr-Fe protocol)
Four chains with active mass-matrix and step-size tuning for a hard wall of 2 h each (in parallel), then SciPy BFGS from
each chain's best point (1 h wall). Deviations (disclosed): chains start at the L-BFGS-B optima of four refit starts
(s0-s3) rather than at the fit starts, because in the F10b_s1 test, chains from the far fit starts (lnL -1.5e5 to -4e5)
reached only -7,300 to -25,000 after about 2,300 evaluations each, where L-BFGS-B reaches -5,409 in 395; the adapted mass
matrix starts from 1/|H_ii| of the finite-difference Hessian at the final optimum (the Hessian is not positive definite:
the likelihood is piecewise through the ZPF terms) instead of identity. BFGS stopped at its first iteration in all four
chains ("precision loss": the line search fails on the piecewise likelihood), so each chain's best point was also
refined with the fit's own L-BFGS-B (scaled), which converged in 19-65 evaluations with gains of 0-2 lnL units.

| chain (start) | start lnL | NUTS iter. (div.) | best lnL after NUTS + BFGS + L-BFGS-B | evaluations | gate | spurious BCC/HCP points (T, x(Zr)) | CuZr melting (K) |
|---|---|---|---|---|---|---|---|
| explore_3 (s3 = final) | -5736.2 | 51 (13) | -5579.7 | 2,237 | 9/13 | 610 (500-1260 K, 0.05-0.41) | 1310 |
| explore_0 (s0) | -5750.5 | 42 (10) | -5574.5 | 2,252 | 9/13 | 81 (1200-1260 K, 0.05-0.16) | 1405 |
| explore_2 (s2) | -5845.5 | 41 (6) | -5597.0 | 2,319 | 7/13 | 246 (640-1260 K, 0.05-0.41) | 1448 |
| explore_1 (s1) | -5467.1 | 47 (15) | -5466.0 | 2,248 | 8/13 | 592 (500-1260 K, 0.05-0.41) | 1226 |
| final L-BFGS-B (s3) | | | -5736.2 | 409 | 9/13 | 82 (1220-1290 K, 0.05-0.16) | 1312 |

(Gate and validity checks as analyze_runs.py and scripts/validity.py; printed CuZr melting 1206 K.)
NUTS raised lnL by 156-249 units above three of the four L-BFGS-B optima (s0, s2, s3), so those were not local maxima
of the likelihood but stalls of the optimizer on its non-smooth surface; it did not improve the best optimum (s1). No
chain found a mode with higher lnL AND fewer gate misses or no spurious (betaZr) field. The higher-lnL points either open
the large BCC field at x(Zr) up to 0.41 (explore_3, explore_2, explore_1), or (explore_0) keep the same sliver as the final
model (81 vs 82 points) and the same 9/13 gate count (it misses L + Cu51Zr14 = Cu8Zr3 instead of the Cu5Zr
peritectic; both miss the CuZr and CuZr2 melting points and the CuZr eutectoid, 1032 K vs 969 K for the final model), with
a worse CuZr melting point (1405 K vs 1312 K; printed 1206 K). With these data and weights, higher likelihood and physical validity pull apart:
the likelihood rewards the BCC field, so the final model's selection by validity, not lnL, still stands.
Cost: 2,138-2,193 NUTS evaluations per chain (2.0-2.1 h wall per chain, parallel), 9,056 evaluations in all with BFGS and
L-BFGS-B, against 1,762 evaluations for the five L-BFGS-B starts. In the paper (Cr-Fe), NUTS matched ensemble MCMC's best
lnL with 2,602 evaluations against 122,824 (47x fewer); here a comparable budget per chain moved three chains by
150-250 lnL units from L-BFGS-B stalls, but did not find a better overall mode.

## (b) UQ at the final model (paper's Cr-Fe-Ni protocol)
One NUTS chain from the final optimum with active adaptation (target 200 tuning iterations; stopped by its 1.83 h wall
after 49 iterations, 2,051 evaluations, 18 divergences); tuned step size 0.0145, halved to 0.0072 as in the paper; then two
chains with fixed mass matrix and step size from the last tuning point (seeds 1, 2), 2.5 h wall each (planned 200 draws).

| quantity | value |
|---|---|
| draws per chain | 39 (wall-limited; every iteration at the maximum tree depth 6 = 63 evaluations) |
| divergences after tuning | 0; mean acceptance 0.88 |
| split rank-normalised R-hat | max 2.12 (parameters), 1.14 (lnL) |
| bulk ESS | min 2.8, median 3.7 (parameters); 13 (lnL) |
| gradient evaluations | tuning 2,051 + sampling 4,992; 1,340-1,780 per effective sample |
| wall | tuning 1.9 h, then sampling 2.5 h (two chains in parallel) |
| lnL of the draws | mean -5587.1, max -5581.8 (final L-BFGS-B optimum -5736.2) |

The chains did not converge, and the tuning chain left the final optimum: within 20 iterations it had moved to lnL of
about -5590, 150 units above the optimum and up to 24 posterior sd away in some parameters (figure b). The draws
therefore sample a neighbouring region of higher likelihood, not the final model's basin, and that region contains the
large spurious BCC field: 0 of 78 draws pass the validity check (median 455 offending grid points), and no draw passes
all 13 gate reactions (8-10 pass). Under p(theta) = 1 there is no posterior mode at the final model; a prior or a
constraint that excludes the BCC field would be needed to quantify uncertainty about that model.

Invariant temperatures (78 draws; 95 % intervals from draws and from the paper's local expansion, Eqs. 10-11, at the
highest-lnL draw, Jacobian by central differences at +-1 posterior sd, cov(theta) from the draws):

| reaction | final model (K) | draws median [2.5, 97.5 %] (K) | local expansion 95 % (K) | gate pass in draws |
|---|---|---|---|---|
| L = FCC + Cu5Zr (eutectic) | 1262 | 1254 [1249, 1258] | [1251, 1259] | 100 % |
| L + Cu51Zr14 = Cu5Zr (peritectic) | 1330 | 1323 [1320, 1328] | [1320, 1330] | 24 % |
| L = Cu51Zr14 (congruent) | 1420 | 1414 [1412, 1415] | [1414, 1417] | 100 % |
| L + Cu51Zr14 = Cu8Zr3 | 1249 | 1267 [1260, 1275] | [1263, 1280] | 0 % |
| L = Cu8Zr3 + Cu10Zr7 (eutectic) | 1142 | 1143 [1142, 1144] | [1142, 1144] | 100 % |
| L = Cu10Zr7 (congruent) | 1151 | 1151 [1149, 1151] | [1149, 1152] | 100 % |
| L = Cu10Zr7 + CuZr (eutectic) | 1141 | 1139 [1136, 1141] | [1136, 1141] | 100 % |
| L = CuZr (congruent) | 1312 | 1296 [1282, 1307] | [1274, 1308] | 0 % |
| L = CuZr + CuZr2 (eutectic) | 1193 | 1193 [1186, 1195] | [1184, 1197] | 67 % |
| L = CuZr2 (congruent) | 1317 | 1322 [1320, 1325] | [1321, 1326] | 0 % |
| L = CuZr2 + BCC (eutectic) | 1264 | 1266 [1264, 1269] | [1264, 1269] | 100 % |
| BCC = CuZr2 + HCP (eutectoid) | 1095 | 1096 [1093, 1099] | [1092, 1099] | 100 % |
| CuZr = Cu10Zr7 + CuZr2 (eutectoid) | 969 | 977 [974, 980] | [974, 983] | 76 % |

The intervals are narrow (2-34 K) and agree between draws and local expansion, but with R-hat up to 2.1 and ESS of 3-4
they describe the local spread of two short, unconverged chains in the wrong basin, not a posterior; they should not be
reported as uncertainty of the final model. The CuZr melting point stays 76-101 K above the printed 1206 K in every draw.

## Comparison with ensemble MCMC (emcee)
Our earlier emcee runs (orchid-espei-demo): v2, two ensembles of 32 walkers x about 955 iterations (16 parameters),
about 61,000 evaluations, stopped unconverged (split R-hat up to 12, autocorrelation time >= 45 iterations, a lower
bound); v3 check, 12 walkers x 100 iterations (6 parameters), not converged. Neither method converged, so a cost per
effective sample cannot be compared fairly: the nominal NUTS figure here is 1,340-1,780 evaluations per effective sample,
the nominal emcee figure is >= 45 evaluations per effective sample for an unconverged ensemble. The evaluations cost
the same (value 3.5 s, value + gradient 3.7 s). What NUTS did that emcee did not: it moved 150-250 lnL units in about
2,200 evaluations per chain from the L-BFGS-B stalls, where the v2 ensembles were still burning in after 61,000. What
limits NUTS here is the tree depth: the posterior is narrow in the liquid and compound parameters (sd 0.01-0.06 kJ/mol)
and wide in the HCP terms (sd 12-27), so a diagonal mass matrix adapted in 49 iterations needs the full 63 leapfrog
steps per draw at the halved step size; the paper's budget (200 tuning iterations, 400 draws) would take about 20 h of
wall here at 3.5 s per evaluation.

## Files
- `scripts/hmc_nuts.py` (driver: explore / tune / sample), `scripts/hmc_hessian.py`, `scripts/hmc_refine.py`,
  `scripts/hmc_launch_sample.sh`, `scripts/hmc_analyze.py`, `scripts/hmc_figure.py`.
- runs/hmc/ (per chain: evals.jsonl = every lnL + gradient evaluation, iters.jsonl = every NUTS iteration with the point,
  summary.json; refine.json; hessian/hessian.json); runs/hmc_test_F10b_s1/ (pipeline test on the previous final model).
- results/hmc/hmc_results.json (all numbers above, the draws' invariants, Jacobian, R-hat/ESS per parameter).
- figures/fig_hmc_si.pdf / .png (600 dpi): (a) best -lnL vs cumulative evaluations (NUTS offset by the L-BFGS-B cost of
  its start), (b) draws as displacement from the final optimum in posterior sd, (c) invariant temperatures with both 95 %
  intervals and printed fit-set values.
