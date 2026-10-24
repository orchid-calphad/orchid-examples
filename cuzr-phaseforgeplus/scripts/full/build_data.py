# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/build_data.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""Copied from orchid-espei-demo scripts/v2/build_data_v2.py (same selection, hold-out and weights, so the two demos use
identical data), with three changes for the PhaseForge demo: output under data/, literature DFT goes to a separate
`dft_check` split (held-out check of the MLIP energies, never fitted: the MLIP supplies the energies), and the target
TDB is CuZr-PhaseForge-MLIP.tdb, whose phase names and sublattice models are those of CuZr-94Zen-LB.tdb (so the
registry mapping to CuZr-94Zen-LB applies unchanged).

Original docstring: v2 (2026-10-23 PM): assemble the Cu-Zr working datasets from the FIXED ORCHID ESPEI bundle (E1-E6 applied in the
bundle) plus the optional phase constraints (invariant brackets, assessed assemblages, liquid stability).

Selection (fixed before any v2 run; PLAN.md Amendment 2):
  phase evidence: experimental/CU-ZR ZPF, invariants/CU-ZR, derived/invariant-brackets/CU-ZR, assessed/CU-ZR/CuZr-94Zen-LB.*,
    derived/liquid-stability/CU-ZR/CuZr-94Zen-LB.* -- a dataset is used when every phase name it uses maps, in the
    ORCHID phase registry, to the same registry phase in its own folder's TDB and in CuZr-94Zen-LB.tdb (for a bracket,
    also every phase of the invariant dataset it was derived from)
  thermochemistry: experimental ACR (Zaitsev 2003 via Wang 2006; liquid reference states, bundle E5), HMR/SMR (Turchanin
    2008) with the bundle's optional minimum-sigma weight (orchid_weight_floor, E6); CuZr-94Zen-LB hmr-07/08 excluded
    (phase LIQUID at 298 K at compound compositions: ambiguous label, as in v1)
  DFT: dft/literature/CU-ZR/CuZr-94Zen-LB.* (Zhou 2010; calibrated sigma, weight per value as written)
Hold-out (as in v1, plus invariant temperatures): ZPF datasets whose reference is Glimois 1985 or the Liu 2017 table, and
  the brackets derived from them; activities at 1223, 1304, 1453, 1573, 1723 K; HMR/SMR of Cu10Zr7.
Per-type weights: each data type's dataset weights are multiplied by w_t = sqrt(Nbar / N_t) (N_t = residuals of type t in
  the fit set, Nbar = mean over types) so that each type's summed log-likelihood scales as 1/N_t, i.e. contributes
  comparably (ESPEI's sigma is default / weight, so the log-likelihood scales with weight^2).
"""
import json, glob, os, copy, hashlib, collections, math, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import ROOT as _R, SGTE_TDB, orchid  # noqa: E402
ROOT = str(_R)
B = str(orchid('espei'))
REG = str(orchid('phases/systems/CU-ZR.json'))
TDB = str(SGTE_TDB)
TARGET = 'CuZr-94Zen-LB'
HOLD_REFS = ('10.1016/0022-5088(85)90279-6', '10.1007/s11669-017-0522-2')
HOLD_ACR_T = {1223.0, 1304.0, 1453.0, 1573.0, 1723.0}
HOLD_HMR_PHASE = {'CU10ZR7'}
EXCLUDE = {'experimental/CU-ZR/CuZr-94Zen-LB.hmr-07.json': 'phase LIQUID at 298.15 K at compound compositions (ambiguous label)',
           'experimental/CU-ZR/CuZr-94Zen-LB.hmr-08.json': 'phase LIQUID at 298.15 K at compound compositions (ambiguous label)'}


def sha(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()


reg = json.load(open(REG))
pid = collections.defaultdict(dict)          # stem -> phase_name -> registry id
for p in reg['phases']:
    for m in p.get('models') or []:
        pid[m['tdb'][:-4]][m['phase_name']] = p['id']


def stem_of(path):
    return os.path.basename(path).rsplit('.', 2)[0] if not os.path.basename(path).count('CU-ZR-p') else os.path.basename(path).split('.')[0]


def compatible(d, stem):
    for ph in d['phases']:
        a, b = pid.get(stem, {}).get(ph), pid[TARGET].get(ph)
        if a is None or b is None or a != b:
            return False
    return True


def kind(rel, d):
    if rel.startswith('derived/invariant-brackets'): return 'ZPF_bracket'
    if rel.startswith('derived/liquid-stability'): return 'ZPF_liquid'
    if rel.startswith('assessed/'): return 'ZPF_assessed'
    if rel.startswith('dft/'): return 'DFT_HM_FORM'
    if d['output'] == 'ZPF': return 'ZPF_literature'
    if d['output'].startswith('ACR'): return 'ACR'
    return d['output']


def nres(d):
    if d['output'] == 'ZPF':
        return sum(1 for r in d['values'] for v in r if v[0] != '__HYPERPLANE__')
    return sum(len(c) for a in d['values'] for c in a)


def is_holdout_src(d):
    return any(r in (d.get('reference') or '') for r in HOLD_REFS)


cands = []
for g in ('experimental', 'invariants', 'derived/invariant-brackets'):
    cands += sorted(glob.glob(f'{B}/{g}/CU-ZR/*.json'))
cands += sorted(glob.glob(f'{B}/assessed/CU-ZR/{TARGET}.*.json')) + sorted(glob.glob(f'{B}/derived/liquid-stability/CU-ZR/{TARGET}.*.json'))
cands += sorted(glob.glob(f'{B}/dft/literature/CU-ZR/{TARGET}.*.json'))
man = {'bundle': B, 'tdb': {'path': TDB, 'sha256': sha(TDB)}, 'datasets': [], 'excluded': [], 'doc': __doc__}
sel = []
for f in cands:
    rel = os.path.relpath(f, B)
    d = json.load(open(f))
    if rel in EXCLUDE:
        man['excluded'].append({'source': rel, 'reason': EXCLUDE[rel]}); continue
    if 'model' in (d.get('tags') or []):
        man['excluded'].append({'source': rel, 'reason': 'model evidence class'}); continue
    st = TARGET if rel.startswith(('assessed/', 'derived/liquid', 'dft/')) else stem_of(rel)
    if not compatible(d, st):
        man['excluded'].append({'source': rel, 'reason': 'a phase name does not map to the same registry phase in %s.tdb' % TARGET}); continue
    if rel.startswith('derived/invariant-brackets'):
        if not os.path.exists(os.path.join(B, d['orchid']['derived_from'])):
            man['excluded'].append({'source': rel, 'reason': 'bracket source dataset no longer in the bundle (removed in the cleanup)'}); continue
        src = json.load(open(os.path.join(B, d['orchid']['derived_from'])))
        if not compatible(src, st):
            man['excluded'].append({'source': rel, 'reason': 'bracket derived from a dataset whose phase names do not all map to %s.tdb (source excluded)' % TARGET}); continue
    k = kind(rel, d)
    w = copy.deepcopy(d)
    part = 'fit'
    if k in ('ZPF_literature',) and is_holdout_src(d): part = 'holdout'
    if k == 'DFT_HM_FORM': part = 'dft_check'
    if k == 'ZPF_bracket':
        src = json.load(open(os.path.join(B, d['orchid']['derived_from'])))
        if is_holdout_src(src): part = 'holdout'
    if k == 'ACR' and float(d['conditions']['T']) in HOLD_ACR_T: part = 'holdout'
    if k in ('HMR', 'SMR', 'HM_FORM', 'SM_FORM'):   # bundle renamed HMR/SMR -> HM_FORM/SM_FORM
        if set(d['phases']) & HOLD_HMR_PHASE: part = 'holdout'
        fl = d.get('orchid_weight_floor')
        if fl:
            w['weight'] = fl['suggested_weight']; w['orchid_demo_note'] = 'weight = orchid_weight_floor.suggested_weight (bundle E6)'
    sel.append((rel, k, part, w))
N = collections.Counter()
for rel, k, part, w in sel:
    if part == 'fit': N[k] += nres(w)
Nbar = sum(N.values()) / len(N)
WT_V2 = {k: math.sqrt(Nbar / n) for k, n in N.items()}   # orchid-espei-demo v2 rule (recorded only)
WT = {k: 1.0 for k in N}   # PhaseForge demo: bundle weights unchanged (PhaseForgePlus/paper configuration; PLAN.md amendment 1)
for sub in ('fit', 'holdout', 'dft_check'):
    os.makedirs(f'{ROOT}/data/{sub}', exist_ok=True)
    for f in glob.glob(f'{ROOT}/data/{sub}/*.json'): os.remove(f)
for rel, k, part, w in sel:
    base = w.get('weight', 1.0)
    if isinstance(base, list):
        w['weight'] = [x * WT.get(k, 1.0) for x in base]
    else:
        w['weight'] = base * WT.get(k, 1.0)
    w['orchid_demo_type_weight'] = WT.get(k, 1.0)
    out = f'{ROOT}/data/{part}/' + rel.replace('/', '__')
    json.dump(w, open(out, 'w'), indent=1)
    man['datasets'].append({'file': os.path.relpath(out, ROOT), 'source': rel, 'source_sha256': sha(os.path.join(B, rel)),
                            'split': part, 'type': k, 'n_residuals': nres(w), 'reference': w.get('reference'),
                            'bundle_weight': base, 'type_weight': WT.get(k, 1.0), 'tags': w.get('tags')})
man['type_counts_fit'] = dict(N)
man['type_weights'] = WT
man['type_weight_rule'] = 'w_t = 1 (bundle weights); v2 rule sqrt(Nbar/N_t) recorded in type_weights_v2_not_used, Nbar = %.2f' % Nbar
man['type_weights_v2_not_used'] = WT_V2
json.dump(man, open(f'{ROOT}/data/manifest.json', 'w'), indent=1)
c = collections.Counter((x['split'], x['type']) for x in man['datasets'])
n = collections.Counter()
for x in man['datasets']: n[(x['split'], x['type'])] += x['n_residuals']
for key in sorted(c): print(key, 'datasets', c[key], 'residuals', n[key], 'type weight %.3f' % WT.get(key[1], 1))
print('excluded', len(man['excluded']))
