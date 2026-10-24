# Full-path script of the ORCHID PhaseForge+ example, copied from orchid-phaseforge-demo/scripts/printed_invariants.py;
# only the paths changed (see _paths.py). Outputs go under work/.
"""(Copied from orchid-espei-demo scripts/v2/printed_invariants.py, paths adapted.) ORCHID printed invariant temperatures for Cu-Zr in the v2 fit / hold-out sets, matched to the model reactions
(invariants.REACTIONS) by phase set (CuZr-94Zen-LB names). Writes results/v2/printed_invariants.json."""
import json, glob, gzip, os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import ROOT as _R, orchid  # noqa: E402  (also puts scripts/ with model.py on sys.path)
import model as iv  # noqa: E402
ROOT = str(_R)
pub = {}
for l in gzip.open(orchid('literature/orchid-literature.jsonl.gz'), 'rt'):
    r = json.loads(l)
    if r.get('system') == 'CU-ZR': pub[r['id']] = r
rows = []
for split in ('fit', 'holdout'):
    for f in sorted(glob.glob(f'{ROOT}/data/{split}/*.json')):
        b = os.path.basename(f)
        if not (b.startswith('experimental__') or b.startswith('invariants__')): continue
        d = json.load(open(f))
        if d['output'] != 'ZPF': continue
        Ts = d['conditions']['T'] if isinstance(d['conditions']['T'], list) else [d['conditions']['T']] * len(d['values'])
        for T, reg, ob in zip(Ts, d['values'], d['orchid']['observations']):
            r = pub.get(ob['id'], {})
            if r.get('observation_type') != 'invariant_reaction': continue
            ph = {v[0] for v in reg if v[0] != '__HYPERPLANE__'}
            rx = [x[0] for x in iv.REACTIONS if x[4] == ph]
            tu = r.get('temperature_uncertainty') or {}
            u = tu['value'] if tu.get('basis') in ('printed', 'printed_range') and tu.get('value') else None
            rows.append({'reaction': rx[0] if rx else None, 'phases': sorted(ph), 'T_K': T, 'u_K': u, 'u_used_K': u or 5.0,
                         'split': split, 'id': ob['id'], 'also_reported_as': [a.get('id') for a in ob.get('also_reported_as', [])],
                         'printed': (r.get('reaction') or {}).get('printed'), 'reference': d.get('reference'), 'file': b})
os.makedirs(f'{ROOT}/results', exist_ok=True)
json.dump(rows, open(f'{ROOT}/results/printed_invariants.json', 'w'), indent=1)
c = collections.defaultdict(list)
for r in rows: c[(r['reaction'], r['split'])].append(r['T_K'])
for k in sorted(c, key=str): print(k, sorted(c[k]))
