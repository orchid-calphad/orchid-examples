"""v4 m7 (Amendment 4 fallback, model-adequacy test): m6 + third liquid term 2L (a only, start 0, bound +-40 kJ/mol), run
at ZPF factor x30 (the v4 factor whose optimum fails the fewest gate reactions: 3 of 13), starts SGTE, P2, P4.
Same data (data/v4_m7 -> data/v4_w30), hold-out, gate and metric functions as summarize_v4.py (analyze_v4.gate, .metrics,
.inv_job). Writes results/v4_m7/summary_v4_m7.{json,md} and tdb/v4/CuZr-94Zen-LB_ORCHID-v4_m7_REVIEW-ONLY.tdb.
No Laplace / Hessian (as summarize_v4.py)."""
import os, sys, json, glob
import numpy as np
os.environ['V4_MODEL'] = 'm7'; os.environ['V4_TAG'] = 'v4_m7'
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.join(HERE, '..', 'v3'), os.path.join(HERE, '..', 'v2')]
import analyze_v4 as A
import model_v4 as mst
assert mst.MODEL == 'm7'
ROOT = mst.ROOT; os.chdir(ROOT)
TAG = 'v4_m7'; A.TAG = TAG
KEY = ['L = CuZr (congruent)', 'L = Cu10Zr7 (congruent)']


def err(invd, split):
    e = [abs(invd[r['reaction']] - r['T_K']) for r in A.PRINTED if r['split'] == split and r['reaction'] and invd.get(r['reaction']) is not None]
    return float(np.mean(e)) if e else None


inv0 = A.inv_job(mst.START); ok0, _, det0 = A.gate([inv0])
R = [json.load(open(f)) for f in sorted(glob.glob(f'runs/{TAG}/*/result.json'))]
best = max(R, key=lambda r: r['lnL_opt']); th = np.array(best['theta_opt'])
tr = [json.loads(l) for l in open(f"runs/{TAG}/{best['name']}/trace.jsonl")]
bt = max((t for t in tr if t['lnL'] is not None), key=lambda t: t['lnL'])
inv = A.inv_job(th); ok, per, det = A.gate([inv])
inv_all = {r['name']: A.inv_job(r['theta_opt']) for r in R}
M = A.metrics({'start': mst.START, 'optimum': th})
def rm(k, s): return M.get(k, {}).get(s, {}).get('rmse')
row = {'model': 'm7', 'zpf_factor': 30, 'data': 'data/v4_w30 (data/v4_m7 is a symlink)',
       'starts': {r['name']: {'lnL': r['lnL_opt'], 'theta': r['theta_opt'], 'message': r['message'], 'nit': r['nit'], 'nfev': r['nfev'], 'wall_s': r['wall_s'],
                              'gate_reactions_pass': int(sum(v['pass'] for v in A.gate([inv_all[r['name']]])[2][0].values()))} for r in R},
       'best_start': best['name'], 'symbols': mst.SYMS, 'theta_opt': th.tolist(), 'lnL_opt': best['lnL_opt'],
       'at_bound': [s for s, t, lo, hi, sc in zip(mst.SYMS, th, mst.LO, mst.HI, mst.SCALE) if min(abs(t - lo), abs(t - hi)) < 1e-4 * sc],
       'lnL_per_type_start': tr[0]['per'] if best['name'] == 'SGTE' else json.loads(open(f'runs/{TAG}/SGTE/trace.jsonl').readline())['per'],
       'lnL_per_type_opt': bt['per'],
       'rmse': {k: {'start': rm(k, 'start'), 'optimum': rm(k, 'optimum'), 'unit': M[k]['unit']} for k in M},
       'invariants_opt': {k: (float(v) if v is not None else None) for k, v in inv.items()},
       'invariants_opt_all_starts': {n: {k: (float(v) if v is not None else None) for k, v in d.items()} for n, d in inv_all.items()},
       'invariant_errors': {'fit_mae_K': err(inv, 'fit'), 'holdout_mae_K': err(inv, 'holdout')},
       'gate_pass': bool(ok[0]), 'gate_reactions_pass': int(sum(v['pass'] for v in det[0].values())), 'gate_reactions': len(det[0]),
       'gate_detail': det[0]}
v4 = json.load(open('results/v4/summary_v4.json')); m6 = [v for v in v4['variants'] if v['zpf_factor'] == 30][0]
act_fit = row['rmse']['fit/ACR']['optimum'] < row['rmse']['fit/ACR']['start']
act_ho = row['rmse']['holdout/ACR']['optimum'] < row['rmse']['holdout/ACR']['start']
out = {'status': 'POST-HOC (Amendment 4 fallback m7), model-adequacy test, not a tuned result; v3 is the pre-registered result',
       'selection_rule': 'ZPF factor whose m6 optimum fails the fewest gate reactions (ties: larger activity improvement)',
       'selection': {f"x{v['zpf_factor']}": v['gate_reactions'] - v['gate_reactions_pass'] for v in v4['variants']},
       'start': {'invariants': {k: (float(v) if v is not None else None) for k, v in inv0.items()},
                 'gate_reactions_pass': int(sum(v['pass'] for v in det0[0].values())), 'invariant_errors': {'fit_mae_K': err(inv0, 'fit'), 'holdout_mae_K': err(inv0, 'holdout')}},
       'm7': row,
       'pass_criterion': {'activities_improved_fit': bool(act_fit), 'activities_improved_holdout': bool(act_ho), 'gate_passed': row['gate_pass'],
                          'PASS': bool(act_fit and act_ho and row['gate_pass'])},
       'printed': {k: sorted(r['T_K'] for r in A.PRINTED if r['reaction'] == k and r['split'] == 'fit') for k in KEY}}
os.makedirs(f'results/{TAG}', exist_ok=True)
json.dump(out, open(f'results/{TAG}/summary_v4_m7.json', 'w'), indent=1, default=float)
mst.write(f'tdb/v4/CuZr-94Zen-LB_ORCHID-{TAG}_REVIEW-ONLY.tdb', dict(zip(mst.SYMS, th)),
          'v4 m7 (model-adequacy test, post-hoc) best optimum, ZPF x30. REVIEW ONLY: demonstration, not an assessment.')
L = ['| model | ACR RMSE fit (kJ/mol) | ACR RMSE held-out | ZPF held-out RMSE (J/mol) | CuZr congruent (K) | Cu10Zr7 congruent (K) | invariant MAE fit / held-out (K) | gate reactions passing | gate |',
     '|---|---|---|---|---|---|---|---|---|']
s = row['rmse']
L.append('| SGTE start | %.2f | %.2f | %.0f | %.0f | %.0f | %.0f / %.0f | %d/13 | fail |' % (s['fit/ACR']['start'] / 1e3, s['holdout/ACR']['start'] / 1e3, s['holdout/ZPF_literature']['start'],
         inv0[KEY[0]], inv0[KEY[1]], out['start']['invariant_errors']['fit_mae_K'], out['start']['invariant_errors']['holdout_mae_K'], out['start']['gate_reactions_pass']))
def line(lab, r):
    q = r['rmse']; iv_ = r['invariants_opt']
    return '| %s | %.2f | %.2f | %.0f | %s | %s | %.0f / %.0f | %d/%d | %s |' % (lab, q['fit/ACR']['optimum'] / 1e3, q['holdout/ACR']['optimum'] / 1e3, q['holdout/ZPF_literature']['optimum'],
             '%.0f' % iv_[KEY[0]] if iv_[KEY[0]] else 'absent', '%.0f' % iv_[KEY[1]] if iv_[KEY[1]] else 'absent',
             r['invariant_errors']['fit_mae_K'], r['invariant_errors']['holdout_mae_K'], r['gate_reactions_pass'], r['gate_reactions'], 'PASS' if r['gate_pass'] else 'fail')
L.append(line('m6 x30 (v4)', m6)); L.append(line('m7 x30', row))
L.append('\nm7 best start %s; L2(LIQUID) a = %.0f J/mol; at bound: %s. Pass criterion (activities better fit AND hold-out AND gate): %s.' % (
    best['name'], th[6], row['at_bound'] or 'none', 'PASS' if out['pass_criterion']['PASS'] else 'FAIL'))
L.append('Failing gate reactions (m7): ' + '; '.join('%s %.0f K vs %.0f K (u %.0f)' % (k, v['model_K'], v['nearest_printed_K'], v['u_K']) if v['model_K'] is not None else '%s absent' % k
                                             for k, v in det[0].items() if not v['pass']))
open(f'results/{TAG}/summary_v4_m7.md', 'w').write('\n'.join(L) + '\n\nPrinted (fit): CuZr congruent %s K; Cu10Zr7 congruent %s K.\n' % (out['printed'][KEY[0]], out['printed'][KEY[1]]))
print('\n'.join(L))
