"""v4 models. V4_MODEL=m6: the v3 six-parameter model (tdb/v3/CuZr-94Zen-LB_param_v3.tdb). V4_MODEL=m7: m6 plus a third
liquid interaction term 2L (a only, VV0006; start 0), tdb/v4/CuZr-94Zen-LB_param_v4_m7.tdb (model-adequacy test).
fullp(p) maps onto the v2 16-vector (+ L2 as element 16 for m7) used by invariants_v4."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(HERE, '..', 'v3')]
import make_start_tdb_v3 as m3
ROOT = m3.ROOT
MODEL = os.environ.get('V4_MODEL', 'm6')
if MODEL == 'm6':
    SYMS, START, SCALE, TDB = m3.SYMS, m3.START.copy(), m3.SCALE.copy(), f'{ROOT}/tdb/v3/CuZr-94Zen-LB_param_v3.tdb'
    LO = np.array([-120000, -30, -40000, -20, -241750 - 17 * 15000, -20104.24 - 2 * 15000])
    HI = np.array([0, 40, 40000, 20, -241750 + 17 * 15000, -20104.24 + 2 * 15000])
else:
    SYMS = m3.SYMS + ['VV0006']; START = np.append(m3.START, 0.0); SCALE = np.append(m3.SCALE, 1000.0)
    TDB = f'{ROOT}/tdb/v4/CuZr-94Zen-LB_param_v4_m7.tdb'
    LO = np.array([-120000, -30, -40000, -20, -241750 - 17 * 15000, -20104.24 - 2 * 15000, -40000])
    HI = np.array([0, 40, 40000, 20, -241750 + 17 * 15000, -20104.24 + 2 * 15000, 40000])


def fullp(p):
    q = m3.full16(np.asarray(p)[:6])
    return np.append(q, p[6]) if len(p) > 6 else q


def write(path, values, label):
    m3.write(path, values, label)
    if MODEL != 'm6':
        txt = open(path).read()
        old = 'PAR  L(LIQUID,CU,ZR;1),,               VV0002+VV0003*T;,,         N 94Zen !\n'
        assert txt.count(old) == 1
        txt = txt.replace(old, old + ' PAR  L(LIQUID,CU,ZR;2),,               VV0006;,,         N demo !\n')
        txt = txt.replace(' TEMP-LIM', f'$ VV0006: L2(LIQUID) a (added term, model-adequacy test)\n FUNCTION VV0006    298.15 {values["VV0006"]:+.6f}; 6000 N !\n TEMP-LIM', 1)
        open(path, 'w').write(txt)


if __name__ == '__main__':
    os.makedirs(f'{ROOT}/tdb/v4', exist_ok=True)
    write(TDB, dict(zip(SYMS, START)), 'Starting model: SGTE values + L2 = 0 (v4 m7).')
    print(TDB)
