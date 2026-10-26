"""Small explicit preparation steps used by the reproducible workflow runner."""
import argparse
import json
import os
from pathlib import Path
import shutil


def prepare_compounds(root):
    files = sorted((root/'structures').glob('*.vasp'))
    if len(files) != 6:raise ValueError('Expected six compound structures')
    for p in files:
        d=root/'runs/phaseforge/COMPOUNDS'/p.stem;d.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,d/'POSCAR');(d/'wait').write_text('')


def b2_variant(root):
    p=root/'tdb/CuZr-PhaseForge-MLIP.tdb';s=p.read_text()
    old='1*GHSERCU+1*GHSERZR+VV0020*1000;'
    if s.count(old)!=1 or 'FUNCTION VV0022' in s:raise ValueError('Unexpected MLIP TDB layout')
    s=s.replace('FUNCTION VV0021','FUNCTION VV0022 298.15 0.000000; 6000 N !\nFUNCTION VV0021',1)
    s=s.replace(old,'1*GHSERCU+1*GHSERZR+VV0020*1000+VV0022*T;')
    header='$ MLIP-only TDB + linear T term (VV0022, start 0) for CUZR_B2 (Amendment 4 variant)\n'
    (root/'tdb/CuZr-PhaseForge-MLIP-B2b.tdb').write_text(header+s)


def gate_run(root, start):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import gate
    gate.TDBS={'SGTE':gate.ROOT/'tdb/CuZr-94Zen-LB.tdb', 'MLIP-only':root/'tdb/CuZr-PhaseForge-MLIP.tdb', 'tuned':root/f'runs/final2/F10b_s{start}/tuned.tdb'}
    # gate.main takes its printed-data path from ROOT; leave model's unary-table root intact.
    old=gate.ROOT;gate.ROOT=root
    (root/'inputs').mkdir(exist_ok=True)
    shutil.copy2(root/'results/printed_invariants.json',root/'inputs/printed_invariants.json')
    gate.main(root/f'results/gate_s{start}')
    gate.ROOT=old


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('step',choices=['prepare-compounds','b2-variant','gate-run']);ap.add_argument('--start',type=int)
    a=ap.parse_args();root=Path(os.environ['ORCHID_WORK'])
    if a.step=='prepare-compounds':prepare_compounds(root)
    elif a.step=='b2-variant':b2_variant(root)
    else:gate_run(root,a.start)
