#!/usr/bin/env python3
"""Run a small ring/dimension survey, or recheck every saved witness."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from zx import Ring, Interpretation, Theory, check, search

HERE = Path(__file__).parent
PRIMES = [3,5,7,11,13,17,19,23,29,31,37,41]


def jobs():
    rings = ['F2']+[f'F{p}' for p in PRIMES]
    rings += ['F4','F8','F9','F16','F25','F27','F49','F81']
    rings += [f'F{p}eps' for p in PRIMES]
    rings += ['F3eps3','F3eps4','F5eps3','F7eps3','F9eps','F9eps3']
    rings += ['Z4','Z8','Z9','Z15','Z25','Z27','Z49','Z81','Z125','Z243','Z625']
    rings += ['F3split','F5split','F9split']
    for ring in rings: yield ('phase_free.zx',ring,4,1)
    for ring,d in [('F9',2),('F3',2),('F5',3),('F9',3),('F9',5),('F5',6),('F7',6)]:
        yield ('phase_free.zx',ring,d,1)
    for ring in ['F5eps','F9eps','F13eps','F17eps','F29eps','F37eps','F41eps']:
        yield ('bialgebra.zx',ring,4,0)
    for ring in ['F5eps','F9','F9eps','F13eps','F17eps','Z25']:
        yield ('bialgebra.zx',ring,4,4)
    for ring in ['F9','F17','F25','F49','F81','F9eps','F17eps','F25eps','F49eps','F81eps','F41eps','F73','F89','F97','Z289']:
        yield ('bialgebra.zx',ring,8,4)


def run(job):
    rules,ring,d,period,seconds = job
    theory = Theory.read((HERE/rules).read_text())
    result = search(theory,'B',Ring.parse(ring),d,period,seconds=seconds,max_bases=10000)
    return {'rules':rules,'ring':ring,'dimension':d,'period':period,
            'limits':{'seconds':seconds,'max_bases':10000},'result':result}


def recheck(rows):
    checked = 0
    for row in rows:
        saved=row['result']
        if not saved['valid']: continue
        theory=Theory.read((HERE/row['rules']).read_text())
        if saved['theory_sha256'] != hashlib.sha256(theory.text.encode()).hexdigest():
            raise ValueError('changed rules: '+row['rules'])
        verified=check(theory,saved['target'],Interpretation.from_dict(saved['interpretation']))
        if not verified['valid'] or verified['witness'] != saved['witness']:
            raise ValueError('invalid or changed witness: '+row['ring'])
        checked += 1
    return checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',type=Path)
    parser.add_argument('--out',type=Path,default=Path('interpretations.json'))
    parser.add_argument('--seconds',type=float,default=5)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.check:
        rows=json.loads(args.check.read_text()); print('Rechecked:',recheck(rows)); return
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(run,[(*j,args.seconds) for j in jobs()]):
            rows.append(row); r=row['result']
            print(row['rules'],row['ring'],f"d={row['dimension']}",f"period={row['period']}",
                  'FOUND' if r['valid'] else r['status'],flush=True)
            args.out.write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    print(f'{recheck(rows)} verified witnesses / {len(rows)} configurations')


if __name__=='__main__': main()
