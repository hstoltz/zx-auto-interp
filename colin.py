"""Small adapter to Sukyn/minimality-auto's Circuit and CandidateModel interface.

Run: python colin.py bialgebra.zx certificate.json
Requires minimality-auto only for that command / candidate(), not for ZX search.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

from zx import Matrix, Interpretation, Term, Theory, check, export_colin, join, parse


def from_colin(node):
    node = node.to_json() if hasattr(node, 'to_json') else node
    if 'gen' in node:
        name = node['gen']
        match = re.fullmatch(r'([ZX])_(\d+)_(\d+)_(-?\d+)', name)
        if match:
            colour, n, m, k = match.groups()
            return parse(f'{colour}({n},{m},{k})')
        if name not in ('H','cup','cap'):
            raise ValueError(f'unknown ZX interpretation for {name!r}')
        return parse(name)
    if 'id' in node:
        return parse(f'id({node["id"]})')
    if 'perm' in node:
        return parse(f'perm({",".join(map(str,node["perm"]))})')
    for kind in ('compose','tensor'):
        if kind in node:
            terms = [from_colin(x) for x in node[kind]]
            return join(kind, reversed(terms) if kind == 'compose' else terms)
    raise ValueError('not an expanded Colin circuit')


def exact_value(interpretation, circuit):
    matrix = interpretation.evaluate(from_colin(circuit))
    return {'shape':[matrix.rows,matrix.cols],
            'entries':[[i,j,interpretation.R.encode(v)] for (i,j),v in sorted(matrix.data.items())]}


def exact_equal(interpretation, left, right):
    if left['shape'] != right['shape']:
        return False
    def matrix(value):
        return Matrix(interpretation.R,*value['shape'],{(i,j):interpretation.R.decode(v) for i,j,v in value['entries']})
    return not matrix(left).difference(matrix(right))


def candidate(interpretation):
    from minimality_auto.search import CandidateModel
    interpretation.validate()
    return CandidateModel(description=f'exact Mat_{interpretation.R}, dimension {interpretation.d}',
                          parameters=interpretation.as_dict(),
                          evaluate=lambda circuit: exact_value(interpretation,circuit),
                          equal=lambda left,right: exact_equal(interpretation,left,right),
                          certified=True)


def main():
    from minimality_auto.core import Theory as ColinTheory
    from minimality_auto.search import Deadline, verify
    if len(sys.argv) != 3:
        raise SystemExit('usage: python colin.py RULES.zx CERTIFICATE.json')
    theory = Theory.read(Path(sys.argv[1]).read_text())
    saved = json.loads(Path(sys.argv[2]).read_text())
    if saved['theory_sha256'] != hashlib.sha256(theory.text.encode()).hexdigest():
        raise ValueError('certificate/rules mismatch')
    interpretation = Interpretation.from_dict(saved['interpretation'])
    if not check(theory,saved['target'],interpretation)['valid']:
        raise ValueError('invalid certificate')
    colin_theory = ColinTheory.from_json(export_colin(theory,interpretation.d))
    result = verify(candidate(interpretation),colin_theory,colin_theory.equation(saved['target']),
                    Deadline.after(60),strategy='zx_exact')
    if result is None:
        raise ValueError('upstream verifier rejected the candidate')
    print(json.dumps(result.as_dict(),indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
