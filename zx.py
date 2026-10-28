#!/usr/bin/env python3
"""Typed ZX expressions, exact Mat_R semantics, and small interpretation search.

No floating point. Composition is mathematical: A ∘ B applies B first.
The JSON boundary uses Colin's execution-order convention instead.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
from itertools import product
import json
from math import isqrt
from pathlib import Path
import re
import time
import sys
from typing import Any


# Keep one Ring class when the CLI imports the optional ring constructors.
if __name__ == '__main__':
    sys.modules.setdefault('zx', sys.modules[__name__])


class UnsupportedFamily(ValueError):
    pass


@dataclass(frozen=True)
class Ring:
    """F_p, F_p[ε]/ε² (encoded a+p*b), or exact algebraic expressions in C."""
    p: int = 5
    dual: bool = False

    def __post_init__(self):
        if type(self.p) is not int or type(self.dual) is not bool:
            raise ValueError('ring characteristic must be an integer')
        if self.p == 0:
            if self.dual:
                raise ValueError('C[ε]/ε² is not implemented; use a finite characteristic.')
        elif self.p < 2 or any(self.p % n == 0 for n in range(2, isqrt(self.p) + 1)):
            raise ValueError("Ring(p) needs a prime; use Ring.parse('Fq') for extension fields.")

    @classmethod
    def parse(cls, text: str) -> Ring:
        if text == 'C':
            return cls(0)
        from rings import parse
        return parse(text)

    @property
    def order(self):
        return self.p**(2 if self.dual else 1)

    @property
    def is_field(self):
        return bool(self.p) and not self.dual

    def __str__(self):
        return 'C' if not self.p else f'F{self.p}' + ('[ε]/ε²' if self.dual else '')

    def q(self, a):
        if self.p:
            if type(a) is not int:
                raise ValueError('finite-ring entries must be exact integers, not floats')
            return a % (self.p**2 if self.dual else self.p)
        import sympy as sp
        if isinstance(a, str):
            def literal(x):
                if isinstance(x, ast.Constant) and type(x.value) is int:
                    return sp.Integer(x.value)
                if isinstance(x, ast.Name) and x.id == 'I':
                    return sp.I
                if isinstance(x, ast.UnaryOp) and isinstance(x.op, (ast.UAdd, ast.USub)):
                    return literal(x.operand) * (-1 if isinstance(x.op, ast.USub) else 1)
                if isinstance(x, ast.BinOp):
                    left, right = literal(x.left), literal(x.right)
                    operations = {ast.Add: lambda: left+right, ast.Sub: lambda: left-right,
                                  ast.Mult: lambda: left*right, ast.Div: lambda: left/right,
                                  ast.Pow: lambda: left**right}
                    if type(x.op) in operations:
                        return operations[type(x.op)]()
                if (isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                        and x.func.id == 'sqrt' and len(x.args) == 1 and not x.keywords):
                    return sp.sqrt(literal(x.args[0]))
                raise ValueError('use exact integers, I, sqrt, and arithmetic for C entries')
            a = literal(ast.parse(a, mode='eval').body)
        else:
            a = sp.sympify(a)
        if a.atoms(sp.Float) or a.free_symbols:
            raise ValueError('C entries must be exact, closed expressions; no floats or variables.')
        a = sp.simplify(a)
        if a.is_finite is not True or a.is_algebraic is not True or a.is_zero is None:
            raise ValueError('C needs finite algebraic entries with decidable exact zero tests')
        return a

    def scalar(self, n: int):
        return n % self.p if self.p else self.q(n)

    def add(self, a, b):
        if not self.dual:
            return self.q(a + b)
        p = self.p
        return (a % p + b % p) % p + p * ((a // p + b // p) % p)

    def neg(self, a):
        if not self.dual:
            return self.q(-a)
        return (-a % self.p) + self.p * ((-(a // self.p)) % self.p)

    def mul(self, a, b):
        if not self.dual:
            return self.q(a * b)
        p = self.p
        return ((a % p) * (b % p)) % p + p * (((a % p) * (b // p) + (a // p) * (b % p)) % p)

    def inv(self, a):
        if not self.p:
            if a == 0:
                raise ZeroDivisionError('zero is not a unit')
            return self.q(1 / a)
        if a % self.p == 0:
            raise ZeroDivisionError(f'{self.display(a)} is not a unit')
        v = pow(a % self.p, -1, self.p)
        return v + (self.p * ((-(a // self.p) * v * v) % self.p) if self.dual else 0)

    def power(self, a, n: int):
        if n < 0:
            return self.power(self.inv(a), -n)
        out = self.scalar(1)
        while n:
            if n & 1:
                out = self.mul(out, a)
            a, n = self.mul(a, a), n // 2
        return out

    def display(self, a):
        if self.dual:
            head, coefficient = a % self.p, a // self.p
            tail = (str(coefficient) if coefficient != 1 else '') + 'ε'
            return (f'{head}+{tail}' if head else tail) if coefficient else str(head)
        return str(a)

    def encode(self, a):
        return [int(a % self.p), int(a // self.p)] if self.dual else (int(a) if self.p else str(a))

    def decode(self, a):
        if self.dual:
            if not isinstance(a, list) or len(a) != 2 or not all(type(x) is int for x in a):
                raise ValueError('dual entries must be [constant, ε-coefficient] integer pairs')
            return a[0] % self.p + self.p * (a[1] % self.p)
        return self.q(a)


@dataclass
class Matrix:
    R: Ring
    rows: int
    cols: int
    data: dict[tuple[int, int], Any]

    def __post_init__(self):
        if max(self.rows,self.cols,len(self.data)) > 2_000_000:
            raise ValueError('Matrix exceeds the sparse size limit (2,000,000 entries).')
        self.data = {ij: self.R.q(v) for ij, v in self.data.items() if self.R.q(v) != 0}

    def __matmul__(self, other):
        if self.R != other.R or self.cols != other.rows:
            raise ValueError('matrix composition type mismatch')
        by_row: dict[int, list] = {}
        for (k, j), v in other.data.items():
            by_row.setdefault(k, []).append((j, v))
        data: dict = {}
        for (i, k), a in self.data.items():
            for j, b in by_row.get(k, ()):
                data[i, j] = self.R.add(data.get((i, j), 0), self.R.mul(a, b))
        return Matrix(self.R, self.rows, other.cols, data)

    def tensor(self, other):
        if self.R != other.R:
            raise ValueError('different coefficient rings')
        if len(self.data)*len(other.data) > 2_000_000:
            raise ValueError('tensor product exceeds the sparse size limit')
        data = {(i * other.rows + k, j * other.cols + l): self.R.mul(a, b)
                for (i, j), a in self.data.items() for (k, l), b in other.data.items()}
        return Matrix(self.R, self.rows * other.rows, self.cols * other.cols, data)

    def difference(self, other):
        if (self.R, self.rows, self.cols) != (other.R, other.rows, other.cols):
            raise ValueError('matrix subtraction type mismatch')
        return {ij: v for ij in sorted(self.data.keys() | other.data.keys())
                if (v := self.R.add(self.data.get(ij, 0), self.R.neg(other.data.get(ij, 0)))) != 0}

    @property
    def T(self):
        return Matrix(self.R, self.cols, self.rows, {(j, i): v for (i, j), v in self.data.items()})

    @classmethod
    def eye(cls, R, d):
        return cls(R, d, d, {(i, i): R.scalar(1) for i in range(d)})

    def dense(self):
        return tuple(tuple(self.data.get((i, j), 0) for j in range(self.cols)) for i in range(self.rows))


@dataclass(frozen=True)
class Term:
    kind: str
    n: int
    m: int
    name: str = ''
    args: tuple = ()
    parts: tuple[Term, ...] = ()

    def colin(self):
        if self.kind == 'id':
            return {'id': self.n}
        if self.kind == 'perm':
            return {'perm': list(self.args)}
        if self.kind == 'gen':
            return {'gen': self.name}
        parts = reversed(self.parts) if self.kind == 'compose' else self.parts
        return {self.kind: [x.colin() for x in parts]}


def join(kind, parts):
    parts = tuple(parts)
    if not parts:
        return Term('id', 0, 0)
    if len(parts) == 1:
        return parts[0]
    if kind == 'compose':
        for a, b in zip(parts, parts[1:]):
            if b.m != a.n:
                raise ValueError(f'ill-typed composition: {a.n}→{a.m} ∘ {b.n}→{b.m}')
        return Term(kind, parts[-1].n, parts[0].m, parts=parts)
    return Term(kind, sum(x.n for x in parts), sum(x.m for x in parts), parts=parts)


def split_top(text, character):
    depth, start, out = 0, 0, []
    for i, c in enumerate(text):
        if c in '([':
            depth += 1
        elif c in ')]':
            depth -= 1
        if depth < 0:
            raise ValueError('unbalanced brackets')
        if c == character and depth == 0:
            out.append(text[start:i].strip())
            start = i + 1
    if depth:
        raise ValueError('unbalanced brackets')
    return out + [text[start:].strip()]


def integer(text, env):
    def walk(x):
        if isinstance(x, ast.Constant) and type(x.value) is int:
            return x.value
        if isinstance(x, ast.Name) and x.id in env:
            return env[x.id]
        if isinstance(x, ast.UnaryOp) and isinstance(x.op, (ast.UAdd, ast.USub)):
            return walk(x.operand) * (-1 if isinstance(x.op, ast.USub) else 1)
        if isinstance(x, ast.BinOp) and isinstance(x.op, (ast.Add, ast.Sub, ast.Mult)):
            a, b = walk(x.left), walk(x.right)
            return a+b if isinstance(x.op, ast.Add) else (a-b if isinstance(x.op, ast.Sub) else a*b)
        raise ValueError(f'not an integer arity/phase expression: {text!r}')
    return walk(ast.parse(text.strip(), mode='eval').body)


def parse(text: str, env=None, macros=None, stack=()) -> Term:
    env, macros = env or {}, macros or {}
    text = text.strip().replace('◦', '∘')
    for op, kind in [('∘', 'compose'), ('⊗', 'tensor')]:
        parts = split_top(text, op)
        if len(parts) > 1:
            return join(kind, [parse(x, env, macros, stack) for x in parts])
    if text.startswith('(') and text.endswith(')'):
        return parse(text[1:-1], env, macros, stack)
    if text in macros:
        if text in stack:
            raise ValueError('cyclic macro')
        return parse(macros[text], env, macros, (*stack, text))
    match = re.fullmatch(r'([\w]+)(?:\((.*)\))?', text)
    if not match:
        raise ValueError(f'invalid map expression: {text!r}')
    name = {'I': 'id', 'e': 'empty', 'σ': 'swap', 'η': 'cup', 'ε': 'cap'}.get(match[1], match[1])
    args = tuple(integer(x, env) for x in split_top(match[2], ',')) if match[2] else ()
    if name in ('Z', 'X'):
        if len(args) not in (2, 3):
            raise ValueError('Use Z(n,m,k) / X(n,m,k); phase k defaults to zero.')
        n, m, *phase = args
        if min(n, m) < 0:
            raise ValueError('spider arities must be natural numbers, including zero')
        k = phase[0] if phase else 0
        return Term('gen', n, m, f'{name}_{n}_{m}_{k}', (name, n, m, k))
    if name in ('id', 'H'):
        if len(args) > 1:
            raise ValueError('id and H take at most one tensor-power argument')
        n = args[0] if args else 1
        if n < 0:
            raise ValueError('negative tensor power')
        return Term('id', n, n) if name == 'id' else join('tensor', [Term('gen', 1, 1, 'H')] * n)
    if name == 'perm':
        if sorted(args) != list(range(len(args))):
            raise ValueError('perm must list 0,...,n-1 exactly once')
        return Term('perm', len(args), len(args), args=args)
    if args or name not in ('swap', 'cup', 'cap', 'empty'):
        raise ValueError(f'unknown map or wrong arguments: {text!r}')
    return {'swap': Term('perm', 2, 2, args=(1, 0)), 'cup': Term('gen', 0, 2, 'cup'),
            'cap': Term('gen', 2, 0, 'cap'), 'empty': Term('id', 0, 0)}[name]


# These exact family templates follow from the README assumptions. A matching
# NAME alone never licenses a family: variables and both bodies must match.
FAMILIES = {
    'green_fusion': ('p,q,r,s;a,b', '(id(q) ⊗ Z(r+1,s,b)) ∘ (Z(p,q+1,a) ⊗ id(r))', 'Z(p+r,q+s,a+b)'),
    'red_fusion': ('p,q,r,s;a,b', '(id(q) ⊗ X(r+1,s,b)) ∘ (X(p,q+1,a) ⊗ id(r))', 'X(p+r,q+s,a+b)'),
    'colour_change': ('n,m;k', 'X(n,m,k)', 'H(m) ∘ Z(n,m,k) ∘ H(n)'),
    'phase_free_green_fusion': ('p,q,r,s', '(id(q) ⊗ Z(r+1,s)) ∘ (Z(p,q+1) ⊗ id(r))', 'Z(p+r,q+s)'),
    'phase_free_red_fusion': ('p,q,r,s', '(id(q) ⊗ X(r+1,s)) ∘ (X(p,q+1) ⊗ id(r))', 'X(p+r,q+s)'),
}


def compact(s):
    return re.sub(r'\s+', '', s).replace('◦', '∘')


@dataclass(frozen=True)
class Rule:
    name: str
    lhs: str
    rhs: str
    variables: str = ''

    def theorem(self):
        key = tuple(map(compact, (self.variables, self.lhs, self.rhs)))
        for label, template in FAMILIES.items():
            a, l, r = map(compact, template)
            if key in ((a, l, r), (a, r, l)):
                return label
        return None

    def terms(self, macros, env=None):
        a, b = parse(self.lhs, env, macros), parse(self.rhs, env, macros)
        if (a.n, a.m) != (b.n, b.m):
            raise ValueError(f'{self.name}: equation type mismatch {a.n}→{a.m} ≠ {b.n}→{b.m}')
        return a, b


@dataclass
class Theory:
    text: str
    rules: list[Rule]
    macros: dict[str, str]

    @classmethod
    def read(cls, text):
        macros, rules, statements = {}, [], []
        for line in text.splitlines():
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            if ':=' in line or re.match(r'^[^\s:\[\]]+(?:\[[^]]*\])?\s*:', line):
                statements.append(line)
            elif statements:
                statements[-1] += ' ' + line
            else:
                raise ValueError('a rule needs a name and colon')
        for line in statements:
            if ':=' in line:
                name, body = map(str.strip, line.split(':=', 1))
                if not name.isidentifier() or name in macros:
                    raise ValueError('invalid or duplicate macro')
                macros[name] = body
                continue
            match = re.fullmatch(r'([^\s:\[\]]+)(?:\[([^]]*)\])?\s*:\s*(.+)', line)
            if not match:
                raise ValueError(f'invalid rule: {line}')
            name, variables, body = match.groups()
            sides = split_top(body, '=')
            if len(sides) != 2 or name in {r.name for r in rules}:
                raise ValueError('each rule needs a unique name and one equality')
            rules.append(Rule(name, *sides, variables or ''))
        if not rules:
            raise ValueError('empty ruleset')
        result = cls(text, rules, macros)
        for rule in rules:
            if not rule.variables:
                rule.terms(macros)
        return result

    def prepare(self, target):
        if target not in {r.name for r in self.rules}:
            raise ValueError(f'unknown target {target!r}')
        finite, guaranteed = [], {}
        for rule in self.rules:
            if rule.variables:
                theorem = rule.theorem()
                if not theorem:
                    raise UnsupportedFamily(f'{rule.name}: no all-arity proof registered for this family. '
                                            'Finite sampling would not prove independence.')
                if rule.name == target:
                    raise ValueError(f'{target} is guaranteed by this ansatz; choose a different ansatz.')
                guaranteed[rule.name] = theorem
            else:
                finite.append((rule.name, *rule.terms(self.macros)))
        return finite, guaranteed


class Interpretation:
    def __init__(self, R: Ring, H, phases, period=0, *, validate=True):
        self.R, self.d, self.period = R, len(phases), period
        if self.d < 1 or len(H) != self.d or any(len(row) != self.d for row in H):
            raise ValueError('H must be d×d, with d phase units')
        if type(period) is not int or period < 0:
            raise ValueError('period must be nonnegative; zero means integer phases')
        self.H = Matrix(R, self.d, self.d, {(i,j): R.q(v) for i,row in enumerate(H) for j,v in enumerate(row)})
        self.phases = tuple(map(R.q, phases))
        self._cache = {}
        if validate:
            self.validate()

    def validate(self):
        if self.H.difference(self.H.T) or (self.H @ self.H).difference(Matrix.eye(self.R, self.d)):
            raise ValueError('H must satisfy Hᵀ=H and H²=id exactly')
        for t in self.phases:
            self.R.inv(t)
            if self.period and self.R.power(t, self.period) != self.R.scalar(1):
                raise ValueError(f'phase unit does not satisfy t^{self.period}=1')

    def evaluate(self, term: Term) -> Matrix:
        if term in self._cache:
            return self._cache[term]
        R, d = self.R, self.d
        total = term.n + term.m
        if d > 1 and (max(term.n,term.m) > 20 or d**max(term.n,term.m) > 2_000_000):
            raise ValueError('matrix exceeds the small-program size limit')
        if term.kind == 'id':
            out = Matrix.eye(R, d**term.n)
        elif term.kind == 'perm':
            data = {}
            for j, word in enumerate(product(range(d), repeat=term.n)):
                i = 0
                for k in term.args:
                    i = i*d + word[k]
                data[i, j] = R.scalar(1)
            out = Matrix(R, d**term.m, d**term.n, data)
        elif term.kind in ('compose', 'tensor'):
            maps = [self.evaluate(x) for x in term.parts]
            out = maps[0]
            for x in maps[1:]:
                out = out @ x if term.kind == 'compose' else out.tensor(x)
        elif term.name == 'H':
            out = self.H
        elif term.name in ('cup', 'cap'):
            out = Matrix(R, d*d, 1, {(j*d+j, 0): R.scalar(1) for j in range(d)})
            if term.name == 'cap':
                out = out.T
        else:
            colour, n, m, k = term.args
            def index(j, arity):
                return sum(j*d**i for i in range(arity))
            data = {}
            for j, t in enumerate(self.phases):
                ij = index(j,m), index(j,n)
                data[ij] = R.add(data.get(ij, 0), R.power(t,k))
            out = Matrix(R, d**m, d**n, data)
            if colour == 'X':
                # Direct symmetric tensor contraction avoids dense H^⊗n intermediates.
                if d**(n+m) > 2_000_000:
                    raise ValueError('spider exceeds the explicit tensor size limit')
                data = {}
                for j,t in enumerate(self.phases):
                    column = [(i,self.H.data.get((i,j),0)) for i in range(d)]
                    column = [(i,h) for i,h in column if h]
                    vector = [(0,R.power(t,k))]
                    for _ in range(n+m):
                        vector = [(d*a+i,R.mul(v,h)) for a,v in vector for i,h in column]
                    for index,v in vector:
                        ij = divmod(index,d**n)
                        data[ij] = R.add(data.get(ij,0),v)
                out = Matrix(R,d**m,d**n,data)
        self._cache[term] = out
        return out

    def as_dict(self):
        return {'ring': str(self.R), 'dimension': self.d, 'period': self.period,
                **({'ring_definition': self.R.definition} if hasattr(self.R,'definition') else {}),
                'H': [[self.R.encode(v) for v in row] for row in self.H.dense()],
                'phases': [self.R.encode(t) for t in self.phases]}

    @classmethod
    def from_dict(cls, data):
        R = Ring.parse(data['ring'])
        if 'ring_definition' in data and data['ring_definition'] != getattr(R,'definition',None):
            raise ValueError('coefficient ring definition mismatch')
        interpretation = cls(R, [[R.decode(x) for x in row] for row in data['H']],
                    [R.decode(x) for x in data['phases']], data.get('period', 0))
        if interpretation.d != data.get('dimension', interpretation.d):
            raise ValueError('certificate dimension mismatch')
        return interpretation


def check(theory, target, interpretation):
    interpretation.validate()
    finite, guaranteed = theory.prepare(target)
    checks, witness = {}, None
    for name, a, b in finite:
        left, right = interpretation.evaluate(a), interpretation.evaluate(b)
        difference = left.difference(right)
        checks[name] = not difference
        if name == target and difference:
            ij = next(iter(difference))
            witness = {'type': [a.n, a.m], 'entry': list(ij),
                       'lhs': interpretation.R.encode(left.data.get(ij,0)),
                       'rhs': interpretation.R.encode(right.data.get(ij,0))}
    valid = witness is not None and all(ok for name, ok in checks.items() if name != target)
    return {'valid': valid, 'scope': 'the supplied rules, including the registered infinite families',
            'target': target, 'theory_sha256': hashlib.sha256(theory.text.encode()).hexdigest(),
            'interpretation': interpretation.as_dict(), 'finite_checks': checks,
            'all_arity_lemmas': guaranteed,
            'structural_guarantee': 'symmetric compact Mat_R; all incident-leg permutations and bends',
            'witness': witness}


def nullspace(rows, size, p):
    """Exact RREF; returns a basis of the kernel over F_p."""
    if not isinstance(p, int):
        if isinstance(p, Ring):
            return nullspace(rows, size, p.p)
        R, A, pivots = p, [list(x) for x in rows if any(x)], []
        r = 0
        for c in range(size):
            k = next((k for k in range(r,len(A)) if A[k][c]), None)
            if k is None: continue
            A[r], A[k] = A[k], A[r]
            v = R.inv(A[r][c]); A[r] = [R.mul(x,v) for x in A[r]]
            for k in range(len(A)):
                if k != r and A[k][c]:
                    v = A[k][c]
                    A[k] = [R.add(x,R.neg(R.mul(v,y))) for x,y in zip(A[k],A[r])]
            pivots.append(c); r += 1
            if r == len(A): break
        out = []
        for free in sorted(set(range(size))-set(pivots)):
            v=[0]*size; v[free]=1
            for row,pivot in zip(A,pivots): v[pivot]=R.neg(row[free])
            out.append(v)
        return out
    A = [list(x) for x in rows if any(x)]
    pivots, r = [], 0
    for c in range(size):
        k = next((k for k in range(r, len(A)) if A[k][c] % p), None)
        if k is None:
            continue
        A[r], A[k] = A[k], A[r]
        inv = pow(A[r][c] % p, -1, p)
        A[r] = [v*inv % p for v in A[r]]
        for k in range(len(A)):
            if k != r and A[k][c] % p:
                scale = A[k][c] % p
                A[k] = [(a-scale*b) % p for a,b in zip(A[k], A[r])]
        pivots.append(c)
        r += 1
        if r == len(A):
            break
    basis = []
    for free in sorted(set(range(size)) - set(pivots)):
        v = [0]*size
        v[free] = 1
        for row, pivot in zip(A, pivots):
            v[pivot] = -row[free] % p
        basis.append(v)
    return basis


def tangent_lift(theory, target, base, checkpoint=lambda: None):
    """All first-order directions at this base: no guess for H or phase lifts."""
    base.validate()
    p, d = base.R.order, base.d
    if not base.R.is_field:
        raise ValueError('lifting needs a base interpretation over a finite field')
    finite, _ = theory.prepare(target)
    if any(base.evaluate(a).difference(base.evaluate(b)) for name,a,b in finite if name != target):
        return None
    pairs = [(i,j) for i in range(d) for j in range(i,d)]
    from rings import dual
    count, R = len(pairs)+d, dual(base.R)
    H0 = [list(row) for row in base.H.dense()]

    def lifted(vector, validate=False):
        H, phases = [row[:] for row in H0], list(base.phases)
        for coefficient, (i,j) in zip(vector, pairs):
            H[i][j] += p*coefficient
            if i != j:
                H[j][i] += p*coefficient
        phases = [a+p*b for a,b in zip(phases, vector[len(pairs):])]
        return Interpretation(R, H, phases, base.period, validate=validate)

    def residual(interpretation):
        values = {('involution', *ij): x for ij,x in
                  (interpretation.H @ interpretation.H).difference(Matrix.eye(R,d)).items()}
        if base.period:
            for j,t in enumerate(interpretation.phases):
                values['period',j] = R.add(R.power(t,base.period),R.neg(1))
        for name,a,b in finite:
            if name != target:
                for ij,v in interpretation.evaluate(a).difference(interpretation.evaluate(b)).items():
                    values[name,*ij] = v
        if any(v % p for v in values.values()):
            raise ValueError('base does not satisfy the constant constraints')
        return {key: v//p for key,v in values.items() if v//p}

    columns = []
    for k in range(count):
        checkpoint()
        columns.append(residual(lifted([int(i==k) for i in range(count)])))
    keys = set().union(*(x.keys() for x in columns))
    rows = [[column.get(key,0) for column in columns] for key in keys]
    kernel = nullspace(rows,count,base.R)
    for vector in [[0]*count, *kernel]:
        checkpoint()
        interpretation = lifted(vector, validate=True)
        certificate = check(theory,target,interpretation)
        if certificate['valid']:
            certificate['search'] = {'method': 'exact first-order kernel', 'base': base.as_dict(),
                                     'variables': count, 'constraint_rows': len(rows),
                                     'kernel_dimension': len(kernel)}
            return certificate
    return None


def seeds(R, d, exhaustive=False, checkpoint=lambda: None):
    """Small exact templates, not an exhaustive enumeration of all H."""
    if not R.p:
        import sympy as sp
        roots = [1/sp.sqrt(d), -1/sp.sqrt(d)]
    else:
        roots = [x for x in range(1,R.order) if R.mul(R.scalar(d),R.mul(x,x)) == 1]
    seen = set()
    if d & (d-1) == 0:
        for c in roots:
            H = tuple(tuple(R.mul(c, R.scalar((-1)**((i&j).bit_count()))) for j in range(d)) for i in range(d))
            if H not in seen:
                seen.add(H)
                yield H
    # Preserve the copy vector by changing basis only on its orthogonal complement.
    # This is still a search template, not an assertion that all interpretations occur here.
    if d in (4,8) and R.p:
        for c in roots:
            W = Matrix(R,d,d,{(i,j): R.mul(c,R.scalar((-1)**((i&j).bit_count())))
                             for i in range(d) for j in range(d)})
            for a in range(R.order):
                checkpoint()
                v = [0]*d
                i,j,k = (1,2,3) if d==4 else (1,2,4)
                v[i],v[j],v[k] = 1,a,R.neg(R.add(1,a))
                norm = 0
                for x in v: norm = R.add(norm,R.mul(x,x))
                try: scale = R.mul(R.scalar(2),R.inv(norm))
                except ZeroDivisionError: continue
                U = Matrix(R,d,d,{(i,j):R.add(R.scalar(int(i==j)),
                    R.neg(R.mul(scale,R.mul(v[i],v[j])))) for i in range(d) for j in range(d)})
                H = (U @ W @ U).dense()
                if H not in seen:
                    seen.add(H); yield H
    # A single reflection sending the all-ones vector to sqrt(d) e_0.
    # Unlike Walsh matrices, this template also works in non-power-of-two dimensions.
    for c in roots:
        delta = R.inv(c)
        v = [R.add(1,R.neg(delta))]+[R.scalar(1)]*(d-1)
        norm = 0
        for x in v: norm=R.add(norm,R.mul(x,x))
        try: scale=R.mul(R.scalar(2),R.inv(norm))
        except ZeroDivisionError: continue
        H = tuple(tuple(R.add(R.scalar(int(i==j)),R.neg(R.mul(scale,R.mul(v[i],v[j]))))
                        for j in range(d)) for i in range(d))
        if H not in seen:
            seen.add(H); yield H
    for diagonal in product([R.scalar(1),R.scalar(-1)], repeat=d):
        H = tuple(tuple(diagonal[i] if i==j else 0 for j in range(d)) for i in range(d))
        if H not in seen:
            seen.add(H)
            yield H
    # Exact Householder reflections provide non-Walsh, non-diagonal bases.
    for vector in product([R.scalar(-1),R.scalar(0),R.scalar(1)], repeat=d):
        checkpoint()
        norm = R.scalar(0)
        for v in vector:
            norm = R.add(norm,R.mul(v,v))
        if norm == 0:
            continue
        try: inv = R.inv(norm)
        except ZeroDivisionError: continue
        H = tuple(tuple(R.add(R.scalar(int(i==j)),R.neg(R.mul(R.scalar(2),R.mul(R.mul(vector[i],vector[j]),inv))))
                        for j in range(d)) for i in range(d))
        if H not in seen:
            seen.add(H)
            yield H

    if exhaustive:
        if not R.p:
            raise ValueError('Exhaustive base enumeration is only available over finite fields.')
        pairs = [(i,j) for i in range(d) for j in range(i,d)]
        for values in product(range(R.order),repeat=len(pairs)):
            checkpoint()
            H = [[0]*d for _ in range(d)]
            for (i,j),a in zip(pairs,values):
                H[i][j] = H[j][i] = a
            H = tuple(map(tuple,H))
            if H in seen:
                continue
            matrix = Matrix(R,d,d,{(i,j):H[i][j] for i in range(d) for j in range(d)})
            if not (matrix @ matrix).difference(Matrix.eye(R,d)):
                yield H


def phase_tuples(R, pool, d, full=True):
    # Try constant and tensor-power characters across H first, then all tuples.
    seen = set()
    for a in pool:
        values = (a,)*d
        if values not in seen:
            seen.add(values); yield values
        if d & (d-1) == 0:
            values = tuple(R.power(a,j.bit_count()) for j in range(d))
            if values not in seen:
                seen.add(values); yield values
    if full:
        for values in product(pool,repeat=d):
            if values not in seen: yield values


def search(theory, target, R, d, period=0, seconds=60, max_bases=10000, exhaustive=False):
    finite, _ = theory.prepare(target)
    if d < 1 or seconds <= 0 or max_bases < 1:
        raise ValueError('dimension, seconds, and max-bases must be positive')
    if exhaustive and not R.p:
        raise ValueError('C has no exhaustive finite enumeration.')
    start, count = time.monotonic(), 0
    def checkpoint():
        if time.monotonic()-start >= seconds:
            raise TimeoutError
    base_ring = (getattr(R, "base", None) or Ring(R.p)) if R.dual else R
    if R.p:
        phase_pool = []
        for x in range(1,base_ring.order):
            try: base_ring.inv(x)
            except ZeroDivisionError: continue
            if not period or base_ring.power(x,period)==1: phase_pool.append(x)
    else:
        import sympy as sp
        phase_pool = [sp.Integer(1),sp.Integer(-1),sp.I,-sp.I]
        phase_pool = [x for x in phase_pool if not period or base_ring.power(x,period)==1]
    if all(t.args[-1]==0 for _,a,b in finite for side in (a,b)
           for t in leaves(side) if t.kind=='gen' and t.args):
        phase_pool = [base_ring.scalar(1)]
    try:
        passes = (True,) if len(phase_pool)==1 else (False,True)
        for full in passes:
            for H in seeds(base_ring,d,exhaustive and full,checkpoint):
                checkpoint()
                # Reject phase-free constraints once per H rather than for every phase tuple.
                neutral = Interpretation(base_ring,H,[base_ring.scalar(1)]*d,period)
                phasefree = [(a,b) for name,a,b in finite if name != target and
                             all(t.args[-1]==0 for side in (a,b) for t in leaves(side) if t.args and t.kind=='gen')]
                if any(neutral.evaluate(a).difference(neutral.evaluate(b)) for a,b in phasefree):
                    continue
                for phases in phase_tuples(base_ring,phase_pool,d,full):
                    checkpoint()
                    count += 1
                    if count > max_bases:
                        return {'valid': False,'status':'base budget exhausted','bases_tried':count-1}
                    base = Interpretation(base_ring,H,phases,period,validate=False)  # seeds already prove H conditions
                    if any(base.evaluate(a).difference(base.evaluate(b)) for name,a,b in finite if name != target):
                        continue
                    certificate = tangent_lift(theory,target,base,checkpoint) if R.dual else check(theory,target,base)
                    if certificate and certificate['valid']:
                        certificate.setdefault('search',{})['bases_tried'] = count
                        certificate['search']['elapsed_seconds'] = round(time.monotonic()-start,3)
                        certificate['search']['H_seed_scope'] = 'Walsh, copy-preserving conjugates, signed diagonal, and small Householder templates; not exhaustive'
                        return certificate
    except TimeoutError:
        return {'valid':False,'status':'timeout','bases_tried':count}
    return {'valid':False,'status':('finite ansatz exhausted' if exhaustive else 'no witness in the searched templates'),
            'bases_tried':count,'exhaustive_within_fixed_ansatz':bool(exhaustive)}


def leaves(term):
    yield term
    for part in term.parts:
        yield from leaves(part)


def export_colin(theory, d, arity=1, phases=(-1,0,1)):
    """A finite JSON presentation in Colin's syntax; family truncation is explicit."""
    generators, equations = {}, []
    for rule in theory.rules:
        assignments = [({},rule.name)]
        if rule.variables:
            sections = rule.variables.split(';')
            naturals = [x.strip() for x in sections[0].split(',') if x.strip()]
            phasevars = [x.strip() for x in sections[1].split(',') if x.strip()] if len(sections)>1 else []
            names = naturals+phasevars
            choices = [range(arity+1)]*len(naturals)+[phases]*len(phasevars)
            assignments = [(dict(zip(names,vs)),f'{rule.name}[{",".join(map(str,vs))}]') for vs in product(*choices)]
        for env,name in assignments:
            a,b = rule.terms(theory.macros,env)
            for side in (a,b):
                for t in leaves(side):
                    if t.kind == 'gen':
                        generators[t.name] = t.n if t.n==t.m else [t.n,t.m]
            equations.append({'id':name,'lhs':a.colin(),'rhs':b.colin()})
    return {'name':'ZX Mat_R finite instances','wire_dimension':d,'generators':generators,'equations':equations,
            'source':{'family_scope':'FINITE INSTANCES ONLY; see the exact all-arity certificate separately',
                      'natural_bound':arity,'phase_samples':list(phases),
                      'theory_sha256':hashlib.sha256(theory.text.encode()).hexdigest()}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('rules',type=Path)
    p.add_argument('--ring',default='F5eps')
    p.add_argument('--dim',type=int,default=4)
    p.add_argument('--target',default='B')
    p.add_argument('--period',type=int,default=0,help='0: integer phases; N>0: phases modulo N')
    p.add_argument('--seconds',type=float,default=60)
    p.add_argument('--max-bases',type=int,default=10000)
    p.add_argument('--exhaustive',action='store_true',help='also enumerate all symmetric base matrices over a finite ring')
    p.add_argument('--out',type=Path,default=Path('certificate.json'))
    p.add_argument('--check',type=Path,help='independently re-evaluate an existing certificate')
    p.add_argument('--colin',type=Path,help='also export finite rule instances in Colin\'s JSON format')
    args = p.parse_args()
    try:
        theory = Theory.read(args.rules.read_text())
        if args.check:
            saved = json.loads(args.check.read_text())
            if saved.get('theory_sha256') != hashlib.sha256(theory.text.encode()).hexdigest():
                raise ValueError('certificate does not match this exact rules file')
            result = check(theory,saved['target'],Interpretation.from_dict(saved['interpretation']))
        else:
            result = search(theory,args.target,Ring.parse(args.ring),args.dim,args.period,args.seconds,args.max_bases,args.exhaustive)
        if args.colin:
            args.colin.write_text(json.dumps(export_colin(theory,result['interpretation']['dimension'] if result['valid'] else args.dim),indent=2,ensure_ascii=False)+'\n')
        args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        if result['valid']:
            print(f"FOUND: {result['target']} fails; every supplied non-target rule holds.")
            print(f"R = {result['interpretation']['ring']}, d = {result['interpretation']['dimension']}")
            period = result['interpretation']['period']
            print('Phase group:', 'Z' if not period else ('trivial (phase-free)' if period == 1 else f'Z/{period}Z'))
            R = Ring.parse(result['interpretation']['ring'])
            pretty = lambda value: R.display(R.decode(value))
            print('H =')
            for row in result['interpretation']['H']:
                print('  [' + ', '.join(pretty(x).rjust(5) for x in row) + ']')
            print('phases = (' + ', '.join(map(pretty,result['interpretation']['phases'])) + ')')
            witness = result['witness']
            print(f"{result['target']} : {witness['type'][0]} → {witness['type'][1]}; "
                  f"entry (row {witness['entry'][0]}, column {witness['entry'][1]}): "
                  f"{pretty(witness['lhs'])} ≠ {pretty(witness['rhs'])}")
            print('certificate:',args.out)
        else:
            print('UNRESOLVED:',result.get('status','candidate failed validation'))
        return 0 if result['valid'] else 1
    except (ValueError,OSError,KeyError,ZeroDivisionError,SyntaxError) as exc:
        p.exit(2,f'error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
