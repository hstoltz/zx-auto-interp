"""Small exact coefficient rings. Integers encode polynomial coefficient lists.

F9 is a FIELD, not Z/9Z. FqepsN means Fq[ε]/(ε^N).
No dependencies; no floating-point arithmetic.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import isqrt
import re


@dataclass(frozen=True)
class Mod:
    n: int

    def __post_init__(self):
        if self.n < 2:
            raise ValueError('modulus must be at least two')

    @property
    def p(self): return self.n
    @property
    def order(self): return self.n
    @property
    def dual(self): return False
    @property
    def is_field(self): return False
    @property
    def definition(self): return {'kind': 'integers_modulo', 'modulus': self.n}
    def __str__(self): return f'Z{self.n}'
    def q(self, a):
        if type(a) is not int:
            raise ValueError('ring entries must be exact integer codes')
        return a % self.n
    def scalar(self, a): return a % self.n
    def add(self, a, b): return (a+b) % self.n
    def neg(self, a): return -a % self.n
    def mul(self, a, b): return a*b % self.n
    def inv(self, a):
        try: return pow(a, -1, self.n)
        except ValueError as e: raise ZeroDivisionError(f'{a} is not a unit modulo {self.n}') from e
    def power(self, a, n):
        if n < 0: return pow(self.inv(a), -n, self.n)
        return pow(a, n, self.n)
    def display(self, a): return str(a)
    def encode(self, a): return int(a)
    def decode(self, a): return self.q(a)


@dataclass(frozen=True)
class Quotient:
    base: object
    polynomial: tuple  # low degree first; monic
    label: str
    variable: str = 'a'
    is_field: bool = False

    @property
    def degree(self): return len(self.polynomial)-1
    @property
    def order(self): return self.base.order**self.degree
    @property
    def p(self): return self.base.p
    @property
    def dual(self): return self.polynomial == (0, 0, 1) and self.base.is_field
    @property
    def definition(self):
        return {'kind': 'polynomial_quotient', 'base': str(self.base),
                'polynomial': [self.base.encode(x) for x in self.polynomial],
                'variable': self.variable, 'is_field': self.is_field}
    def __str__(self): return self.label
    def q(self, a):
        if type(a) is not int: raise ValueError('ring entries must be exact integer codes')
        return a % self.order
    def scalar(self, a): return self.base.scalar(a)
    def coefficients(self, a):
        q = self.base.order
        return tuple((a//q**j) % q for j in range(self.degree))
    def pack(self, a): return sum(v*self.base.order**j for j,v in enumerate(a))
    @lru_cache(maxsize=65536)
    def add(self, a, b):
        return self.pack([self.base.add(x,y) for x,y in zip(self.coefficients(a), self.coefficients(b))])
    @lru_cache(maxsize=16384)
    def neg(self, a): return self.pack([self.base.neg(x) for x in self.coefficients(a)])
    @lru_cache(maxsize=65536)
    def mul(self, a, b):
        A,B = self.coefficients(a),self.coefficients(b)
        R,n = self.base,self.degree
        out = [0]*(2*n-1)
        for i,x in enumerate(A):
            if x:
                for j,y in enumerate(B):
                    if y: out[i+j] = R.add(out[i+j], R.mul(x,y))
        for k in range(2*n-2,n-1,-1):
            if out[k]:
                for j,c in enumerate(self.polynomial[:-1]):
                    if c: out[k-n+j] = R.add(out[k-n+j],R.neg(R.mul(out[k],c)))
        return self.pack(out[:n])
    def power(self, a, n):
        if n < 0: return self.power(self.inv(a),-n)
        out = 1
        while n:
            if n&1: out = self.mul(out,a)
            a,n = self.mul(a,a),n//2
        return out
    @lru_cache(maxsize=16384)
    def inv(self, a):
        if a == 0: raise ZeroDivisionError('zero is not a unit')
        if self.is_field: return self.power(a,self.order-2)
        if not any(self.polynomial[:-1]):
            # Invert a truncated power series, over any supported base ring.
            A = self.coefficients(a); B = [self.base.inv(A[0])]
            for k in range(1,self.degree):
                s = 0
                for j in range(1,k+1): s = self.base.add(s,self.base.mul(A[j],B[k-j]))
                B.append(self.base.neg(self.base.mul(B[0],s)))
            return self.pack(B)
        for b in range(1,self.order):
            if self.mul(a,b)==1: return b
        raise ZeroDivisionError(f'{self.display(a)} is not a unit')
    def display(self, a):
        terms = []
        for k,c in enumerate(self.coefficients(a)):
            if not c: continue
            s = self.base.display(c)
            if not k: terms.append(s); continue
            v = self.variable+(f'^{k}' if k>1 else '')
            terms.append(v if c==1 else f'({s}){v}')
        return '+'.join(terms) or '0'
    def encode(self, a): return [self.base.encode(c) for c in self.coefficients(a)]
    def decode(self, a):
        if not isinstance(a,list) or len(a)!=self.degree:
            raise ValueError(f'{self.label} entries need {self.degree} coefficients')
        return self.pack([self.base.decode(x) for x in a])


def irreducible(p,n):
    """First monic irreducible; exact trial division (only small extensions)."""
    for coefficients in product(range(p),repeat=n):
        if not coefficients[0]: continue
        f = tuple(coefficients)+(1,)
        reducible = False
        for degree in range(1,n//2+1):
            for lower in product(range(p),repeat=degree):
                g = lower+(1,); r=list(f)
                for k in range(n,degree-1,-1):
                    a=r[k]
                    for j in range(degree+1): r[k-degree+j]=(r[k-degree+j]-a*g[j])%p
                if not any(r): reducible=True; break
            if reducible: break
        if not reducible: return f
    raise ValueError('no irreducible polynomial found')


@lru_cache(maxsize=256)
def field(q):
    from zx import Ring
    if q < 2: raise ValueError('field order must be a prime power')
    p = next((i for i in range(2,isqrt(q)+1) if q%i==0),q)
    n,r=0,q
    while r%p==0: r//=p; n+=1
    if r!=1: raise ValueError('field order must be a prime power')
    if n==1: return Ring(p)
    if n>4 or q>4096: raise ValueError('small extension backend: degree <= 4, order <= 4096')
    return Quotient(Ring(p),irreducible(p,n),f'F{q}',is_field=True)


def dual(base):
    from zx import Ring
    if isinstance(base,Ring): return Ring(base.p,True)
    return Quotient(base,(0,0,1),f'{base}eps',variable='ε')


def parse(text):
    match = re.fullmatch(r'Z(\d+)',text)
    if match: return Mod(int(match[1]))
    match = re.fullmatch(r'F(\d+)(?:eps(\d*)|\[ε\]/ε²)?',text)
    if match:
        q = int(match[1]); base = field(q)
        if 'eps' not in text and '[ε]' not in text: return base
        n = int(match[2] or '2')
        if n<2 or n>8: raise ValueError('nilpotency order must be 2..8')
        if n==2: return dual(base)
        return Quotient(base,(0,)*n+(1,),f'F{q}eps{n}',variable='ε')
    match = re.fullmatch(r'F(\d+)split',text)
    if match:
        base=field(int(match[1]))
        return Quotient(base,(base.neg(1),0,1),text,variable='u')
    raise ValueError('Use C, Fq, Fqeps, Fqeps3, Zn, or Fqsplit (u²=1).')
