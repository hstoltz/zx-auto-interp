"""Exact ring tests and an independent evaluator for the most useful interpretations."""
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zx import Ring, Interpretation, Theory, check, search, parse, Matrix

HERE=Path(__file__).parent


class ExtraRings(unittest.TestCase):
    def test_f9_is_not_z9(self):
        F,Z=Ring.parse('F9'),Ring.parse('Z9')
        self.assertEqual(F.scalar(3),0)
        self.assertEqual(Z.scalar(3),3)
        self.assertEqual(F.mul(3,3),2)  # a²=-1 in F9
        self.assertEqual(Z.mul(3,3),0)
        self.assertTrue(F.is_field)
        self.assertFalse(Z.is_field)

    def test_ring_axioms(self):
        for name in ['F4','F8','F9','Z9','F3eps3','F3split']:
            R=Ring.parse(name)
            for a,b,c in product(range(R.order),repeat=3):
                self.assertEqual(R.add(R.add(a,b),c),R.add(a,R.add(b,c)))
                self.assertEqual(R.mul(R.mul(a,b),c),R.mul(a,R.mul(b,c)))
                self.assertEqual(R.mul(a,R.add(b,c)),R.add(R.mul(a,b),R.mul(a,c)))
            for a in range(R.order):
                self.assertEqual(R.decode(R.encode(a)),a)
                self.assertEqual(R.add(a,R.neg(a)),0)
                self.assertEqual(R.mul(a,1),a)
                try: b=R.inv(a)
                except ZeroDivisionError:
                    self.assertFalse(any(R.mul(a,b)==1 for b in range(R.order)))
                else: self.assertEqual(R.mul(a,b),1)

    def test_fields_and_extensions(self):
        for q in [4,8,9,16,25,27,49,81]:
            R=Ring.parse(f'F{q}')
            for a in range(q):
                self.assertEqual(R.power(a,q),a)
                if a: self.assertEqual(R.mul(a,R.inv(a)),1)
        for q in [6,10,12,15]:
            with self.assertRaises(ValueError): Ring.parse(f'F{q}')

    def test_higher_nilpotents_and_units(self):
        R=Ring.parse('F3eps3'); e=3
        self.assertNotEqual(R.mul(e,e),0)
        self.assertEqual(R.power(e,3),0)
        self.assertEqual(R.mul(R.add(1,e),R.inv(R.add(1,e))),1)
        R=Ring.parse('F9eps'); e=9
        self.assertEqual(R.mul(e,e),0)
        self.assertEqual(R.mul(3,3),2)
        self.assertEqual(R.mul(R.add(3,e),R.inv(R.add(3,e))),1)

    def test_cli_certificate_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            for ring in ['F9','F5eps','F9eps','Z9','F3eps3']:
                f=Path(d)/'witness.json'; g=Path(d)/'checked.json'
                cmd=[sys.executable,str(HERE/'zx.py'),str(HERE/'phase_free.zx')]
                subprocess.run(cmd+['--ring',ring,'--dim','4','--period','1','--out',str(f)],
                               capture_output=True,check=True,timeout=15)
                subprocess.run(cmd+['--check',str(f),'--out',str(g)],
                               capture_output=True,check=True,timeout=15)
                self.assertTrue(json.loads(g.read_text())['valid'])

    def test_direct_spider_agrees_with_conjugation(self):
        R=Ring.parse('F9'); H=[[1,1,1,1],[1,2,1,2],[1,1,2,2],[1,2,2,1]]
        M=Interpretation(R,H,[1,3,3,2],4)
        for n,m,k in [(0,0,0),(0,3,1),(2,1,-1),(1,2,2),(2,0,3)]:
            lhs=M.evaluate(parse(f'X({n},{m},{k})'))
            rhs=M.evaluate(parse(f'H({m})')) @ M.evaluate(parse(f'Z({n},{m},{k})')) @ M.evaluate(parse(f'H({n})'))
            self.assertFalse(lhs.difference(rhs))


def independent_reference(saved):
    """No production ring or matrix operations; direct index-sum equations."""
    interpretation=saved['interpretation']; d=interpretation['dimension']; label=interpretation['ring']
    if label=='F9':
        decode=lambda x:x[0]+3*x[1]
        add=lambda x,y:(x%3+y%3)%3+3*((x//3+y//3)%3)
        mul=lambda x,y:((x%3)*(y%3)-(x//3)*(y//3))%3+3*(((x%3)*(y//3)+(x//3)*(y%3))%3)
        scalar=lambda x:x%3
        values=range(9)
    elif label in ['F5','F17','F73','F89','Z9','Z289']:
        modulus=int(label[1:])
        decode=lambda x:x
        add=lambda x,y:(x+y)%modulus
        mul=lambda x,y:x*y%modulus
        scalar=lambda x:x%modulus
        values=range(modulus)
    else: raise ValueError(label)
    def total(xs):
        out=0
        for x in xs: out=add(out,x)
        return out
    H=[[decode(x) for x in row] for row in interpretation['H']]
    t=[decode(x) for x in interpretation['phases']]
    inv=[next(b for b in values if mul(a,b)==1) for a in t]
    assert H==[list(row) for row in zip(*H)]
    assert all(total(mul(H[i][k],H[k][j]) for k in range(d))==int(i==j) for i,j in product(range(d),repeat=2))
    if interpretation['period']==4: assert all(mul(mul(x,x),mul(x,x))==1 for x in t)
    rho=[total(row) for row in H]; delta=total(rho)
    theta=total(mul(mul(x,x),x) for row in H for x in row)
    assert all(mul(delta,rho[i])*(i==j)==mul(rho[i],rho[j]) for i,j in product(range(d),repeat=2))
    if interpretation['period']==1:
        assert mul(delta,theta)==1 and mul(delta,delta)==scalar(d)
    else:
        assert mul(scalar(d),mul(theta,theta))==1
        z=total(mul(x,x) for x in t)
        assert all(z==mul(z,rho[i]) for i in range(d))
        v=[total(mul(H[b][a],inv[b]) for b in range(d)) for a in range(d)]
        assert all(mul(mul(t[i],t[j]),total(mul(mul(H[i][a],H[j][a]),v[a]) for a in range(d)))==H[i][j]
                   for i,j in product(range(d),repeat=2))
    X=[[[total(mul(mul(H[k][a],H[i][a]),H[j][a]) for a in range(d)) for j in range(d)]
        for i in range(d)] for k in range(d)]
    disagreements=[]
    for row,col in product(range(d*d),repeat=2):
        k,l=divmod(row,d);i,j=divmod(col,d)
        lhs=mul(scalar(d),mul(X[k][i][j],X[l][i][j]))
        rhs=mul(delta,X[k][i][j]) if k==l else 0
        if lhs!=rhs: disagreements.append((row,col,lhs,rhs))
    assert disagreements
    w=saved['witness']; row,col,lhs,rhs=disagreements[0]
    assert [row,col]==w['entry'] and lhs==decode(w['lhs']) and rhs==decode(w['rhs'])


class InterpretationSurvey(unittest.TestCase):
    def test_saved_interpretations_recheck(self):
        from explore import recheck
        rows=json.loads((HERE/'interpretations.json').read_text())
        self.assertGreaterEqual(recheck(rows),20)

    def test_independent_reference(self):
        rows=json.loads((HERE/'interpretations.json').read_text())
        for ring,d,period in [('F9',4,1),('F9',8,4),('F5',6,1),('Z9',4,1),('F17',8,4),('F73',8,4),('F89',8,4),('Z289',8,4)]:
            row=next(x for x in rows if (x['ring'],x['dimension'],x['period'])==(ring,d,period))
            self.assertTrue(row['result']['valid'])
            independent_reference(row['result'])


if __name__=='__main__': unittest.main()
