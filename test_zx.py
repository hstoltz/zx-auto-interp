"""Exact regression tests. Run: python -m unittest -v"""
from itertools import permutations, product
import json
from pathlib import Path
import unittest

from zx import Ring, Matrix, Interpretation, Theory, parse, check, search, tangent_lift, export_colin, nullspace, UnsupportedFamily

HERE = Path(__file__).parent


class RingTests(unittest.TestCase):
    def test_dual_ring_axioms_exhaustively(self):
        R, values = Ring(3,True), range(9)
        self.assertEqual(R.mul(3,3),0)
        for a,b,c in product(values,repeat=3):
            self.assertEqual(R.add(R.add(a,b),c),R.add(a,R.add(b,c)))
            self.assertEqual(R.mul(R.mul(a,b),c),R.mul(a,R.mul(b,c)))
            self.assertEqual(R.mul(a,R.add(b,c)),R.add(R.mul(a,b),R.mul(a,c)))
        for a in values:
            self.assertEqual(R.add(a,R.neg(a)),0)
            if a % 3:
                self.assertEqual(R.mul(a,R.inv(a)),1)
            else:
                with self.assertRaises(ZeroDivisionError):
                    R.inv(a)

    def test_no_fake_finite_fields_or_floating_point(self):
        for p in (1,4,9,15):
            with self.assertRaises(ValueError):
                Ring(p)
        with self.assertRaises(ValueError):
            Ring(5).q(1.1)
        with self.assertRaises(ValueError):
            Ring(0).q('0.5')
        self.assertEqual(Ring(0).q('1/2')*2,1)
        with self.assertRaises(ValueError):
            Ring(0).q('__import__("os")')

    def test_matrix_interchange_and_composition(self):
        R = Ring(5,True)
        A = Matrix(R,2,2,{(0,0):1,(0,1):5,(1,1):2})
        B = Matrix(R,2,2,{(0,1):3,(1,0):1})
        self.assertFalse(((A.tensor(B)) @ (B.tensor(A))).difference((A@B).tensor(B@A)))
        self.assertFalse(((A@B).T).difference(B.T@A.T))

    def test_nullspace(self):
        basis = nullspace([[1,2,3],[2,4,1]],3,5)
        self.assertEqual(len(basis),2)
        for v in basis:
            self.assertEqual(sum(a*b for a,b in zip([1,2,3],v))%5,0)


class SyntaxTests(unittest.TestCase):
    def test_zero_arities_and_unicode(self):
        for expr,n,m in [('Z(0,0)',0,0),('X(0,2,-1)',0,2),('cap ◦ cup',0,0),('H(0)',0,0)]:
            t = parse(expr)
            self.assertEqual((t.n,t.m),(n,m))
        self.assertEqual(parse('id ⊗ swap ⊗ id').n,4)
        self.assertEqual(parse('perm(2,0,1)').args,(2,0,1))
        self.assertEqual(Theory.read('B2′: H = id').rules[0].name,'B2′')

    def test_composition_reversed_only_at_colin_boundary(self):
        t = parse('Z(1,2) ∘ X(2,1)')
        self.assertEqual(t.colin(),{'compose':[{'gen':'X_2_1_0'},{'gen':'Z_1_2_0'}]})

    def test_typing_and_input_errors(self):
        for text in ['Z(-1,2)','H ∘ cup','X(1)','perm(0,0)','Z(1/2,1)','__import__(1)','(H']:
            with self.assertRaises((ValueError,SyntaxError)):
                parse(text)
        with self.assertRaises(ValueError):
            Theory.read('bad: cup = H')
        with self.assertRaises(ValueError):
            Theory.read('a := b\nb := a\nbad: a = H')

    def test_unknown_families_never_certified_by_sampling(self):
        theory = Theory.read('arbitrary[n; k]: Z(n,n,k) = id(n)\nB: H = id')
        with self.assertRaises(UnsupportedFamily):
            theory.prepare('B')

    def test_family_name_alone_is_not_a_proof(self):
        theory = Theory.read('Sg[p,q,r,s;a,b]: Z(p+r,q+s,a+b) = id(p+r)\nB: H = id')
        with self.assertRaises(UnsupportedFamily):
            theory.prepare('B')


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.theory = Theory.read((HERE/'bialgebra.zx').read_text())
        cls.result = search(cls.theory,'B',Ring(5,True),4,seconds=20)
        assert cls.result['valid'], cls.result
        cls.interpretation = Interpretation.from_dict(cls.result['interpretation'])

    def test_actual_search_not_seeded_witness(self):
        self.assertEqual(self.result['search']['variables'],14)
        self.assertEqual(self.result['search']['kernel_dimension'],1)
        self.assertFalse(self.result['finite_checks']['B'])
        self.assertTrue(all(ok for name,ok in self.result['finite_checks'].items() if name!='B'))
        self.assertEqual(len(self.result['all_arity_lemmas']),3)

    def test_recheck_and_phase_group(self):
        self.assertTrue(check(self.theory,'B',self.interpretation)['valid'])
        data = json.loads(json.dumps(self.interpretation.as_dict()))
        data['period'] = 4
        with self.assertRaises(ValueError):
            Interpretation.from_dict(data)

    def test_tampered_H_rejected(self):
        data = json.loads(json.dumps(self.interpretation.as_dict()))
        data['H'][0][1] = [0,0]
        with self.assertRaises(ValueError):
            Interpretation.from_dict(data)

    def test_symmetric_involution_does_not_mean_adjoint(self):
        import sympy as sp
        R = Ring(0)
        H = [[sp.sqrt(2),sp.I],[sp.I,-sp.sqrt(2)]]
        interpretation = Interpretation(R,H,[1,1])
        self.assertFalse((interpretation.H @ interpretation.H).difference(Matrix.eye(R,2)))
        self.assertNotEqual(H[0][1],sp.conjugate(H[1][0]))

    def test_standard_complex_model(self):
        import sympy as sp
        R = Ring(0)
        h = sp.sqrt(2)/2
        interpretation = Interpretation(R,[[h,h],[h,-h]],[1,sp.I],4)
        self.assertFalse(interpretation.evaluate(parse('X(0,2)')).difference(interpretation.evaluate(parse('cup'))))
        self.assertEqual(interpretation.evaluate(parse('Z(0,0,2)')).data,{})

    def test_snakes_and_loops(self):
        M = self.interpretation
        for expression in ['(cap ⊗ id) ∘ (id ⊗ cup)','(id ⊗ cap) ∘ (cup ⊗ id)']:
            self.assertFalse(M.evaluate(parse(expression)).difference(M.evaluate(parse('id'))))
        self.assertEqual(M.evaluate(parse('cap ∘ cup')).data,{(0,0):4})
        self.assertEqual(M.evaluate(parse('Z(0,0)')).data,{(0,0):4})

    def test_fusion_boundary_cases(self):
        for rule in self.theory.rules[:2]:
            for nat in product(range(2),repeat=4):
                for a,b in product([-1,0,1],repeat=2):
                    env = dict(zip(['p','q','r','s'],nat)) | {'a':a,'b':b}
                    left,right = rule.terms(self.theory.macros,env)
                    self.assertFalse(self.interpretation.evaluate(left).difference(self.interpretation.evaluate(right)))

    def test_colour_change_both_directions(self):
        M = self.interpretation
        for n,m,k in product(range(3),range(3),[-1,0,1,4]):
            left = M.evaluate(parse(f'H({m}) ∘ X({n},{m},{k}) ∘ H({n})'))
            right = M.evaluate(parse(f'Z({n},{m},{k})'))
            self.assertFalse(left.difference(right))

    def test_full_incident_leg_symmetry_and_bending(self):
        M = self.interpretation
        for colour,k in product(['Z','X'],[-1,0,1]):
            state = M.evaluate(parse(f'{colour}(0,3,{k})'))
            bent = M.evaluate(parse(f'{colour}(1,2,{k})'))
            for word in product(range(M.d),repeat=3):
                index = lambda w: sum(a*M.d**(len(w)-i-1) for i,a in enumerate(w))
                value = state.data.get((index(word),0),0)
                self.assertEqual(value,bent.data.get((index(word[:2]),word[2]),0))
                for perm in permutations(range(3)):
                    self.assertEqual(value,state.data.get((index(tuple(word[i] for i in perm)),0),0))

    def test_original_user_witness(self):
        R = Ring(5,True)
        raw = [[1,1,1,1],[1,24,1,9],[1,1,9,24],[1,9,24,1]]
        H = [[R.mul(3,a) for a in row] for row in raw]
        M = Interpretation(R,H,[1,13,18,4])
        result = check(self.theory,'B',M)
        self.assertTrue(result['valid'])
        _,a,b = next(x for x in self.theory.prepare('B')[0] if x[0]=='B')
        self.assertEqual(M.evaluate(a).data.get((2,5),0),5) # ε
        self.assertEqual(M.evaluate(b).data.get((2,5),0),0)

    def test_phase_free_search_and_standard_soundness(self):
        import sympy as sp
        theory = Theory.read((HERE/'phase_free.zx').read_text())
        result = search(theory,'B',Ring(5,True),4,period=1,seconds=10)
        self.assertTrue(result['valid'])
        self.assertEqual(result['interpretation']['phases'],[[1,0]]*4)
        h = sp.sqrt(2)/2
        standard = Interpretation(Ring(0),[[h,h],[h,-h]],[1,1],1)
        comparison = check(theory,'B',standard)
        self.assertFalse(comparison['valid'])
        self.assertTrue(all(comparison['finite_checks'].values()))

    def test_all_nonzero_tangent_multiples(self):
        original = self.interpretation.as_dict()
        for a in range(5):
            data = json.loads(json.dumps(original))
            data['H'] = [[[x,(a*y)%5] for x,y in row] for row in original['H']]
            data['phases'] = [[x,(a*y)%5] for x,y in original['phases']]
            result = check(self.theory,'B',Interpretation.from_dict(data))
            self.assertEqual(result['valid'],a!=0)

    def test_forced_family_not_searchable(self):
        with self.assertRaises(ValueError):
            search(self.theory,'Sg',Ring(5,True),4)

    def test_finite_exhaustion_does_not_claim_derivability(self):
        theory = Theory.read('B: H ∘ H = id')
        out = search(theory,'B',Ring(3,True),1,seconds=10,exhaustive=True)
        self.assertFalse(out['valid'])
        self.assertEqual(out['status'],'finite ansatz exhausted')

    def test_periodic_phases_when_characteristic_divides_period(self):
        theory = Theory.read('E: Z(1,1,3) = id\nB: Z(1,1,1) = id')
        out = search(theory,'B',Ring(3,True),1,period=3,seconds=5)
        self.assertTrue(out['valid'])
        self.assertTrue(check(theory,'B',Interpretation.from_dict(out['interpretation']))['valid'])

    def test_export_includes_scalar_types_and_warns_about_families(self):
        data = export_colin(self.theory,4,arity=0,phases=(0,))
        self.assertEqual(data['generators']['Z_0_0_0'],0)
        self.assertEqual(data['generators']['cup'],[0,2])
        self.assertIn('FINITE INSTANCES ONLY',data['source']['family_scope'])
        self.assertEqual(len(data['equations']),10)


if __name__ == '__main__':
    unittest.main()
