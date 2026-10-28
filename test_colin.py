"""JSON round-trip tests; the upstream integration test runs when installed."""
import importlib.util
from itertools import product
import json
from pathlib import Path
import unittest

from colin import from_colin, candidate, exact_value, exact_equal
from zx import Theory, Ring, Interpretation, export_colin, parse, search


class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.theory = Theory.read(Path(__file__).with_name('bialgebra.zx').read_text())
        cls.cert = search(cls.theory,'B',Ring(5,True),4,seconds=20)
        cls.interpretation = Interpretation.from_dict(cls.cert['interpretation'])
        cls.export = export_colin(cls.theory,4)

    def test_all_exported_equations_roundtrip(self):
        for equation in self.export['equations']:
            a,b = (self.interpretation.evaluate(from_colin(equation[s])) for s in ('lhs','rhs'))
            self.assertEqual(bool(a.difference(b)),equation['id']=='B')

    def test_noncommuting_composition_not_accidentally_reversed(self):
        t = parse('H ∘ Z(1,1,1)')
        original = self.interpretation.evaluate(t)
        converted = self.interpretation.evaluate(from_colin(t.colin()))
        self.assertFalse(original.difference(converted))
        wrong = self.interpretation.evaluate(parse('Z(1,1,1) ∘ H'))
        self.assertTrue(original.difference(wrong))

    def test_encoded_values_keep_dual_coefficients(self):
        term = parse('H ∘ Z(1,1,1)')
        value = exact_value(self.interpretation,term.colin())
        json.dumps(value)
        self.assertTrue(all(isinstance(entry[2],list) and len(entry[2])==2 for entry in value['entries']))
        self.assertTrue(exact_equal(self.interpretation,value,value))

    @unittest.skipUnless(importlib.util.find_spec('minimality_auto'),'upstream package is not installed')
    def test_actual_upstream_parser_and_verifier(self):
        from minimality_auto.core import Theory as UpstreamTheory
        from minimality_auto.search import Deadline, verify
        upstream = UpstreamTheory.from_json(self.export)
        result = verify(candidate(self.interpretation),upstream,upstream.equation('B'),
                        Deadline.after(30),strategy='zx_exact')
        self.assertIsNotNone(result)
        self.assertTrue(result.certified)
        self.assertEqual(len(result.checked_equations),len(self.export['equations'])-1)
        json.dumps(result.as_dict())


if __name__ == '__main__':
    unittest.main()
