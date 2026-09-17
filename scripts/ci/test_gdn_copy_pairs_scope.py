"""Width isolation and restoration for the complete-request candidate."""

import unittest
from unittest.mock import Mock, patch

import gdn_copy_pairs_scope as scope
from gdn_copy_pairs_variants import POLICIES, validate_route
from test_gdn_copy_pairs import CopyPairsTests


class ScopeTests(unittest.TestCase):
    def test_width_isolation_and_restoration(self):
        kernels = dict(recurrence=dict(compute=CopyPairsTests().source()), norm_gate=dict(compute='norm'))
        original = Mock(return_value='program')
        original._copy_pairs_override = False
        with patch.object(scope.split, 'build_program', original):
            with scope.scoped_copy_pairs({}) as evidence:
                for stage, rows in (('recurrence', 1), ('recurrence', 4), ('norm_gate', 16), ('recurrence', 16)):
                    scope.split.build_program(None, None, None, kernels, stage, rows)
                for call in original.call_args_list[:3]:
                    self.assertIs(call.args[3], kernels)
                candidate = original.call_args.args[3]
                self.assertEqual(candidate['norm_gate'], kernels['norm_gate'])
                self.assertNotEqual(candidate['recurrence']['compute'], kernels['recurrence']['compute'])
            self.assertIs(scope.split.build_program, original)
        self.assertTrue(evidence['restored'])
        self.assertEqual(len(evidence['loads']), 1)

    def test_restores_on_exception_and_rejects_nesting(self):
        original = scope.split.build_program
        with self.assertRaisesRegex(RuntimeError, 'abort'):
            with scope.scoped_copy_pairs({}) as evidence:
                with self.assertRaises(ValueError):
                    with scope.scoped_copy_pairs({}):
                        pass
                raise RuntimeError('abort')
        self.assertIs(scope.split.build_program, original)
        self.assertTrue(evidence['restored'])

    def test_matched_policies_and_missing_evidence(self):
        self.assertEqual(POLICIES['publication'], dict(POLICIES['control'], gdn_copy_pairs=True))
        with patch('gdn_copy_pairs_variants.publication_route'):
            with self.assertRaises(ValueError):
                validate_route({}, 'publication')
            validate_route({}, 'control')


if __name__ == '__main__':
    unittest.main()
