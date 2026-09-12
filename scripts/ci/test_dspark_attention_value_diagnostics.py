"""Host fixture isolation and poison preservation for value diagnostics."""

import unittest

import torch

from dspark_attention_value_diagnostics import KINDS, diagnostic_fixture


class ValueDiagnosticsTests(unittest.TestCase):
    def test_value_only_changes_and_poison_preservation(self):
        fixture = dict(history_value=torch.full((2, 4, 8, 128), 8192.),
            query_value=torch.full((2, 4, 4, 128), -8192.), key=torch.ones(1))
        original = {name: value.clone() for name, value in fixture.items()}
        for kind in KINDS:
            values = diagnostic_fixture(fixture, kind, 6, 3)
            self.assertIs(values['key'], fixture['key'])
            self.assertTrue(torch.equal(values['history_value'][:, :, 6:], fixture['history_value'][:, :, 6:]))
            self.assertTrue(torch.equal(values['query_value'][:, :, 3:], fixture['query_value'][:, :, 3:]))
            history = values['history_value'][0, 0, :6, 0].tolist()
            proposal = values['query_value'][0, 0, :3, 0].tolist()
            self.assertEqual(history, [1.] * 6 if kind == 'constant' else [1., 0., 0., 0., 0., 0.] if kind == 'oldest' else [0.] * 6)
            self.assertEqual(proposal, [1.] * 3 if kind == 'constant' else [0., 0., 1.] if kind == 'last_proposal' else [0.] * 3)
        for name in fixture:
            self.assertTrue(torch.equal(fixture[name], original[name]))

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            diagnostic_fixture({}, 'unknown', 6, 3)


if __name__ == '__main__':
    unittest.main()
