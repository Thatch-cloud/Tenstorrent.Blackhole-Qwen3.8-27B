"""lever_n_m3native_gate.retired_binder_leaks: only retired binders count."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lever_n_m3native_gate import RETIRED_LABELS, retired_binder_leaks  # noqa: E402

V4_ROUND = {'decode norm': 129, 'full-attention forward': 16, 'MLP forward': 0, 'GDN output projection': 0}


class RetiredBinderLeakTests(unittest.TestCase):
    def test_the_v4_payload_is_not_a_leak(self):
        self.assertEqual(retired_binder_leaks([V4_ROUND, V4_ROUND]), [])

    def test_a_retired_mlp_call_is_a_leak(self):
        leaked = dict(V4_ROUND, **{'MLP forward': 3})
        self.assertEqual(retired_binder_leaks([V4_ROUND, leaked]), [{'MLP forward': 3}])

    def test_native_attn_guards_are_retired_labels(self):
        self.assertIn('sliced attn_decode_prep', RETIRED_LABELS)
        self.assertIn('two-tile head concat', RETIRED_LABELS)
        leaked = dict(V4_ROUND, **{'two-tile head concat': 16})
        self.assertEqual(retired_binder_leaks([leaked]), [{'two-tile head concat': 16}])

    def test_unknown_labels_are_ignored(self):
        self.assertEqual(retired_binder_leaks([{'something else': 5}]), [])


if __name__ == '__main__':
    unittest.main()
