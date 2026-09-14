from contextlib import nullcontext
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import dspark_64k_entry as entry


class EntryTests(unittest.TestCase):
    def test_only_qualified_comparison_options_are_accepted(self):
        forbidden = ('target_attention_variants', 'combined_variants', 'mlp_down',
            'score_layout', 'native_attention_variants', 'profile_drafter',
            'profile_verifier', 'request_variants', 'fused_t16_mlp',
            'history_profile', 'banked_proposal', 'native_slot_gdn', 'mlp_equal_footprint')
        options = SimpleNamespace(request=True, captured_publication=True,
            norm_scatter_variants=True, max_new_tokens=256,
            **dict.fromkeys(forbidden, False))
        entry.validate_options(options)
        for name in forbidden:
            with self.subTest(option=name):
                setattr(options, name, True)
                with self.assertRaises(ValueError):
                    entry.validate_options(options)
                setattr(options, name, False)
        for name, value in (('request', False), ('captured_publication', False),
                ('norm_scatter_variants', False), ('max_new_tokens', 1024)):
            with self.subTest(option=name):
                original = getattr(options, name)
                setattr(options, name, value)
                with self.assertRaises(ValueError):
                    entry.validate_options(options)
                setattr(options, name, original)

    def test_requires_qualified_scope_for_device_execution(self):
        environment = dict(QWEN_DSPARK_64K_TRIAL='1', QWEN_HARDWARE_TESTS='1',
            QWEN_CARDS_ALLOCATED='1', QWEN_LADDER_BACKEND='hardware', TT_METAL_HOME='/native')
        arguments = ['request', '--request', '--captured-publication', '--norm-scatter-variants']
        main = Mock(return_value='result')
        with patch.dict(os.environ, environment, clear=True), patch.object(sys, 'argv', arguments), \
                patch.object(entry, 'runtime_scope', return_value=nullcontext({})) as scope:
            self.assertEqual(entry.run(main), 'result')
            self.assertEqual(scope.call_args.kwargs['context'], 65536)
            self.assertEqual(scope.call_args.kwargs['output_tokens'], 256)
            main.assert_called_once()
            arguments.append('--combined-variants')
            with self.assertRaisesRegex(ValueError, 'Unqualified'):
                entry.run(main)
            self.assertEqual(main.call_count, 1)

    def test_unallocated_entry_is_rejected_before_main(self):
        main = Mock()
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                entry.run(main)
        main.assert_not_called()
