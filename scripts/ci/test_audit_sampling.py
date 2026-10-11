"""The sampled host audits (audit_sampling, QWEN_FAST_AUDIT_FIRST / QWEN_FAST_AUDIT_EVERY) and the places the verify block reads it.

The defect it fixes (tp4/octo-3): the octo-2 audited glue8 attach read two GDN layers' audit entries off four chips after EVERY round and all 48 layers on round one; at eight users that is
about a thousand small to_torch calls a round, the boot warmed for 412 s and decoded at 3.4 tok/s a seat until its users timed out. The audits now sample: the first N rounds and every
K-th after; unset, every round is audited exactly as before (nothing here is read)."""

import os
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import audit_sampling
import verify_trace_t2

HERE = Path(__file__).resolve().parent


def env(**values):
    return patch.dict(os.environ, values, clear=False)


class SettingsTests(unittest.TestCase):
    def test_unset_is_off_and_every_round_is_audited_in_its_own_number(self):
        with patch.dict(os.environ, {}, clear=False):
            for name in audit_sampling.NAMES:
                os.environ.pop(name, None)
            self.assertIsNone(audit_sampling.settings())
            self.assertFalse(audit_sampling.enabled())
            self.assertEqual([audit_sampling.sampled(r) for r in range(1, 9)], [True] * 8)
            self.assertEqual([audit_sampling.ordinal(r) for r in range(1, 9)], list(range(1, 9)))
            self.assertEqual(audit_sampling.label(), '')
            self.assertEqual(audit_sampling.problems(), [])

    def test_empty_values_are_unset(self):
        self.assertIsNone(audit_sampling.settings({'QWEN_FAST_AUDIT_FIRST': '', 'QWEN_FAST_AUDIT_EVERY': ''}))

    def test_every_alone_means_the_first_round_then_every_kth(self):
        self.assertEqual(audit_sampling.settings({'QWEN_FAST_AUDIT_EVERY': '8'}), (1, 8))
        self.assertEqual(audit_sampling.audited_rounds(30, {'QWEN_FAST_AUDIT_EVERY': '8'}), [1, 9, 17, 25])

    def test_first_and_every(self):
        environ = {'QWEN_FAST_AUDIT_FIRST': '3', 'QWEN_FAST_AUDIT_EVERY': '5'}
        self.assertEqual(audit_sampling.audited_rounds(20, environ), [1, 2, 3, 8, 13, 18])
        self.assertEqual([audit_sampling.ordinal(r, environ) for r in audit_sampling.audited_rounds(20, environ)], [1, 2, 3, 4, 5, 6])
        self.assertEqual(audit_sampling.label(environ), 'first:3,every:5')

    def test_the_ordinal_keeps_the_layer_rotation_whole(self):
        """The k-th audited round audits what the k-th round did: round one every layer, then two a round in rotation; a sampled run still covers all 48 layers."""
        environ = {'QWEN_FAST_AUDIT_FIRST': '1', 'QWEN_FAST_AUDIT_EVERY': '4'}
        covered = set()
        for round_number in audit_sampling.audited_rounds(4 * 30, environ):
            covered.update(verify_trace_t2.audit_layers(audit_sampling.ordinal(round_number, environ), 48))
        self.assertEqual(covered, set(range(48)))
        # (round one alone already names every layer; the rotation after it is what a long run keeps covering)
        later = set()
        for round_number in audit_sampling.audited_rounds(4 * 30, environ)[1:]:
            later.update(verify_trace_t2.audit_layers(audit_sampling.ordinal(round_number, environ), 48))
        self.assertEqual(later, set(range(48)))

    def test_a_sampled_run_audits_a_small_share_of_rounds(self):
        environ = {'QWEN_FAST_AUDIT_FIRST': '2', 'QWEN_FAST_AUDIT_EVERY': '16'}
        rounds = audit_sampling.audited_rounds(1000, environ)
        self.assertEqual(len(rounds), 2 + 62)
        self.assertEqual(rounds[:3], [1, 2, 18])

    def test_malformed_values_name_the_flag(self):
        for bad in ('0', '-1', '1.5', 'x', ' 3', '03', '1e3'):
            for name in audit_sampling.NAMES:
                other = 'QWEN_FAST_AUDIT_EVERY' if name == 'QWEN_FAST_AUDIT_FIRST' else 'QWEN_FAST_AUDIT_FIRST'
                with self.subTest(name=name, value=bad):
                    with self.assertRaisesRegex(ValueError, name):
                        audit_sampling.settings({name: bad, other: '4'})
                    self.assertEqual(len(audit_sampling.problems({name: bad, other: '4'})), 1)

    def test_first_without_every_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_AUDIT_FIRST=5 without QWEN_FAST_AUDIT_EVERY'):
            audit_sampling.settings({'QWEN_FAST_AUDIT_FIRST': '5'})

    def test_rounds_are_one_based(self):
        with self.assertRaises(ValueError):
            audit_sampling.sampled(0, {'QWEN_FAST_AUDIT_EVERY': '2'})


class VerifierWiringTests(unittest.TestCase):
    """packed_verifier reads the module for the four host read-back audits and for nothing else; the module is in the image."""

    def source(self):
        return (HERE / 'packed_verifier.py').read_text(encoding='utf-8')

    def test_the_four_read_back_audits_are_gated_by_the_sample_and_use_its_ordinal(self):
        text = self.source()
        self.assertIn('audit_round_number = audit_sampling.ordinal(self.rounds + 1) if audit_sampling.sampled(self.rounds + 1) else None', text)
        for call in ('verify_trace_t2.audit_round(self.operations, self.fixture.retained.records, audit_round_number)',
                     'tp4_vglue.audit_round(self.operations, self.fixture.retained.records, audit_round_number,',
                     'gdn_conv_gates_spread.audit_round(self.operations, self.fixture.retained.records, audit_round_number)'):
            self.assertEqual(text.count(call), 1, call)
        self.assertIn('self.seq_block_audit,\n                                          audit_round_number)', text)
        # the counter audits and the extent audit are NOT sampled: they cost a few words, or guard a round's own state
        self.assertIn('kv_page_writer_tp4.audit_round(self.operations, self.rounds + 1)', text)
        self.assertIn('mlp.audit_round(self.operations, self.fixture, self.rounds + 1)', text)
        self.assertIn('tile_collective_tp.audit_round(self.operations, self.fixture, self.rounds + 1)', text)
        self.assertIn('self.audit_extent(extent_users, self.rounds + 1)', text)

    def test_a_malformed_flag_fails_the_block_construction_by_name(self):
        text = self.source()
        self.assertRegex(text, r'audit_sampling\.settings\(\)\n\s+self\.prestaged = ')

    def test_the_module_ships_in_the_image(self):
        overlay = (HERE.parent.parent / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8')
        self.assertIn('scripts/ci/audit_sampling.py', overlay)

    def test_the_vglue_audit_line_names_the_sample_only_while_sampling(self):
        import tp4_vglue

        lines = []
        records = [(None, {'vglue_audit': [dict(label='piece user 0', mine=1, served=2)]}, None)] * 48
        with patch.object(tp4_vglue, 'log_line', side_effect=lines.append), patch.object(tp4_vglue, 'compare_entry', return_value=[]), \
                patch.object(tp4_vglue, 'missing_entries', return_value=[]), patch.object(tp4_vglue, 'enabled', return_value=True):
            with patch.dict(os.environ, {}, clear=False):
                for name in audit_sampling.NAMES:
                    os.environ.pop(name, None)
                tp4_vglue.audit_round(object(), records, 1)
            with env(QWEN_FAST_AUDIT_FIRST='1', QWEN_FAST_AUDIT_EVERY='8'):
                tp4_vglue.audit_round(object(), records, 2)
        self.assertEqual(len(lines), 2)
        self.assertFalse(re.search(r' sample=', lines[0]), lines[0])
        self.assertTrue(lines[1].endswith(' sample=first:1,every:8'), lines[1])


if __name__ == '__main__':
    unittest.main()
