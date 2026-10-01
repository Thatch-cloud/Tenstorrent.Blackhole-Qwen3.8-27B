"""tp_addresses: the chip-count-generic address helpers, identical to the pinned two-chip ones at the pair, and the
startup seam that rebinds the pinned names at four cards only."""

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import gdn_multitoken_conv as pinned
import tp_addresses

PINNED_ADDRESSES, PINNED_RELEASE = pinned.addresses, pinned.release_owned


def tensor_on(*addresses):
    return SimpleNamespace(shards=[SimpleNamespace(buffer_address=lambda address=address: address)
                                   for address in addresses])


def operations():
    return SimpleNamespace(get_device_tensors=lambda value: value.shards, deallocate=Mock())


class AddressTests(unittest.TestCase):
    def test_at_the_pair_it_is_the_pinned_function_call_for_call(self):
        with patch.dict(os.environ, {}, clear=True):
            for value in (tensor_on(10, 20), tensor_on(7, 7)):
                self.assertEqual(tp_addresses.addresses(operations(), value),
                                 PINNED_ADDRESSES(operations(), value))
            for value in (tensor_on(1), tensor_on(1, 2, 3, 4)):
                with self.assertRaises(ValueError) as theirs:
                    PINNED_ADDRESSES(operations(), value)
                with self.assertRaises(ValueError) as ours:
                    tp_addresses.addresses(operations(), value)
                self.assertEqual(str(ours.exception), str(theirs.exception))

    def test_at_four_cards_it_reads_four_chips_in_order(self):
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}):
            self.assertEqual(tp_addresses.addresses(operations(), tensor_on(4, 3, 2, 1)), (4, 3, 2, 1))
            for value in (tensor_on(1, 2), tensor_on(1, 2, 3)):
                with self.assertRaisesRegex(ValueError, 'All 4 chips required'):
                    tp_addresses.addresses(operations(), value)

    def test_release_frees_each_distinct_tensor_once(self):
        first, twin, other = tensor_on(1, 2, 3, 4), tensor_on(1, 2, 3, 4), tensor_on(5, 6, 7, 8)
        ops = operations()
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}):
            tp_addresses.release_owned(ops, [first, twin, other])
        self.assertEqual(ops.deallocate.call_count, 2)
        freed = [call.args[0] for call in ops.deallocate.call_args_list]
        self.assertIn(other, freed)
        self.assertEqual(sum(value in (first, twin) for value in freed), 1)


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.holder = types.ModuleType('tp_addresses_holder')
        self.holder.addresses, self.holder.release_owned = PINNED_ADDRESSES, PINNED_RELEASE
        self.unrelated = types.ModuleType('tp_addresses_unrelated')
        self.unrelated.addresses = lambda operations, tensor: 'not the pinned one'
        sys.modules[self.holder.__name__] = self.holder
        sys.modules[self.unrelated.__name__] = self.unrelated
        self.holders = [module for module in list(sys.modules.values())
                        if getattr(module, '__dict__', None) is not None
                        and (module.__dict__.get('addresses') is PINNED_ADDRESSES
                             or module.__dict__.get('release_owned') is PINNED_RELEASE)]

        def restore():
            tp_addresses.uninstall()
            for name in (self.holder.__name__, self.unrelated.__name__):
                sys.modules.pop(name, None)

        self.addCleanup(restore)

    def test_the_pair_never_installs(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                tp_addresses.install()
        self.assertIs(pinned.addresses, PINNED_ADDRESSES)
        self.assertIs(self.holder.addresses, PINNED_ADDRESSES)

    def test_four_cards_rebind_the_pinned_module_and_every_module_holding_its_functions(self):
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}):
            rebound = tp_addresses.install()
        self.assertGreaterEqual(rebound, 4)
        self.assertIs(pinned.addresses, tp_addresses.addresses)
        self.assertIs(pinned.release_owned, tp_addresses.release_owned)
        self.assertIs(self.holder.addresses, tp_addresses.addresses)
        self.assertIs(self.holder.release_owned, tp_addresses.release_owned)
        self.assertEqual(self.unrelated.addresses(None, None), 'not the pinned one')
        for module in self.holders:
            self.assertIsNot(module.__dict__.get('addresses'), PINNED_ADDRESSES, module)

    def test_the_rebound_release_counts_four_chips(self):
        ops = operations()
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}):
            tp_addresses.install()
            pinned.release_owned(ops, [tensor_on(1, 2, 3, 4)])
        ops.deallocate.assert_called_once()

    def test_the_gdn_twins_replace_the_literal_carrying_helpers_by_identity(self):
        import gdn_commit_dma
        import gdn_records
        pairs = dict(validate_projected=pinned.validate_projected, restore_prefix=pinned.restore_prefix,
                     retain=gdn_records.retain_checkpoint_histories, publish=gdn_commit_dma.publish,
                     prepare=gdn_commit_dma.prepare, shapes=gdn_commit_dma.validate_shapes)
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}):
            tp_addresses.install()
        import gdn_commit_dma_tp
        import gdn_multitoken_conv_tp
        import gdn_records_tp
        self.assertIs(pinned.validate_projected, gdn_multitoken_conv_tp.validate_projected)
        self.assertIs(pinned.restore_prefix, gdn_multitoken_conv_tp.restore_prefix)
        self.assertIs(gdn_records.retain_checkpoint_histories, gdn_records_tp.retain_checkpoint_histories)
        self.assertIs(gdn_commit_dma.publish, gdn_commit_dma_tp.publish)
        self.assertIs(gdn_commit_dma.prepare, gdn_commit_dma_tp.prepare)
        self.assertIs(gdn_commit_dma.validate_shapes, gdn_commit_dma_tp.validate_shapes)
        tp_addresses.uninstall()
        self.assertIs(pinned.validate_projected, pairs['validate_projected'])
        self.assertIs(gdn_commit_dma.publish, pairs['publish'])
        self.assertIs(pinned.addresses, PINNED_ADDRESSES)

    def test_a_second_install_changes_nothing(self):
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}):
            self.assertGreater(tp_addresses.install(), 0)
            self.assertEqual(tp_addresses.install(), 0)


class FlaggedTwinTests(unittest.TestCase):
    """The twins tp4/next added bind only under their lever flag; unflagged, production keeps the qualified originals."""

    FOUR = {'QWEN_FAST_TP': '4'}

    def setUp(self):
        import extent_attention_replay_tp
        import fused_commit
        import gdn_device_loop_state
        self.reader = extent_attention_replay_tp.PackedExtentReplayReader
        self.state = gdn_device_loop_state.DeviceLoopState
        self.fused_module = sys.modules['fused_commit']
        self.assertIs(self.fused_module, fused_commit)
        self.addCleanup(tp_addresses.uninstall)

    def bound(self, environ):
        import extent_attention_replay_tp
        import gdn_device_loop_state
        tp_addresses.install(dict(environ))
        return (gdn_device_loop_state.DeviceLoopState, extent_attention_replay_tp.PackedExtentReplayReader,
                sys.modules['fused_commit'])

    def test_the_flagged_rows_are_the_three_tp4_next_added(self):
        self.assertEqual(sorted(tp_addresses.FLAGGED_TWINS), [('extent_attention_replay_tp', 'PackedExtentReplayReader'),
                                                              ('gdn_device_loop_state', 'DeviceLoopState')])
        self.assertEqual(list(tp_addresses.FLAGGED_MODULE_TWINS), ['fused_commit'])
        for key in tp_addresses.FLAGGED_TWINS:
            self.assertIn(key, [row[:2] for row in tp_addresses.TWINS])

    def test_flags_unset_or_zero_bind_exactly_the_s2_classes(self):
        for extra in ({}, {'QWEN_FAST_TP4_GDN_GLUE': '0', 'QWEN_FAST_TP4_ATTN_FOLD': '0', 'QWEN_FAST_FUSED_COMMIT': '0'}):
            with self.subTest(extra=extra):
                state, reader, fused = self.bound(dict(self.FOUR, **extra))
                self.assertIs(state, self.state)
                self.assertIs(reader, self.reader)
                self.assertIs(fused, self.fused_module)
                self.assertNotEqual(state.__module__, 'gdn_device_loop_state_tp')
                self.assertNotEqual(reader.__module__, 'extent_attention_fold_tp')
                tp_addresses.uninstall()

    def test_the_other_twins_stay_bound_with_the_flags_off(self):
        rows, modules = tp_addresses.bound_twins(dict(self.FOUR))
        self.assertEqual(len(rows), len(tp_addresses.TWINS) - 2)
        self.assertEqual([row[0] for row in modules], ['extent_attention_replay', 'quad_draft'])
        tp_addresses.install(dict(self.FOUR))
        import quad_draft
        self.assertEqual(quad_draft.__name__, 'quad_draft_tp')

    def test_each_flag_binds_its_own_twin_only(self):
        cases = (('QWEN_FAST_TP4_GDN_GLUE', 0), ('QWEN_FAST_TP4_ATTN_FOLD', 1), ('QWEN_FAST_FUSED_COMMIT', 2))
        for flag, index in cases:
            with self.subTest(flag=flag):
                bound = self.bound(dict(self.FOUR, **{flag: '1'}))
                originals = (self.state, self.reader, self.fused_module)
                for place, (got, original) in enumerate(zip(bound, originals)):
                    if place == index:
                        self.assertIsNot(got, original)
                    else:
                        self.assertIs(got, original)
                tp_addresses.uninstall()
        self.assertEqual(self.bound(dict(self.FOUR, QWEN_FAST_TP4_GDN_GLUE='1'))[0].__module__, 'gdn_device_loop_state_tp')
        tp_addresses.uninstall()
        self.assertEqual(self.bound(dict(self.FOUR, QWEN_FAST_TP4_ATTN_FOLD='1'))[1].__module__, 'extent_attention_fold_tp')
        tp_addresses.uninstall()
        self.assertEqual(self.bound(dict(self.FOUR, QWEN_FAST_FUSED_COMMIT='1'))[2].__name__, 'fused_commit_tp')

    def test_a_malformed_flag_is_refused_not_ignored(self):
        for flag in ('QWEN_FAST_TP4_GDN_GLUE', 'QWEN_FAST_TP4_ATTN_FOLD', 'QWEN_FAST_FUSED_COMMIT'):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                tp_addresses.install(dict(self.FOUR, **{flag: '2'}))


if __name__ == '__main__':
    unittest.main()
