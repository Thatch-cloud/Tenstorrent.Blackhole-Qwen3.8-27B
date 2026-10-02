"""QWEN_FAST_TRACED_PUBLISH's K/V fusion at the served width (docs/tp4-traced-publish.md).

dflash_traced_publish._fused_kv_history_prepare sliced (1, 4, ...): four KV heads per chip, the pair's. At four cards a chip holds
two (tp_shapes.draft_kv_heads) and the drafter's cache is the sibling draft_kv_history_tp.DraftKVHistory. These tests hold:

  - the fusion at the sibling's two heads is bit-for-bit the sibling's own six-op chain, every accepted prefix, ramp, steady and the
    transitional round, and its slices carry the head count the bank has (recorded: torch clamps an over-long slice silently, so a
    surviving literal 4 would not fail on values alone);
  - with QWEN_FAST_TP_KV_SLIDE=1 (every four-card traffic profile) the install DECLINES, because the sibling's prepare is then one slide
    op per bank and the fusion would replace it with two ops and a copy; the decline names its reason under the audit flag;
  - a subclass the module does not recognise, and a bank that is not the served width's, decline too;
  - the pair keeps its four heads and its install exactly (the pair's tests, test_dflash_traced_publish, run beside this file);
  - the flag stays 0 in every four-card profile: this change leaves the lever OFF.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from draft_head_preparation import rope_reference
import test_dflash_device_publish as publish_fixtures
import test_dflash_round_b1 as round_b1_fixtures
import draft_kv_history
import draft_kv_history_tp
import tp_shapes

HERE = Path(__file__).resolve().parent


class RecordingOps:
    """Torch stand-ins for the ttnn calls the cache makes, recording every slice's end so a stale head literal shows."""

    def __init__(self):
        self.slices = []
        self.bfloat16 = torch.bfloat16
        self.TILE_LAYOUT, self.DRAM_MEMORY_CONFIG = 'tile', 'dram'
        self.copy = Mock(side_effect=lambda source, destination: destination.copy_(source))
        self.synchronize_device = Mock()
        self.deallocate = Mock()
        self.zeros_like = torch.zeros_like

    def from_torch(self, value, **kwargs):
        return value.clone()

    def ReplicateTensorToMesh(self, mesh):
        return mesh

    def slice(self, value, start, end):
        self.slices.append(tuple(end))
        return value[tuple(slice(first, last) for first, last in zip(start, end, strict=True))]

    def pad(self, value, padding, fill):
        return torch.nn.functional.pad(value, tuple(item for pair in reversed(padding) for item in pair), value=fill)

    def concat(self, values, dim, **kwargs):
        return torch.cat(values, dim=dim)

    def get_device_tensors(self, value):
        return [value] * tp_shapes.chip_count()

    def to_torch(self, value):
        return value


def address(operations, value):
    pointer = value.untyped_storage().data_ptr()
    return pointer, pointer + 1


class Fixture:
    heads = 2
    cls = draft_kv_history_tp.DraftKVHistory

    def features(self, rows, seed):
        return torch.randn((1, 1, rows, 5120), generator=torch.Generator().manual_seed(seed)).bfloat16()

    def project(self, operations, inputs, query, tables, retain, *, parameters):
        rows = inputs.shape[2]
        width = self.heads * 128
        value = (inputs[..., :width] * (parameters['layer'] + 1)).reshape(1, rows, self.heads, 128).transpose(1, 2).contiguous()
        return dict(k=retain(rope_reference(value, *tables)), v=retain(value))

    @contextmanager
    def fixture(self, features, position, *, layers=2, cls=None):
        cls = cls or self.cls
        operations = RecordingOps()
        patches = [patch('draft_kv_history.project_key_value', side_effect=self.project),
                   patch('draft_kv_projection.project_key_value', side_effect=self.project),
                   patch('draft_kv_history.addresses', side_effect=address),
                   patch('draft_kv_history.release_owned',
                         side_effect=lambda operations, owned: [operations.deallocate(value) for value in owned])]
        if issubclass(cls, draft_kv_history_tp.DraftKVHistory):
            patches.append(patch('draft_kv_history_tp.project_key_value', side_effect=self.project))
        for item in patches:
            item.start()
        try:
            cache = cls(operations, object(), [dict(layer=layer) for layer in range(layers)], features,
                        position=position, history_rows=min(position, 2048))
            try:
                yield cache, operations
            finally:
                cache.close()
        finally:
            for item in reversed(patches):
                item.stop()


def four_cards(slide=None, audit=None):
    environment = {'QWEN_FAST_TP': '4'}
    if slide is not None:
        environment['QWEN_FAST_TP_KV_SLIDE'] = slide
    if audit is not None:
        environment['QWEN_FAST_PACKED_AUDIT'] = audit
    return patch.dict(os.environ, environment)


def same_bits(left, right):
    return torch.equal(left.view(torch.int16), right.view(torch.int16))


class FourCardEagerFusionTests(Fixture, unittest.TestCase):
    def test_the_fusion_equals_the_siblings_chain_for_every_prefix_and_slices_two_heads(self):
        from dflash_traced_publish import install_fused_kv_history

        position = 4093
        features = self.features(2048, position)
        with four_cards(slide='0'):
            with self.fixture(features, position) as (general, general_ops), self.fixture(features, position) as (fused, fused_ops):
                restore = install_fused_kv_history(fused)
                self.assertIsNotNone(restore)
                self.assertIn('prepare', vars(fused))
                try:
                    for prefix in range(1, 33):
                        candidate = self.features(32, general.position)
                        general.commit(general.prepare(candidate, prefix, position=general.position))
                        fused.commit(fused.prepare(candidate, prefix, position=fused.position))
                        for layer, (left, right) in enumerate(zip(general.active, fused.active, strict=True)):
                            for name in ('k', 'v'):
                                self.assertEqual(tuple(right[name].shape), (1, 2, 2048, 128))
                                self.assertTrue(same_bits(left[name], right[name]), 'layer %d %s prefix %d' % (layer, name, prefix))
                finally:
                    restore()
                self.assertNotIn('prepare', vars(fused))
                # Every slice the fused rounds made is 2 heads wide (the constructor's own slices are history_rows wide at 2 too).
                banks = [end for end in fused_ops.slices if end[3] == 128]
                self.assertTrue(banks)
                self.assertEqual({end[1] for end in banks}, {2})

    def test_the_transitional_round_into_2048_and_the_ramp_match(self):
        from dflash_traced_publish import install_fused_kv_history

        for position, prefix in ((2030, 32), (170, 7)):
            features = self.features(position, position)
            with self.subTest(position=position), four_cards(slide='0'):
                with self.fixture(features, position) as (general, unused), self.fixture(features, position) as (fused, unused_too):
                    restore = install_fused_kv_history(fused)
                    try:
                        candidate = self.features(32, general.position)
                        general.commit(general.prepare(candidate, prefix, position=general.position))
                        fused.commit(fused.prepare(candidate, prefix, position=fused.position))
                    finally:
                        restore()
                    self.assertEqual(general.history_rows, fused.history_rows)
                    for left, right in zip(general.active, fused.active, strict=True):
                        for name in ('k', 'v'):
                            self.assertTrue(same_bits(left[name], right[name]))

    def test_install_publish_options_installs_and_restores_both_overrides(self):
        from dflash_traced_publish import install_publish_options

        with four_cards(slide='0'), self.fixture(self.features(2048, 4093), 4093) as (cache, unused):
            drafter = SimpleNamespace(kv_history=cache)
            restore = install_publish_options(drafter, merge_release=False, fused_steady_state=True)
            self.assertIn('prepare', vars(cache))
            self.assertIn('prepare_publication', vars(drafter))
            restore()
            self.assertNotIn('prepare', vars(cache))
            self.assertNotIn('prepare_publication', vars(drafter))

    def test_slide_unset_is_the_chain_not_the_slide(self):
        from dflash_traced_publish import install_fused_kv_history

        with four_cards(), self.fixture(self.features(2048, 4093), 4093) as (cache, unused):
            restore = install_fused_kv_history(cache)
            self.assertIsNotNone(restore, 'QWEN_FAST_TP_KV_SLIDE unset keeps the eager chain, which the fusion shortens')
            restore()


class FourCardDeclineTests(Fixture, unittest.TestCase):
    def test_with_the_slide_on_the_install_declines_and_leaves_prepare_alone(self):
        from dflash_traced_publish import install_fused_kv_history

        with four_cards(slide='1'), self.fixture(self.features(2048, 4093), 4093) as (cache, unused):
            self.assertIsNone(install_fused_kv_history(cache))
            self.assertNotIn('prepare', vars(cache))

    def test_the_decline_names_its_reason_under_the_audit_flag_and_is_silent_without_it(self):
        from dflash_traced_publish import FUSION_DECLINED_LINE, install_fused_kv_history

        with four_cards(slide='1', audit='1'), self.fixture(self.features(2048, 4093), 4093) as (cache, unused):
            with patch('dflash_packed_proposal_coordinator.audit_log') as log:
                self.assertIsNone(install_fused_kv_history(cache))
            log.assert_called_once_with(FUSION_DECLINED_LINE, reason='tp_slide_live')
        with four_cards(slide='1', audit='0'), self.fixture(self.features(2048, 4093), 4093) as (cache, unused):
            with patch('dflash_packed_proposal_coordinator.audit_log') as log:
                self.assertIsNone(install_fused_kv_history(cache))
            log.assert_not_called()

    def test_install_publish_options_keeps_the_history_fusion_and_skips_the_kv_override_under_the_slide(self):
        from dflash_traced_publish import install_publish_options

        with four_cards(slide='1'), self.fixture(self.features(2048, 4093), 4093) as (cache, unused):
            drafter = SimpleNamespace(kv_history=cache, calls=[])
            restore = install_publish_options(drafter, merge_release=False, fused_steady_state=True)
            self.assertNotIn('prepare', vars(cache))
            self.assertIn('prepare_publication', vars(drafter))
            restore()
            self.assertNotIn('prepare_publication', vars(drafter))

    def test_an_unrecognised_subclass_with_its_own_prepare_declines(self):
        from dflash_traced_publish import install_fused_kv_history

        class Other(draft_kv_history_tp.DraftKVHistory):
            def prepare(self, features, prefix, *, position):
                return super().prepare(features, prefix, position=position)

        with four_cards(slide='0'), self.fixture(self.features(2048, 4093), 4093, cls=Other) as (cache, unused):
            self.assertIsNone(install_fused_kv_history(cache))

    def test_a_sibling_whose_banks_are_not_the_served_width_declines(self):
        from dflash_traced_publish import install_fused_kv_history

        with four_cards(slide='0'), self.fixture(self.features(2048, 4093), 4093) as (cache, unused):
            cache.active[0]['k'] = torch.zeros((1, 4, 2048, 128), dtype=torch.bfloat16)
            self.assertIsNone(install_fused_kv_history(cache))

    def test_the_pairs_text_patched_slide_scope_is_never_installed_over_the_sibling(self):
        from dflash_traced_publish import install_fused_kv_history

        def marked(*args, **kwargs):
            raise AssertionError('not called')

        marked._draft_kv_slide = True
        with four_cards(slide='0'), self.fixture(self.features(2048, 4093), 4093) as (cache, unused):
            with patch.object(draft_kv_history.DraftKVHistory, 'prepare', marked):
                self.assertIsNone(install_fused_kv_history(cache))


class PairIsUnchangedTests(Fixture, unittest.TestCase):
    heads = 4
    cls = draft_kv_history.DraftKVHistory

    def test_the_pairs_class_keeps_four_heads_with_or_without_the_four_card_environment(self):
        from dflash_traced_publish import PAIR_KV_HEADS, install_fused_kv_history

        self.assertEqual(PAIR_KV_HEADS, 4)
        self.assertEqual(tp_shapes.geometry(2).draft_kv_heads, PAIR_KV_HEADS)
        position = 4093
        features = self.features(2048, position)
        with self.fixture(features, position) as (general, unused), self.fixture(features, position) as (fused, fused_ops):
            restore = install_fused_kv_history(fused)
            self.assertIsNotNone(restore)
            try:
                candidate = self.features(32, general.position)
                general.commit(general.prepare(candidate, 9, position=general.position))
                fused.commit(fused.prepare(candidate, 9, position=fused.position))
            finally:
                restore()
            self.assertEqual({end[1] for end in fused_ops.slices[-8:] if end[3] == 128}, {4})
            for left, right in zip(general.active, fused.active, strict=True):
                for name in ('k', 'v'):
                    self.assertTrue(same_bits(left[name], right[name]))


class FourCardHistoryWriteTests(unittest.TestCase):
    """What the flag buys at four cards with the slide on: DFlashDevice.prepare_publication's fused steady-state branch, which under
    QWEN_FAST_ROUND_B1 (the image's own setting) is C7: the 2048-row feature-history write is skipped where nothing can read it.
    With the flag at 0 (the traffic profile) the general five-operation write runs for every user, every round. The branch has no
    width in it (5120 is the hidden size); this holds it at four-card tap widths (1,280 columns a chip)."""

    def publication(self, fused):
        tap = lambda: SimpleNamespace(shape=(1, 1, 4096, tp_shapes.geometry(4).draft_taps), dtype='bf16')
        with patch.dict(os.environ, {'QWEN_FAST_TP': '4'}), patch.object(publish_fixtures, 'make_feature_tap', tap):
            return round_b1_fixtures.recorded_publication(True, fused=fused)

    def test_the_flag_off_runs_the_five_operation_write_and_on_skips_all_of_it(self):
        off_names, off_touching, off_device, off_pending = self.publication(False)
        on_names, on_touching, on_device, on_pending = self.publication(True)
        self.assertEqual(off_touching, ['slice', 'copy'])
        self.assertEqual(on_touching, [], 'no operation reads or writes either history buffer')
        index = off_names.index('concat') - 1
        self.assertEqual(off_names[index:index + 5], ['slice', 'concat', 'slice', 'pad', 'copy'])
        self.assertEqual(on_names, off_names[:index] + off_names[index + 5:], 'every other operation, in order')
        self.assertTrue(on_device.history_stale)
        self.assertFalse(getattr(off_device, 'history_stale', False))
        self.assertIs(on_pending.history, on_device.spare_history, 'commit still swaps the same pair')
        self.assertEqual((on_pending.rows, on_pending.prefix), (off_pending.rows, off_pending.prefix))


class FlagStaysOffTests(unittest.TestCase):
    def test_no_profile_turns_traced_publish_on(self):
        profiles = json.loads((HERE / 'qwen_c2_profiles.json').read_text())
        profiles = profiles.get('profiles', profiles)
        seen = 0
        for name, profile in profiles.items():
            value = profile.get('env', {}).get('QWEN_FAST_TRACED_PUBLISH')
            if value is not None:
                seen += 1
                self.assertEqual(value, '0', name)
        self.assertGreater(seen, 10)


if __name__ == '__main__':
    unittest.main()
