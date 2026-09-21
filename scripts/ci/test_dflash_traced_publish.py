from contextlib import contextmanager
import os
from types import SimpleNamespace
import unittest
import unittest.mock
from unittest.mock import Mock, patch

import torch

from draft_head_preparation import rope_reference, rope_tables
import draft_kv_history
from draft_kv_history import KV_SHAPE, QUERY_SHAPE, DraftKVHistory


class FlagTests(unittest.TestCase):
    def test_default_is_off(self):
        from dflash_traced_publish import traced_publish_enabled

        self.assertFalse(traced_publish_enabled({}))

    def test_only_0_or_1_accepted(self):
        from dflash_traced_publish import traced_publish_enabled

        self.assertFalse(traced_publish_enabled({'QWEN_FAST_TRACED_PUBLISH': '0'}))
        self.assertTrue(traced_publish_enabled({'QWEN_FAST_TRACED_PUBLISH': '1'}))
        for bad in ('true', 'yes', '2', ' 1', ''):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    traced_publish_enabled({'QWEN_FAST_TRACED_PUBLISH': bad})

    def test_reads_the_real_os_environ_by_default(self):
        from dflash_traced_publish import traced_publish_enabled

        with unittest.mock.patch.dict(os.environ, {'QWEN_FAST_TRACED_PUBLISH': '1'}):
            self.assertTrue(traced_publish_enabled())
        self.assertNotIn('QWEN_FAST_TRACED_PUBLISH', os.environ)
        self.assertFalse(traced_publish_enabled())


class FakeDrafter:
    """A minimal DFlashDevice-shaped object: prepare_publication as the class
    defines it (a real method taking both options), so install_publish_options'
    rebind can be checked without any device machinery."""

    def __init__(self):
        self.calls = []

    def prepare_publication(self, features, prefix, *, position, merge_release=False, fused_steady_state=False):
        self.calls.append((features, prefix, position, merge_release, fused_steady_state))
        return 'publication'


class InstallPublishOptionsTests(unittest.TestCase):
    def test_the_rebound_method_forces_both_options_together(self):
        from dflash_traced_publish import install_publish_options

        drafter = FakeDrafter()
        restore = install_publish_options(drafter, merge_release=True, fused_steady_state=True)
        result = drafter.prepare_publication('features', 5, position=100)
        self.assertEqual(result, 'publication')
        self.assertEqual(drafter.calls, [('features', 5, 100, True, True)])
        restore()

    def test_either_option_can_be_forced_independently(self):
        from dflash_traced_publish import install_publish_options

        drafter = FakeDrafter()
        restore = install_publish_options(drafter, merge_release=False, fused_steady_state=True)
        drafter.prepare_publication('f', 1, position=2)
        self.assertEqual(drafter.calls[-1], ('f', 1, 2, False, True))
        restore()
        restore = install_publish_options(drafter, merge_release=True, fused_steady_state=False)
        drafter.prepare_publication('f', 1, position=2)
        self.assertEqual(drafter.calls[-1], ('f', 1, 2, True, False))
        restore()

    def test_restore_removes_the_override_and_the_class_method_runs_again(self):
        from dflash_traced_publish import install_publish_options

        drafter = FakeDrafter()
        restore = install_publish_options(drafter, merge_release=True, fused_steady_state=True)
        self.assertIn('prepare_publication', drafter.__dict__)
        restore()
        self.assertNotIn('prepare_publication', drafter.__dict__)
        drafter.prepare_publication('f', 1, position=2)
        self.assertEqual(drafter.calls[-1], ('f', 1, 2, False, False), 'both default off after restore')

    def test_double_install_is_refused(self):
        from dflash_traced_publish import install_publish_options

        drafter = FakeDrafter()
        restore = install_publish_options(drafter, merge_release=True, fused_steady_state=True)
        with self.assertRaises(ValueError):
            install_publish_options(drafter, merge_release=True, fused_steady_state=True)
        restore()

    def test_non_bool_options_are_refused(self):
        from dflash_traced_publish import install_publish_options

        drafter = FakeDrafter()
        for kwargs in (dict(merge_release=1, fused_steady_state=True), dict(merge_release=True, fused_steady_state=1)):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    install_publish_options(drafter, **kwargs)

    def test_works_on_any_object_not_just_a_real_drafter(self):
        """The shim only ever sets and later deletes an instance attribute - no
        dependency on DFlashDevice's own class machinery."""
        from dflash_traced_publish import install_publish_options

        class Bare:
            def prepare_publication(self, features, prefix, *, position, merge_release=False, fused_steady_state=False):
                return (features, prefix, position, merge_release, fused_steady_state)

        drafter = Bare()
        restore = install_publish_options(drafter, merge_release=True, fused_steady_state=True)
        self.assertEqual(drafter.prepare_publication('f', 1, position=2), ('f', 1, 2, True, True))
        restore()
        self.assertEqual(drafter.prepare_publication('f', 1, position=2), ('f', 1, 2, False, False))


class FusedKVHistoryFixture:
    """Real-tensor DraftKVHistory fixture for install_fused_kv_history /
    _fused_kv_history_prepare - the K/V-cache half of QWEN_FAST_TRACED_PUBLISH's
    steady-state fusion, re-homed here (not inside DraftKVHistory.prepare's own
    source) after v29 (commit b05c8af8) broke draft_kv_slide_adapter.build_prepare's
    exact-text match by restructuring prepare() in place - see dflash_traced_publish's
    module docstring. Mirrors test_draft_kv_history.py's own fixture, with one
    addition: project_key_value is ALSO patched on draft_kv_projection itself (not
    just draft_kv_history's re-exported name), because _fused_kv_history_prepare
    imports it directly from there, lazily, at call time - draft_kv_history.py's own
    prepare() must stay byte-identical, so it could gain no shared import seam."""

    def operations(self):
        return SimpleNamespace(bfloat16=torch.bfloat16, TILE_LAYOUT='tile', DRAM_MEMORY_CONFIG='dram',
            from_torch=lambda value, **kwargs: value.clone(), ReplicateTensorToMesh=lambda mesh: mesh,
            slice=lambda value, start, end: value[tuple(slice(first, last) for first, last in zip(start, end, strict=True))],
            pad=lambda value, padding, fill: torch.nn.functional.pad(value, tuple(item for pair in reversed(padding) for item in pair), value=fill),
            concat=lambda values, dim, **kwargs: torch.cat(values, dim=dim), zeros_like=torch.zeros_like,
            copy=Mock(side_effect=lambda source, destination: destination.copy_(source)),
            synchronize_device=Mock(), deallocate=Mock(), get_device_tensors=lambda value: [value, value], to_torch=lambda value: value)

    def features(self, rows, seed):
        return torch.randn((1, 1, rows, 5120), generator=torch.Generator().manual_seed(seed)).bfloat16()

    @staticmethod
    def address(operations, value):
        pointer = value.untyped_storage().data_ptr()
        return pointer, pointer + 1

    @staticmethod
    def project(operations, inputs, query, tables, retain, *, parameters):
        rows = inputs.shape[2]
        if torch.count_nonzero(query).item():
            raise AssertionError('The projection reads a zero query')
        value = (inputs[..., :512] * (parameters['layer'] + 1)).reshape(1, rows, 4, 128).transpose(1, 2).contiguous()
        return dict(k=retain(rope_reference(value, *tables)), v=retain(value))

    @contextmanager
    def fixture(self, features, position, *, layers=2):
        operations = self.operations()
        with patch('draft_kv_history.project_key_value', side_effect=self.project), \
                patch('draft_kv_projection.project_key_value', side_effect=self.project), \
                patch('draft_kv_history.addresses', side_effect=self.address), \
                patch('draft_kv_history.release_owned', side_effect=lambda operations, owned: [operations.deallocate(value) for value in owned]):
            cache = DraftKVHistory(operations, object(), [dict(layer=layer) for layer in range(layers)],
                features, position=position, history_rows=min(position, 2048))
            try:
                yield cache, operations
            finally:
                cache.close()


class FusedOverrideBehaviorTests(FusedKVHistoryFixture, unittest.TestCase):
    """(b)/(c) from the v29 fix instructions: the installed override must match the
    general path bit-for-bit in steady state across every accepted prefix, fall
    through to the untouched original before the ramp completes, and leave
    kv_history.prepare completely alone (the true default path, byte-identical)
    whenever fused_steady_state is off."""

    def test_fused_override_matches_the_general_path_bit_for_bit(self):
        from dflash_traced_publish import install_fused_kv_history

        position = 4093
        features = self.features(2048, position)
        with self.fixture(features, position) as (general, unused_general_ops), \
                self.fixture(features, position) as (fused, unused_fused_ops):
            restore = install_fused_kv_history(fused)
            self.assertIsNotNone(restore)
            self.assertIn('prepare', vars(fused))
            try:
                for prefix in range(1, 33):
                    self.assertEqual(general.history_rows, 2048, 'steady state throughout')
                    self.assertEqual(fused.history_rows, 2048, 'steady state throughout')
                    candidate = self.features(32, general.position)
                    self.assertEqual(general.position, fused.position)
                    general.commit(general.prepare(candidate, prefix, position=general.position))
                    fused.commit(fused.prepare(candidate, prefix, position=fused.position))
                    for layer, (general_pair, fused_pair) in enumerate(zip(general.active, fused.active, strict=True)):
                        for name in ('k', 'v'):
                            self.assertTrue(torch.equal(general_pair[name].view(torch.int16), fused_pair[name].view(torch.int16)),
                                'layer %d %s diverged at prefix=%d' % (layer, name, prefix))
            finally:
                restore()
            self.assertNotIn('prepare', vars(fused))

    def test_override_falls_through_to_the_original_before_the_ramp_completes(self):
        from dflash_traced_publish import install_fused_kv_history

        position = 170
        features = self.features(position, position)
        with self.fixture(features, position) as (plain, unused_plain_ops), \
                self.fixture(features, position) as (overridden, unused_over_ops):
            restore = install_fused_kv_history(overridden)
            try:
                candidate = self.features(32, plain.position)
                plain.commit(plain.prepare(candidate, 7, position=plain.position))
                overridden.commit(overridden.prepare(candidate, 7, position=overridden.position))
            finally:
                restore()
            self.assertEqual(overridden.history_rows, 177)
            self.assertEqual(plain.history_rows, overridden.history_rows)
            for layer, (plain_pair, over_pair) in enumerate(zip(plain.active, overridden.active, strict=True)):
                for name in ('k', 'v'):
                    self.assertTrue(torch.equal(plain_pair[name].view(torch.int16), over_pair[name].view(torch.int16)),
                        'layer %d %s diverged during the ramp' % (layer, name))

    def test_a_transitional_round_that_first_reaches_2048_also_matches(self):
        """history_rows < 2048 but history_rows + prefix >= 2048: the one round
        _fused_kv_history_prepare's own comment calls out as the sole case where the
        drop offset (history_rows + prefix - rows) is nonzero."""
        from dflash_traced_publish import install_fused_kv_history

        position = 2030
        features = self.features(position, position)
        with self.fixture(features, position) as (general, unused_general_ops), \
                self.fixture(features, position) as (fused, unused_fused_ops):
            restore = install_fused_kv_history(fused)
            try:
                candidate = self.features(32, general.position)
                self.assertEqual(general.history_rows + 32, 2062, 'crosses 2048 mid-prefix')
                general.commit(general.prepare(candidate, 32, position=general.position))
                fused.commit(fused.prepare(candidate, 32, position=fused.position))
            finally:
                restore()
            self.assertEqual(general.history_rows, 2048)
            self.assertEqual(fused.history_rows, 2048)
            for layer, (general_pair, fused_pair) in enumerate(zip(general.active, fused.active, strict=True)):
                for name in ('k', 'v'):
                    self.assertTrue(torch.equal(general_pair[name].view(torch.int16), fused_pair[name].view(torch.int16)),
                        'layer %d %s diverged on the transitional round' % (layer, name))

    def test_install_publish_options_leaves_kv_history_untouched_when_fusion_is_off(self):
        """(c): the default (flag-off) path never installs anything - kv_history.
        prepare stays the exact class method, byte-identical to before this fix."""
        from dflash_traced_publish import install_publish_options

        with self.fixture(self.features(170, 3), 170) as (cache, unused_ops):
            drafter = SimpleNamespace(kv_history=cache)
            restore = install_publish_options(drafter, merge_release=True, fused_steady_state=False)
            self.assertNotIn('prepare', vars(cache))
            self.assertEqual(cache.prepare, draft_kv_history.DraftKVHistory.prepare.__get__(cache))
            restore()
            self.assertNotIn('prepare', vars(cache))

    def test_install_publish_options_installs_and_restores_the_kv_history_override_together(self):
        from dflash_traced_publish import install_publish_options

        with self.fixture(self.features(2048, 4093), 4093) as (cache, unused_ops):
            drafter = SimpleNamespace(kv_history=cache)
            restore = install_publish_options(drafter, merge_release=False, fused_steady_state=True)
            self.assertIn('prepare', vars(cache))
            self.assertIn('prepare_publication', vars(drafter))
            restore()
            self.assertNotIn('prepare', vars(cache))
            self.assertNotIn('prepare_publication', vars(drafter))

    def test_double_install_on_the_same_kv_history_is_refused(self):
        from dflash_traced_publish import install_fused_kv_history

        with self.fixture(self.features(2048, 4093), 4093) as (cache, unused_ops):
            restore = install_fused_kv_history(cache)
            try:
                with self.assertRaises(ValueError):
                    install_fused_kv_history(cache)
            finally:
                restore()

    def test_none_and_non_draft_kv_history_objects_are_a_silent_noop(self):
        from dflash_traced_publish import install_fused_kv_history

        self.assertIsNone(install_fused_kv_history(None))
        self.assertIsNone(install_fused_kv_history(SimpleNamespace(prepare=lambda *a, **k: None)))


class FusedOverrideDeclinesForTheSlideCandidateTests(FusedKVHistoryFixture, unittest.TestCase):
    """(d): install_fused_kv_history must not install - and must not guess at
    reproducing - whenever draft_kv_slide_scope.scoped_publication's own candidate is
    CURRENTLY the live class-level DraftKVHistory.prepare (its _draft_kv_slide
    marker, set by draft_kv_slide_scope.py itself). This module does not attempt the
    "reproduce it" half of the instruction - it cannot verify prepare_slide's own
    transport matches without hardware - so it always takes the "decline" half,
    logging why under QWEN_FAST_PACKED_AUDIT=1."""

    @staticmethod
    def marked_slide_candidate():
        def candidate(self, features, prefix, *, position):
            raise AssertionError('the slide candidate must never actually run here - '
                                  'only its liveness (the _draft_kv_slide marker) is checked')
        candidate._draft_kv_slide = True
        return candidate

    def test_declines_and_logs_reason_under_the_audit_flag(self):
        from dflash_traced_publish import FUSION_DECLINED_LINE, install_fused_kv_history

        with self.fixture(self.features(2048, 4093), 4093) as (cache, unused_ops):
            with patch.object(draft_kv_history.DraftKVHistory, 'prepare', self.marked_slide_candidate()), \
                    patch.dict(os.environ, {'QWEN_FAST_PACKED_AUDIT': '1'}), \
                    patch('dflash_packed_proposal_coordinator.audit_log') as audit_log:
                restore = install_fused_kv_history(cache)
            self.assertIsNone(restore)
            self.assertNotIn('prepare', vars(cache))
            audit_log.assert_called_once_with(FUSION_DECLINED_LINE, reason='slide_candidate_live')

    def test_decline_is_silent_without_the_audit_flag(self):
        from dflash_traced_publish import install_fused_kv_history

        with self.fixture(self.features(170, 5), 170) as (cache, unused_ops):
            self.assertNotIn('QWEN_FAST_PACKED_AUDIT', os.environ)
            with patch.object(draft_kv_history.DraftKVHistory, 'prepare', self.marked_slide_candidate()), \
                    patch('dflash_packed_proposal_coordinator.audit_log') as audit_log:
                restore = install_fused_kv_history(cache)
            self.assertIsNone(restore)
            audit_log.assert_not_called()

    def test_installs_normally_once_the_slide_candidate_is_no_longer_live(self):
        """Same cache, same class: live during the patch.object scope (declines),
        gone once it exits (installs) - proves the check is read fresh at install
        time, not cached from a stale first look."""
        from dflash_traced_publish import install_fused_kv_history

        with self.fixture(self.features(2048, 4093), 4093) as (cache, unused_ops):
            with patch.object(draft_kv_history.DraftKVHistory, 'prepare', self.marked_slide_candidate()):
                self.assertIsNone(install_fused_kv_history(cache))
            self.assertNotIn('prepare', vars(cache))
            restore = install_fused_kv_history(cache)
            self.assertIsNotNone(restore)
            self.assertIn('prepare', vars(cache))
            restore()


if __name__ == '__main__':
    unittest.main()
