import types
import unittest
import unittest.mock
from types import SimpleNamespace


def fake_operations():
    operations = SimpleNamespace(bfloat16='bf16', float32='f32', DRAM_MEMORY_CONFIG='dram')
    operations.MatmulMultiCoreReuseMultiCast1DProgramConfig = unittest.mock.Mock(side_effect=lambda **k: object())
    operations.slice = unittest.mock.Mock(side_effect=lambda *a, **k: SimpleNamespace(shape=(1, 1, 5120, 5120)))
    operations.pad = unittest.mock.Mock(side_effect=lambda *a, **k: object())
    operations.matmul = unittest.mock.Mock(side_effect=lambda *a, **k: object())
    operations.typecast = unittest.mock.Mock(side_effect=lambda *a, **k: object())
    operations.rms_norm = unittest.mock.Mock(side_effect=lambda *a, **k: object())
    operations.concat = unittest.mock.Mock(side_effect=lambda values, dim: SimpleNamespace(shape=(1, 1, 5120, 5120)))
    operations.copy = unittest.mock.Mock()
    operations.synchronize_device = unittest.mock.Mock()
    return operations


def make_feature_tap():
    return SimpleNamespace(shape=(1, 1, 4096, 2560), dtype='bf16')


def build_device(operations, *, kv_history=None):
    from dflash_device import DFlashDevice

    device = SimpleNamespace(
        operations=operations, mesh='mesh', collectives='collectives', kernel='kernel',
        projection='projection', feature_norm='feature_norm',
        closed=False, pending=None, position=100, history_rows=50,
        history=SimpleNamespace(shape=(1, 1, 2048, 5120)),
        spare_history=SimpleNamespace(shape=(1, 1, 2048, 5120)),
        kv_history=kv_history, owned=[], borrowed=[])
    device.temporaries = types.MethodType(DFlashDevice.temporaries, device)
    device.release_except = types.MethodType(DFlashDevice.release_except, device)
    device.project_features = types.MethodType(DFlashDevice.project_features, device)
    device.prepare_publication = types.MethodType(DFlashDevice.prepare_publication, device)
    return device


def patched(operations):
    """Context manager stack patching dflash_device's module-level helpers so
    project_features/prepare_publication run against plain mock objects instead of
    real feature-projection/collective math - exactly what those functions are
    (deliberately) agnostic to here."""
    return (
        unittest.mock.patch('dflash_device.addresses', side_effect=lambda operations, value: (id(value),)),
        unittest.mock.patch('dflash_device.release_owned'),
        unittest.mock.patch('dflash_device.concatenate_local_features', side_effect=lambda operations, parts: object()),
        unittest.mock.patch('dflash_device.gather_add_projection',
            side_effect=lambda operations, mesh, collectives, partial, **k: object()),
    )


class ProjectFeaturesTests(unittest.TestCase):
    def test_default_scope_syncs_and_releases_on_its_own(self):
        operations = fake_operations()
        device = build_device(operations)
        with unittest.mock.patch('dflash_device.addresses', side_effect=lambda operations, value: (id(value),)), \
                unittest.mock.patch('dflash_device.release_owned') as release, \
                unittest.mock.patch('dflash_device.concatenate_local_features', side_effect=lambda operations, parts: object()), \
                unittest.mock.patch('dflash_device.gather_add_projection',
                    side_effect=lambda operations, mesh, collectives, partial, **k: object()):
            output = device.project_features([make_feature_tap() for _ in range(5)], 1)
        self.assertIsNotNone(output)
        operations.synchronize_device.assert_called_once_with('mesh')
        release.assert_called_once()

    def test_caller_owned_retain_skips_sync_and_release(self):
        operations = fake_operations()
        device = build_device(operations)
        collected = []
        def caller_retain(value):
            collected.append(value)
            return value
        with unittest.mock.patch('dflash_device.addresses', side_effect=lambda operations, value: (id(value),)), \
                unittest.mock.patch('dflash_device.release_owned') as release, \
                unittest.mock.patch('dflash_device.concatenate_local_features', side_effect=lambda operations, parts: object()), \
                unittest.mock.patch('dflash_device.gather_add_projection',
                    side_effect=lambda operations, mesh, collectives, partial, **k: object()):
            output = device.project_features([make_feature_tap() for _ in range(5)], 1, retain=caller_retain)
        self.assertIsNotNone(output)
        operations.synchronize_device.assert_not_called()
        release.assert_not_called()
        self.assertIn(output, collected, "the caller's retain must see every temporary, output included")

    def test_bad_feature_taps_are_refused_before_any_retain_mode_matters(self):
        operations = fake_operations()
        device = build_device(operations)
        with self.assertRaises(ValueError):
            device.project_features([make_feature_tap() for _ in range(4)], 1)
        with self.assertRaises(ValueError):
            device.project_features([make_feature_tap() for _ in range(4)], 1, retain=lambda v: v)


class PreparePublicationTests(unittest.TestCase):
    def test_merge_release_type_is_checked(self):
        operations = fake_operations()
        device = build_device(operations)
        with self.assertRaises(ValueError):
            device.prepare_publication([make_feature_tap() for _ in range(5)], 1, position=100, merge_release='yes')

    def test_default_merge_release_false_is_unconditional_one_sync_no_kv_history(self):
        operations = fake_operations()
        device = build_device(operations, kv_history=None)
        p = patched(operations)
        with p[0], p[1], p[2], p[3]:
            pending = device.prepare_publication([make_feature_tap() for _ in range(5)], 1, position=100)
        self.assertEqual(pending.status, 'prepared')
        # project_features' own sync (default retain=None) plus prepare_publication's
        # own trailing sync - today's unconditional two, no kv_history involved.
        self.assertEqual(operations.synchronize_device.call_count, 2)

    def test_merge_release_with_no_kv_history_still_syncs_once_more(self):
        """No kv_history means nothing else in this call synchronizes, so
        prepare_publication's own trailing fence cannot be skipped even with
        merge_release - only project_features' own (now merged) fence disappears."""
        operations = fake_operations()
        device = build_device(operations, kv_history=None)
        p = patched(operations)
        with p[0], p[1], p[2], p[3]:
            pending = device.prepare_publication([make_feature_tap() for _ in range(5)], 1, position=100, merge_release=True)
        self.assertEqual(pending.status, 'prepared')
        self.assertEqual(operations.synchronize_device.call_count, 1, 'only the trailing fence remains')

    def test_merge_release_with_a_committed_cache_skips_the_trailing_fence(self):
        kv_history = SimpleNamespace(
            prepare=unittest.mock.Mock(side_effect=lambda projected, prefix, position: (
                operations.synchronize_device('mesh'), SimpleNamespace(status='prepared'))[1]))
        operations = fake_operations()
        device = build_device(operations, kv_history=kv_history)
        p = patched(operations)
        with p[0], p[1], p[2], p[3]:
            pending = device.prepare_publication([make_feature_tap() for _ in range(5)], 1, position=100, merge_release=True)
        self.assertEqual(pending.status, 'prepared')
        kv_history.prepare.assert_called_once()
        # kv_history.prepare()'s own synchronize_device call is the ONLY one: project_
        # features' fence was merged away, prepare_publication's own trailing fence was
        # skipped as redundant with it.
        self.assertEqual(operations.synchronize_device.call_count, 1)

    def test_default_false_still_syncs_unconditionally_even_with_a_committed_cache(self):
        """Byte-identical default: merge_release's whole optimisation is inert unless
        explicitly requested, regardless of whether a cache is present."""
        kv_history = SimpleNamespace(
            prepare=unittest.mock.Mock(side_effect=lambda projected, prefix, position: (
                operations.synchronize_device('mesh'), SimpleNamespace(status='prepared'))[1]))
        operations = fake_operations()
        device = build_device(operations, kv_history=kv_history)
        p = patched(operations)
        with p[0], p[1], p[2], p[3]:
            device.prepare_publication([make_feature_tap() for _ in range(5)], 1, position=100)
        # project_features' own sync + kv_history.prepare()'s own sync + prepare_
        # publication's own trailing sync - three, exactly as today.
        self.assertEqual(operations.synchronize_device.call_count, 3)

    def test_fused_steady_state_type_is_checked(self):
        operations = fake_operations()
        device = build_device(operations)
        with self.assertRaises(ValueError):
            device.prepare_publication([make_feature_tap() for _ in range(5)], 1, position=100, fused_steady_state='yes')

    def test_fused_steady_state_never_threads_a_kwarg_into_kv_history_prepare(self):
        """kv_history.prepare's OWN steady-state fusion is no longer selected by an
        argument prepare_publication passes through: draft_kv_slide_adapter.
        build_prepare source-text-patches DraftKVHistory.prepare's exact body at
        combined-runtime attach time, so prepare()'s own source (and therefore its
        signature) must stay untouched (see dflash_device.py's own comment at this
        call site, and dflash_traced_publish's module docstring, for why - v29,
        commit b05c8af8). kv_history.prepare is always called with exactly
        (projected, prefix, position=position); whether it fuses is decided by
        whatever dflash_traced_publish.install_fused_kv_history has - or has not -
        installed on THIS kv_history instance before this call, tested separately in
        test_dflash_traced_publish.py."""
        kv_history = SimpleNamespace(prepare=unittest.mock.Mock(return_value=SimpleNamespace(status='prepared')))
        operations = fake_operations()
        device = build_device(operations, kv_history=kv_history)
        device.history_rows = 2048
        p = patched(operations)
        with p[0], p[1], p[2], p[3]:
            device.prepare_publication([make_feature_tap() for _ in range(5)], 1, position=100, fused_steady_state=True)
        kv_history.prepare.assert_called_once_with(unittest.mock.ANY, 1, position=100)

    def test_fused_steady_state_off_also_never_passes_the_kwarg(self):
        kv_history = SimpleNamespace(prepare=unittest.mock.Mock(return_value=SimpleNamespace(status='prepared')))
        operations = fake_operations()
        device = build_device(operations, kv_history=kv_history)
        p = patched(operations)
        with p[0], p[1], p[2], p[3]:
            device.prepare_publication([make_feature_tap() for _ in range(5)], 1, position=100, fused_steady_state=False)
        kv_history.prepare.assert_called_once_with(unittest.mock.ANY, 1, position=100)

    def test_fused_steady_state_issues_fewer_ops_when_rows_is_2048(self):
        """slice+concat+copy replaces slice+concat+slice+pad+copy for prepare_
        publication's own direct sequence (project_features and kv_history.prepare
        are separately gated and separately tested)."""
        operations = fake_operations()
        device = build_device(operations, kv_history=None)
        device.history_rows = 2048
        p = patched(operations)
        with p[0], p[1], p[2], p[3]:
            device.prepare_publication([make_feature_tap() for _ in range(5)], 1, position=100, fused_steady_state=True)
        general_ops = fake_operations()
        general_device = build_device(general_ops, kv_history=None)
        general_device.history_rows = 2048
        p2 = patched(general_ops)
        with p2[0], p2[1], p2[2], p2[3]:
            general_device.prepare_publication([make_feature_tap() for _ in range(5)], 1, position=100)
        self.assertLess(operations.slice.call_count + operations.pad.call_count,
            general_ops.slice.call_count + general_ops.pad.call_count)
        # project_features pads its own narrow feature taps regardless of
        # fused_steady_state (unrelated to this branch) - the difference is
        # prepare_publication's OWN direct pad call, which the fused path skips
        # entirely (rows == 2048 already, nothing left to pad).
        self.assertEqual(operations.pad.call_count, general_ops.pad.call_count - 1,
            "fused skips prepare_publication's own pad call, and only that one")

    def test_fused_steady_state_falls_back_to_the_general_path_before_rows_is_2048(self):
        """The flag alone does not force the fused branch - rows must also already
        be 2048 (history_rows == 2048 permanently, once reached). Below that,
        fused_steady_state=True runs the SAME op sequence as the default."""
        operations = fake_operations()
        device = build_device(operations, kv_history=None)
        device.history_rows = 50  # rows = min(2048, 50+1) = 51, not 2048
        p = patched(operations)
        with p[0], p[1], p[2], p[3]:
            device.prepare_publication([make_feature_tap() for _ in range(5)], 1, position=100, fused_steady_state=True)
        self.assertGreater(operations.pad.call_count, 0, 'the general path still pads')


class PreparePublicationFusedCorrectnessTests(unittest.TestCase):
    """Real-tensor check for the algebraic identity prepare_publication's own
    fused-branch comment proves: with a committed cache (kv_history not exercised
    here - draft_kv_history's own identity is checked separately, in
    test_draft_kv_history.py, against the real DraftKVHistory), fused_steady_state
    must write the SAME self.spare_history content as the general path, at every
    prefix, once history_rows == 2048."""

    def functional_operations(self):
        import torch

        return SimpleNamespace(bfloat16=torch.bfloat16,
            slice=lambda value, start, end: value[tuple(slice(a, b) for a, b in zip(start, end, strict=True))],
            pad=lambda value, padding, fill: torch.nn.functional.pad(
                value, tuple(item for pair in reversed(padding) for item in pair), value=fill),
            concat=lambda values, dim, **kwargs: torch.cat(values, dim=dim),
            copy=lambda source, destination: destination.copy_(source),
            synchronize_device=lambda mesh: None)

    def run_prepare(self, *, fused_steady_state, prefix, history_rows=2048):
        import torch
        import types as _types
        from dflash_device import DFlashDevice

        operations = self.functional_operations()
        generator = torch.Generator().manual_seed(7)
        history = torch.randn((1, 1, 2048, 5120), generator=generator).bfloat16()
        projected = torch.randn((1, 1, prefix, 5120), generator=generator).bfloat16()
        spare_history = torch.zeros((1, 1, 2048, 5120), dtype=torch.bfloat16)
        device = SimpleNamespace(operations=operations, mesh='mesh', closed=False, pending=None,
            position=100, history_rows=history_rows, history=history, spare_history=spare_history,
            kv_history=None, owned=[], borrowed=[],
            project_features=lambda features, prefix, retain=None: projected)
        device.temporaries = _types.MethodType(DFlashDevice.temporaries, device)
        with unittest.mock.patch('dflash_device.addresses', side_effect=lambda operations, value: (id(value),)), \
                unittest.mock.patch('dflash_device.release_owned'):
            DFlashDevice.prepare_publication(device, [make_feature_tap()] * 5, prefix, position=100,
                fused_steady_state=fused_steady_state)
        return spare_history

    def test_fused_matches_general_at_every_prefix_in_steady_state(self):
        for prefix in range(1, 33):
            with self.subTest(prefix=prefix):
                general = self.run_prepare(fused_steady_state=False, prefix=prefix)
                fused = self.run_prepare(fused_steady_state=True, prefix=prefix)
                import torch

                self.assertTrue(torch.equal(general.view(torch.int16), fused.view(torch.int16)))


if __name__ == '__main__':
    unittest.main()
