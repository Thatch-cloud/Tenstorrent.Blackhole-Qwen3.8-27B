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


if __name__ == '__main__':
    unittest.main()
