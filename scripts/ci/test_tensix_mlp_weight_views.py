from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tensix_mlp_view_gate import NATIVE_SOURCES, SOURCES, prerequisite, qualify
from tensix_mlp_weight_views import WEIGHTS, qualify_views, weight_views


class Weight(SimpleNamespace):
    def memory_config(self):
        return self.memory


def fixture():
    def tensor(shape, dtype, addresses):
        return Weight(shape=shape, padded_shape=shape, dtype=dtype, layout='tile', memory='dram', addresses=addresses)
    def shards(value):
        return [Weight(**{**value.__dict__, 'buffer_address': lambda address=address: address})
            for address in value.addresses]
    operations = SimpleNamespace(bfloat4_b='bfloat4_b', bfloat8_b='bfloat8_b', TILE_LAYOUT='tile',
        DRAM_MEMORY_CONFIG='dram', get_device_tensors=shards,
        experimental=SimpleNamespace(view=Mock(side_effect=lambda value, shape: tensor(shape, value.dtype, value.addresses))))
    weights = {name: tensor((inner, width), dtype, (1024 * (index + 1), 2048 * (index + 1)))
        for index, (name, (inner, width, dtype)) in enumerate(WEIGHTS.items())}
    return operations, weights


def simulator_fixture():
    operations, weights = fixture()
    unused_views, checks = weight_views(operations, weights)
    return dict(passed=True, closed_cleanly=True, backend='simulator', stage='complete',
        sources={name: 'a' * 64 for name in SOURCES}, native_sources=NATIVE_SOURCES, weight_views=checks,
        shard_axes=dict(gate=1, up=1, down=0), program_cache=[0, 0],
        checks=[dict(weight=name, chip=chip, contents_exact=True, native_unchanged_after_view_release=True,
            canonical_identity=True, distinct_shards=True) for name in WEIGHTS for chip in range(2)])


class MlpWeightViewTests(unittest.TestCase):
    def test_both_buffers_are_borrowed_and_native_rank_is_unchanged(self):
        operations, weights = fixture()
        views, checks = weight_views(operations, weights)
        self.assertEqual(operations.experimental.view.call_count, 3)
        for name, weight in weights.items():
            self.assertEqual(len(weight.shape), 2)
            self.assertEqual(views[name].shape, (1, 1, *weight.shape))
            self.assertEqual(views[name].addresses, weight.addresses)
        self.assertEqual(qualify_views(checks)['chip_aliases'], 6)

    def test_canonical_tensors_do_not_create_more_views(self):
        operations, weights = fixture()
        views, unused_checks = weight_views(operations, weights)
        operations.experimental.view.reset_mock()
        repeated, unused_checks = weight_views(operations, views)
        operations.experimental.view.assert_not_called()
        self.assertTrue(all(repeated[name] is views[name] for name in WEIGHTS))

    def test_bad_native_specs_fail_before_any_view_is_created(self):
        for field, value in (('shape', (1, 5120, 8704)), ('shape', (8704, 5120)),
                ('padded_shape', (5120, 8736)), ('dtype', 'bfloat8_b'), ('layout', 'row_major'),
                ('memory', 'sharded_dram'), ('addresses', (1024,)), ('addresses', (1024, 0))):
            operations, weights = fixture()
            setattr(weights['gate'], field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                weight_views(operations, weights)
            operations.experimental.view.assert_not_called()

    def test_reallocation_and_mutating_native_metadata_fail(self):
        for failure in ('new_buffer', 'native_metadata', 'wrong_shape'):
            operations, weights = fixture()
            original = operations.experimental.view.side_effect
            def broken(value, shape):
                result = original(value, shape)
                if failure == 'new_buffer':
                    result.addresses = (9090, 9090)
                elif failure == 'native_metadata':
                    value.shape = shape
                else:
                    result.shape = (1, *shape)
                return result
            operations.experimental.view.side_effect = broken
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                weight_views(operations, weights)

    def test_each_chip_metadata_is_validated(self):
        operations, weights = fixture()
        original = operations.get_device_tensors
        def wrong_chip(value):
            shards = original(value)
            shards[1].dtype = 'bf16'
            return shards
        operations.get_device_tensors = wrong_chip
        with self.assertRaises(ValueError):
            weight_views(operations, weights)


class MlpViewGateTests(unittest.TestCase):
    def test_outer_exit_and_exact_report_are_required(self):
        report = simulator_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / 'tensix-mlp-view-simulator.json'
            status_path = root / 'tensix-mlp-view-simulator.exit-status'
            report_path.write_text(json.dumps(report))
            with patch('tensix_mlp_view_gate.hashes', return_value=report['sources']):
                with self.assertRaises(FileNotFoundError):
                    prerequisite(root, NATIVE_SOURCES)
                status_path.write_text('1\n')
                with self.assertRaisesRegex(ValueError, 'wrapper exit'):
                    prerequisite(root, NATIVE_SOURCES)
                status_path.write_text('0\n')
                result = prerequisite(root, NATIVE_SOURCES)
                self.assertTrue(result['gate']['passed'])
                self.assertEqual(len(result['report_sha256']), 64)
                changed = deepcopy(report)
                changed['checks'].pop()
                report_path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    prerequisite(root, NATIVE_SOURCES)

    def test_exact_source_matched_full_shape_result_passes(self):
        report = simulator_fixture()
        self.assertEqual(qualify(report, report['sources'], NATIVE_SOURCES)['chip_checks'], 6)

    def test_incomplete_or_changed_evidence_fails(self):
        for field, value in (('passed', False), ('closed_cleanly', False), ('stage', 'uploaded'),
                ('error', 'failed'), ('sources', {}), ('native_sources', {}), ('shard_axes', dict(gate=1, up=1, down=1)),
                ('program_cache', [0, 1]), ('program_cache', [False, False]), ('checks', []), ('weight_views', {})):
            reference = simulator_fixture()
            with self.subTest(field=field), self.assertRaises(ValueError):
                qualify({**reference, field: value}, reference['sources'], NATIVE_SOURCES)

    def test_alias_or_content_failure_cannot_be_hidden(self):
        for failure in ('address', 'contents_exact', 'native_unchanged_after_view_release', 'canonical_identity', 'distinct_shards'):
            reference = simulator_fixture()
            report = deepcopy(reference)
            if failure == 'address':
                report['weight_views']['down']['candidate']['shards'][1]['address'] += 32
            else:
                report['checks'][0][failure] = False
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                qualify(report, reference['sources'], NATIVE_SOURCES)


if __name__ == '__main__':
    unittest.main()
