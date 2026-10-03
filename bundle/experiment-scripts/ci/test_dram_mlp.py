from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dram_mlp


class DramMlpTests(unittest.TestCase):
    def test_projection_scratch_and_hidden_are_released(self):
        source = SimpleNamespace(shape=(1, 1, 16, 5120), dtype='bf16')
        released, kept, calls = [], [], []
        hidden = object()
        operations = SimpleNamespace(bfloat16='bf16', L1_MEMORY_CONFIG='l1',
            mul=lambda *args, **kwargs: hidden)
        outputs = {name: object() for name in ('gate', 'up', 'down')}
        scratch = []
        def project(ops, value, weight, config, compute, retain, **kwargs):
            calls.append((value, weight, kwargs))
            temporary = object()
            scratch.append(temporary)
            retain(temporary)
            return retain(outputs[weight])
        with patch.object(dram_mlp, 'project', side_effect=project), patch.object(
                dram_mlp, 'release_owned', side_effect=lambda ops, values: released.extend(values)):
            result = dram_mlp.execute(operations, source, {name: name for name in outputs},
                {name: {} for name in outputs}, None, kept.append)
        self.assertIs(result, outputs['down'])
        self.assertEqual(kept, [result])
        self.assertEqual([entry[1] for entry in calls], ['gate', 'up', 'down'])
        self.assertIs(calls[2][0], hidden)
        self.assertTrue(all(entry[2] == {'preserve_partials': True} for entry in calls))
        self.assertCountEqual(released, scratch + [outputs['gate'], outputs['up'], hidden])
        self.assertFalse(any(value is source or value is result for value in released))

    def test_missing_projection_rejected_before_execution(self):
        with self.assertRaises(ValueError):
            dram_mlp.execute(None, None, {}, {}, None, None)

    def test_collective_can_take_ownership_without_retaining_partial(self):
        source = SimpleNamespace(shape=(1, 1, 16, 5120), dtype='bf16')
        operations = SimpleNamespace(bfloat16='bf16', L1_MEMORY_CONFIG='l1',
            mul=lambda *args, **kwargs: object())
        outputs = {name: object() for name in ('gate', 'up', 'down')}
        released = []
        def project(ops, value, weight, config, compute, retain, **kwargs):
            return retain(outputs[weight])
        with patch.object(dram_mlp, 'project', side_effect=project), patch.object(
                dram_mlp, 'release_owned', side_effect=lambda ops, values: released.extend(values)):
            partial = dram_mlp.execute(operations, source, {name: name for name in outputs},
                {name: {} for name in outputs}, None, lambda value: value)
        self.assertIs(partial, outputs['down'])
        self.assertFalse(any(value is partial for value in released))
        released.append(partial)
        self.assertEqual(sum(value is partial for value in released), 1)

    def test_failed_projection_releases_existing_outputs_and_scratch(self):
        source = SimpleNamespace(shape=(1, 1, 16, 5120), dtype='bf16')
        operations = SimpleNamespace(bfloat16='bf16')
        allocated, released = [], []
        def project(ops, value, weight, config, compute, retain, **kwargs):
            tensor = object()
            allocated.append(tensor)
            retain(tensor)
            if weight == 'up':
                raise RuntimeError('projection failed')
            return tensor
        names = ('gate', 'up', 'down')
        with patch.object(dram_mlp, 'project', side_effect=project), patch.object(
                dram_mlp, 'release_owned', side_effect=lambda ops, values: released.extend(values)):
            with self.assertRaisesRegex(RuntimeError, 'projection failed'):
                dram_mlp.execute(operations, source, {name: name for name in names},
                    {name: {} for name in names}, None, lambda value: None)
        self.assertCountEqual(released, allocated)
