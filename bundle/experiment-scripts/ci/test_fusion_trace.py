from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from fusion_trace import validate_replays


class FusionTraceTests(unittest.TestCase):
    def run_gate(self, stale=False, timing=False):
        hidden = torch.ones((1, 1, 1, 5120), dtype=torch.bfloat16)
        inputs = SimpleNamespace(data=hidden.clone())
        callbacks, events = {}, []
        operations = SimpleNamespace(bfloat16='bf16', TILE_LAYOUT='tile', active=None,
            from_torch=lambda value, **kwargs: value.clone(), ReplicateTensorToMesh=lambda mesh: None,
            synchronize_device=Mock(), get_device_tensors=lambda value: [value, value], to_torch=lambda value: value.data,
            copy_host_to_device_tensor=lambda source, target: setattr(target, 'data', source.clone()),
            release_trace=lambda mesh, trace: events.append(('release', trace)),
            deallocate=lambda value: events.append(('deallocate', id(value))))
        def begin(mesh, **kwargs):
            operations.active = len(callbacks) + 1
            return operations.active
        def end(mesh, trace, **kwargs):
            operations.active = None
        def execute(mesh, trace, **kwargs):
            if not stale:
                callbacks[trace]()
        operations.begin_trace_capture = begin
        operations.end_trace_capture = end
        operations.execute_trace = execute
        def project(owned):
            output = SimpleNamespace(data=inputs.data * 2)
            owned.append(output)
            if operations.active is not None:
                callbacks[operations.active] = lambda: setattr(output, 'data', inputs.data * 2)
            return output
        with patch('fusion_trace.addresses', side_effect=lambda operations, value: id(value)), \
                patch('gdn_multitoken_conv.addresses', side_effect=lambda operations, value: id(value)):
            try:
                return validate_replays(operations, object(), inputs, hidden,
                    [hidden * 2, hidden * 2], project, project, (), timing=timing)
            finally:
                releases = [index for index, event in enumerate(events) if event[0] == 'release']
                self.assertEqual(len(releases), 2)
                self.assertEqual([event[0] for event in events[releases[0]:]],
                    ['release', 'release', 'deallocate', 'deallocate'])

    def test_changed_input_and_release_order(self):
        result = self.run_gate()
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['checks']), 12)
        self.assertEqual(len(result['negative_controls']), 4)

    def test_stale_trace_output_fails(self):
        with self.assertRaisesRegex(AssertionError, 'differs from native reference'):
            self.run_gate(stale=True)

    def test_paired_trace_timing_preserves_exactness(self):
        with patch('fusion_trace.time.perf_counter', side_effect=[value / 1000 for value in range(24)]):
            result = self.run_gate(timing=True)
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['timings']), 3)
        for block in result['timings']:
            self.assertAlmostEqual(block['control_ms'], 1)
            self.assertAlmostEqual(block['fused_ms'], 1)
