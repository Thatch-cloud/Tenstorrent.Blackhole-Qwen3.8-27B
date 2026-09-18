import hashlib
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mlp_block_stream_runtime import require_hardware, scoped_block_stream


class HardwareScopeTests(unittest.TestCase):
    environment = dict(QWEN_MLP_BLOCK_STREAM_EXPERIMENT='1', QWEN_CARDS_ALLOCATED='1', QWEN_HARDWARE_TESTS='1')

    def test_explicit_physical_unprofiled_environment_required(self):
        require_hardware(self.environment)
        for name in self.environment:
            environment = dict(self.environment)
            environment.pop(name)
            with self.assertRaises(ValueError):
                require_hardware(environment)
        for name in ('TT_METAL_SIMULATOR', 'QWEN_SIM_ONLY', 'TT_METAL_DEVICE_PROFILER'):
            with self.assertRaises(ValueError):
                require_hardware(dict(self.environment, **{name: '1'}))

    def test_scope_binds_all_layers_and_restores_on_failure(self):
        self.check_scope(pipeline=False)

    def test_pipeline_scope_binds_all_layers_and_restores_on_failure(self):
        self.check_scope(pipeline=True)

    def test_pipeline_cannot_be_enabled_by_evidence_alone(self):
        with patch.dict(os.environ, self.environment, clear=True), self.assertRaisesRegex(ValueError, 'bulk-pipeline'):
            with scoped_block_stream('.', '.', runtime_root='.', operations=None,
                    weights=[], streams=[], pipeline_evidence='.'):
                self.fail('unadmitted pipeline entered')

    def check_scope(self, *, pipeline):
        source = '''class FusedProjection:
    def __init__(self, mesh, weights, **options):
        self.weights = weights
        self.compute = 'compute'
        self.manifest = {'baseline': True}
    def __call__(self, value):
        import ttnn
        validate_binding(self, ttnn)
        if 'reader_source' in globals():
            assert reader_source('original') == 'pipeline'
        return value
'''
        weights = [SimpleNamespace(address=layer) for layer in range(64)]
        streams = [SimpleNamespace(address=layer + 1000, dtype='uint32', layout='row',
            shape=(1, 1, 1820, 6912), memory_config=lambda: 'dram') for layer in range(64)]
        operations = SimpleNamespace(uint32='uint32', ROW_MAJOR_LAYOUT='row', DRAM_MEMORY_CONFIG='dram',
            get_device_tensors=lambda value: [SimpleNamespace(buffer_address=lambda: value.address)] * 2)
        original = SimpleNamespace(FusedProjection=object(), qualify_simulator=object())
        original_projection = original.FusedProjection
        baseline = dict(passed=True, kernels=[dict(baseline=True)])
        admission = dict(passed=True, kernels=[dict(fused_compute_sha256=hashlib.sha256(b'compute').hexdigest(),
            reader_sha256={'fused_1d_weights.cpp': hashlib.sha256(b'pipeline').hexdigest()})])
        environment = dict(self.environment, **({'QWEN_BULK_PIPELINE_EXPERIMENT': '1'} if pipeline else {}))
        with patch.dict(os.environ, environment, clear=True), \
                patch.dict('sys.modules', {'fused_t16_scope': original, 'ttnn': operations}), \
                patch('mlp_block_stream_runtime.qualify_register', return_value=baseline), \
                patch('mlp_block_stream_runtime.qualify', return_value=admission), \
                patch('mlp_block_stream_pipeline_gate.qualify', return_value=admission), \
                patch('mlp_block_stream_pipeline.transform', return_value='pipeline'), \
                patch('mlp_block_stream.reader_source', return_value='serial'), \
                patch('mlp_block_stream_runtime.Path.read_text', return_value=source), \
                patch('mlp_block_stream_runtime.adapt_projection', return_value=source), \
                patch('mlp_block_stream_runtime.CANDIDATE_SHA256', hashlib.sha256(source.encode()).hexdigest()):
            with self.assertRaisesRegex(RuntimeError, 'request failed'):
                with scoped_block_stream('.', '.', runtime_root='.', operations=operations,
                        weights=weights, streams=streams, pipeline_evidence='.' if pipeline else None) as audit:
                    projections = [original.FusedProjection(None, weight) for weight in weights]
                    for projection in projections:
                        self.assertEqual(projection('output'), 'output')
                    self.assertEqual(audit['constructions'], 64)
                    self.assertEqual(audit['calls'], 64)
                    self.assertIs(audit.get('bulk_pipeline', False), pipeline)
                    with self.assertRaises(ValueError):
                        original.FusedProjection(None, weights[0])
                    streams[0].address += 100
                    with self.assertRaises(ValueError):
                        projections[0]('output')
                    streams[0].address -= 100
                    raise RuntimeError('request failed')
            self.assertIs(original.FusedProjection, original_projection)
            self.assertTrue(audit['restored'])
            with self.assertRaises(ValueError):
                projections[0]('output')


if __name__ == '__main__':
    unittest.main()
