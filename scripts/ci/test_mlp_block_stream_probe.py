from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

from mlp_block_stream_probe import adapt_probe, stream_fingerprints


class StreamProbeTests(unittest.TestCase):
    def test_probe_owns_stream_before_replay_and_preserves_checks(self):
        source = Path(__file__).with_name('fused-batch-probe.py').read_text()
        candidate = adapt_probe(source)
        self.assertLess(candidate.index('owned.append(device_stream)'), candidate.index('validate_replays('))
        self.assertLess(candidate.index('bind_stream(operation'), candidate.index('actual = operation(inputs)'))
        self.assertIn('(device_gate, device_up, device_packed, device_stream)', candidate)
        self.assertIn('stream_bytes_after', candidate)
        self.assertIn('compare_packed_weights(', candidate)
        self.assertIn('not (options.target_math and options.trace_t16', candidate)
        self.assertIn('and options.trace_replay and options.device_weight_check)', candidate)
        self.assertIn('for rows in (16,):', candidate)
        with self.assertRaises(ValueError):
            adapt_probe(candidate)

    def test_fingerprints_detect_every_byte_and_both_chips(self):
        import torch

        shards = [torch.arange(16, dtype=torch.int64).to(torch.uint32) for chip in range(2)]
        operations = SimpleNamespace(get_device_tensors=lambda stream: stream, to_torch=lambda shard: shard)
        before = stream_fingerprints(operations, shards)
        shards[1][-1] = 0xffffffff
        after = stream_fingerprints(operations, shards)
        self.assertEqual(before[0], after[0])
        self.assertNotEqual(before[1], after[1])
        with self.assertRaises(ValueError):
            stream_fingerprints(operations, shards[:1])

    def test_staging_requires_admitted_transport_and_frozen_sources(self):
        from mlp_block_stream_stage import stage
        from test_mlp_block_stream_admission import TransportAdmissionTests

        root = Path(__file__).parent
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            scripts = checkout / 'scripts/ci'
            scripts.mkdir(parents=True)
            originals = {name: (root / name).read_text() for name in ('fused_1d.py', 'fused-batch-probe.py')}
            for name, source in originals.items():
                (scripts / name).write_text(source)
            (scripts / 'simulator-suite.sh').write_text(
                'timeout -k 15 9000 python3 -u /experiment-scripts/ci/fused-batch-probe.py\n')
            report = checkout / 'transport.json'
            report.write_text(json.dumps(TransportAdmissionTests().fixture()))
            manifest = checkout / 'manifest.json'
            with patch('mlp_block_stream_stage.subprocess.check_output',
                    side_effect=lambda args: originals[args[-1].split('/')[-1]].encode()):
                stage(checkout, report, manifest)
                for path in scripts.glob('*.py'):
                    compile(path.read_text(), str(path), 'exec')
                evidence = json.loads(manifest.read_text())
                self.assertTrue(evidence['transport_admission']['transport_qualified'])
                self.assertFalse(evidence['simulator_qualified'])
                self.assertIn('validate_binding', (scripts / 'fused_1d.py').read_text())
                with self.assertRaises(ValueError):
                    stage(checkout, report, manifest)


if __name__ == '__main__':
    unittest.main()
