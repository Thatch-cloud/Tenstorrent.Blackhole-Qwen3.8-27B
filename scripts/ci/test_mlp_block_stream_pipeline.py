import json
from pathlib import Path
import tempfile
import unittest

from mlp_block_stream import reader_source
from mlp_block_stream_pipeline import transform, packet_ranges, SERIAL, PIPELINED
from mlp_block_stream_pipeline_stage import stage


class BulkPipelineTests(unittest.TestCase):
    def test_only_serial_bulk_reader_loop_changes(self):
        original = reader_source(Path(__file__).with_name('fused_1d_weights.cpp').read_text())
        candidate = transform(original)
        self.assertEqual(candidate.replace(PIPELINED, SERIAL), original)
        self.assertIn('cb_reserve_back(1, 2 * block_tiles)', candidate)
        self.assertIn('noc_async_read_barrier_with_trid(1 + block % 2)', candidate)
        with self.assertRaises(ValueError):
            transform(candidate)

    def test_packets_cover_exact_bulk_page_without_padding_reads(self):
        for burst in (1024, 2048, 4096, 8192):
            ranges = packet_ranges(burst)
            self.assertEqual(sum(length for offset, length in ranges), 27648)
            self.assertEqual([offset for offset, length in ranges], list(range(0, 27648, burst)))
            self.assertTrue(all(0 < length <= burst and length % 32 == 0 for offset, length in ranges))
        for invalid in (True, 0, -32, 33):
            with self.assertRaises(ValueError):
                packet_ranges(invalid)

    def test_staging_preserves_compute_projection_and_records_reader(self):
        directory = Path(__file__).parent
        with tempfile.TemporaryDirectory() as temporary:
            scripts = Path(temporary) / 'scripts/ci'
            scripts.mkdir(parents=True)
            for name in ('mlp_block_stream.py', 'fused_1d_weights.cpp', 'fused_1d.py'):
                (scripts / name).write_bytes((directory / name).read_bytes())
            projection = (scripts / 'fused_1d.py').read_bytes()
            manifest = Path(temporary) / 'manifest.json'
            stage(temporary, manifest)
            report = json.loads(manifest.read_text())
            self.assertFalse(report['arithmetic_changed'])
            self.assertFalse(report['hardware_qualified'])
            self.assertEqual(report['extra_buffer_bytes'], 0)
            self.assertNotEqual(report['serial_reader_sha256'], report['candidate_reader_sha256'])
            self.assertEqual((scripts / 'fused_1d.py').read_bytes(), projection)
            with self.assertRaises(ValueError):
                stage(temporary, manifest)
