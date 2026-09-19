import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from dspark_precision_gate import WEIGHTS, RUNTIME, QUERY_REPORT, candidate_manifest, qualify
from dspark_projection_precision_report import SOURCE_NAMES
from dspark_projection_precision_stage import PROJECTIONS
from test_dspark_projection_precision_report import fixture


class PrecisionGateTests(unittest.TestCase):
    def test_legacy_manifest_is_bound_to_query_and_exact_artifact(self):
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            raw = json.dumps(dict(component_execution_only=True, sources={})).encode()
            (folder / 'weight-pipeline-candidate.json').write_bytes(raw)
            with self.assertRaises(ValueError):
                candidate_manifest(folder, 'self_attn.q_proj.weight', QUERY_REPORT)
            with patch('dspark_precision_gate.QUERY_MANIFEST', hashlib.sha256(raw).hexdigest()):
                self.assertEqual(candidate_manifest(folder, 'self_attn.q_proj.weight', QUERY_REPORT)['projection'],
                    'self_attn.q_proj.weight')
                for projection, digest in (('mlp.up_proj.weight', QUERY_REPORT), ('self_attn.q_proj.weight', 'a' * 64)):
                    with self.assertRaises(ValueError):
                        candidate_manifest(folder, projection, digest)

    def setup_files(self, root):
        directory, evidence = root / 'sources', root / 'evidence'
        directory.mkdir()
        sources = {}
        for name in SOURCE_NAMES:
            raw = name.encode()
            (directory / name).write_bytes(raw)
            sources[name] = hashlib.sha256(raw).hexdigest()
        hashes = {}
        for projection in PROJECTIONS:
            folder = evidence / projection
            folder.mkdir(parents=True)
            report = fixture()
            report.update(projection=projection, sources=sources, sources_after=sources, weight_sha256=WEIGHTS[projection])
            raw = json.dumps(report).encode()
            (folder / 'dspark-projection-hifi2.json').write_bytes(raw)
            hashes[projection] = hashlib.sha256(raw).hexdigest()
            (folder / 'dspark-projection-hifi2.exit-status').write_text('0')
            (folder / 'simulator-runtime.txt').write_text(RUNTIME)
            (folder / 'container-cleanup.json').write_text(json.dumps(dict(
                stop_exit=0, logs_exit=0, copy_exit=0, remove_exit=0)))
            (folder / 'precision-candidate.json').write_text(json.dumps(dict(
                projection=projection, component_execution_only=True, sources=sources)))
        return directory, evidence, hashes

    def test_complete_reviewed_set_is_component_only(self):
        with TemporaryDirectory() as temporary:
            directory, evidence, hashes = self.setup_files(Path(temporary))
            result = qualify(directory, evidence, reviewed_reports=hashes)
            self.assertTrue(result['component_execution_passed'])
            self.assertFalse(result['hardware_qualified'])
            self.assertFalse(result['target_correctness_qualified'])
            self.assertIsNone(result['committed_tg'])

    def test_runtime_cleanup_policy_and_report_tampering_rejected(self):
        for filename, content in (('simulator-runtime.txt', 'other'),
                ('dspark-projection-hifi2.exit-status', '1'),
                ('container-cleanup.json', '{}'), ('precision-candidate.json', '{}'),
                ('dspark-projection-hifi2.json', '{}')):
            with TemporaryDirectory() as temporary:
                directory, evidence, hashes = self.setup_files(Path(temporary))
                (evidence / PROJECTIONS[-1] / filename).write_text(content)
                with self.assertRaises(ValueError):
                    qualify(directory, evidence, reviewed_reports=hashes)

    def test_live_source_drift_and_incomplete_reviews_rejected(self):
        with TemporaryDirectory() as temporary:
            directory, evidence, hashes = self.setup_files(Path(temporary))
            (directory / 'dspark_layer.py').write_text('changed')
            with self.assertRaises(ValueError):
                qualify(directory, evidence, reviewed_reports=hashes)
            hashes.pop(PROJECTIONS[-1])
            with self.assertRaises(ValueError):
                qualify(directory, evidence, reviewed_reports=hashes)
