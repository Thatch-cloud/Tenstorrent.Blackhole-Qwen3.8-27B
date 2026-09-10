import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dspark_commit_gate as gate


class CommitGateTests(unittest.TestCase):
    def test_t8_evidence_cannot_enable_the_t16_arm(self):
        with patch.object(gate, 'SHA256', None), patch.object(gate, 'digest') as checksum:
            with self.assertRaisesRegex(ValueError, 'T16.*not yet'):
                gate.qualify('missing')
            checksum.assert_not_called()

    def test_complete_all_prefix_matrix_and_current_adapter_hashes_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = dict(backend='ttsim', rows=16, last_stage=dict(stage='complete'),
                native_hashes=gate.HASHES, handoff_runtime_hashes=gate.HANDOFF_HASHES, generated_hashes=gate.GENERATED,
                model_adapter_checks=17, continuation_checks=17, precommit_unchanged_checks=17, stale_controls=1,
                batched_adapter_sha256='source', model_adapter_sha256='source', adapter_sha256='source',
                commit_sources={'publication': 'source'}, commit_sources_after={'publication': 'source'},
                window_prefix_hashes=dict(py='source', cpp='source'), window_dma_hashes=dict(py='source', cpp='source'))
            for name in ('passed', 'norm_gate', 'convolution', 'batched_convolution', 'dma_windows', 'packed_checkpoints',
                    'continuation_enabled', 'compact_prologue', 'norm_batch_layer', 'deferred_conv_publication', 'commit_only_gdn'):
                report[name] = True
            path = root / gate.REPORT
            path.with_suffix('.exit-status').write_text('0\n')
            with patch.object(gate, 'SHA256', 'report'), patch.object(gate, 'source_hashes', return_value={'publication': 'source'}), patch.object(gate, 'digest',
                    side_effect=lambda value: 'report' if value.name == gate.REPORT else 'source'):
                path.write_text(json.dumps(report))
                self.assertEqual(gate.qualify(root), {gate.REPORT: 'report'})
                for name, changed in (('rows', 8), ('continuation_checks', 16), ('precommit_unchanged_checks', 16),
                        ('stale_controls', 0), ('commit_only_gdn', False), ('adapter_sha256', 'changed'),
                        ('commit_sources_after', {'publication': 'changed'})):
                    path.write_text(json.dumps({**report, name: changed}))
                    with self.subTest(name=name), self.assertRaises(ValueError):
                        gate.qualify(root)


if __name__ == '__main__':
    unittest.main()
