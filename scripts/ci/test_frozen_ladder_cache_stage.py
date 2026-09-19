import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from frozen_ladder_stage import wide_cache_payloads


class CacheStageTests(unittest.TestCase):
    def test_full_window_requires_its_own_qualified_sources(self):
        source = b'def run():\n    with sampler_links(sampler.tt_sampling, 4):\n        pass\n'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {}
            for name in ('frozen_context_geometry.py', 'ordered_cache.py', 'attention_batch.py'):
                (root / name).write_bytes(name.encode())
                sources[name] = hashlib.sha256(name.encode()).hexdigest()
            evidence = root / '261888'
            evidence.mkdir()
            for name in ('ladder-cache.json', 'ladder-cache.exit-status', 'simulator-runtime.txt'):
                (evidence / name).write_text('fixture')
            (root / 'reports.json').write_text(json.dumps({'261888': 'c' * 64}))
            with patch('frozen_ladder_cache_gate.qualify', return_value=dict(sources=sources)) as qualify:
                payloads = wide_cache_payloads(root, root, source, full_window=True)
                self.assertEqual(qualify.call_count, 1)
                self.assertEqual(qualify.call_args.args[1:], (evidence, 'c' * 64, 261888))
                self.assertIn('frozen-cache-evidence/261888/ladder-cache.json', payloads)
                (root / 'ordered_cache.py').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'Staged cache source differs'):
                    wide_cache_payloads(root, root, source, full_window=True)
            for digests in ({'131072': 'a' * 64}, {'262144': 'a' * 64}, {}):
                (root / 'reports.json').write_text(json.dumps(digests))
                with self.assertRaisesRegex(ValueError, 'Full-window simulator report'):
                    wide_cache_payloads(root, root, source, full_window=True)

    def test_both_reports_and_staged_sources_required(self):
        source = b'def run():\n    with sampler_links(sampler.tt_sampling, 4):\n        pass\n'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {}
            for name in ('frozen_context_geometry.py', 'ordered_cache.py', 'attention_batch.py'):
                (root / name).write_bytes(name.encode())
                sources[name] = hashlib.sha256(name.encode()).hexdigest()
            for context in (65536, 131072):
                (root / str(context)).mkdir()
                for name in ('ladder-cache.json', 'ladder-cache.exit-status', 'simulator-runtime.txt'):
                    (root / str(context) / name).write_text('fixture')
            (root / 'reports.json').write_text(json.dumps({'65536': 'a' * 64, '131072': 'b' * 64}))
            with patch('frozen_ladder_cache_gate.qualify', return_value=dict(sources=sources)) as qualify:
                payloads = wide_cache_payloads(root, root, source)
                self.assertEqual(qualify.call_count, 2)
                self.assertIn(b'with cache_scope(Path(__file__).parent)', payloads['dspark_request_experiment.py'])
                for context in (65536, 131072):
                    self.assertIn(f'frozen-cache-evidence/{context}/ladder-cache.json', payloads)
                (root / 'ordered_cache.py').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'Staged cache source differs'):
                    wide_cache_payloads(root, root, source)
            (root / 'reports.json').write_text(json.dumps({'65536': 'a' * 64}))
            with self.assertRaisesRegex(ValueError, 'Both long-context'):
                wide_cache_payloads(root, root, source)


if __name__ == '__main__':
    unittest.main()
