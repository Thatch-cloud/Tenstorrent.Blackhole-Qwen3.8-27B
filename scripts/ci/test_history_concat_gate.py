import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from history_concat_gate import SOURCES, RUNTIME, qualify, validate
from history_concat_stage import stage


def fixture():
    root = Path(__file__).parent
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCES}
    checks = [dict(arm=arm, count=count, seed=seed, tensor=tensor, chip=chip, exact=True)
        for arm in ('original', 'candidate') for count in (65, 128) for seed in (0, 1)
        for tensor in ('output', 'borrowed') for chip in (0, 1)]
    return dict(passed=True, closed_cleanly=True, backend='simulator', performance_qualified=False,
        model_integrated=False, sources=sources, sources_after=dict(sources), checks=checks)


def evidence(root):
    raw = json.dumps(fixture()).encode()
    (root / 'history-concat.json').write_bytes(raw)
    (root / 'history-concat.exit-status').write_text('0')
    (root / 'simulator-runtime.txt').write_text(RUNTIME)
    return hashlib.sha256(raw).hexdigest()


class ConcatGateTests(unittest.TestCase):
    def test_complete_matrix_and_incomplete_results(self):
        self.assertEqual(len(validate(fixture())['checks']), 32)
        for change in ('missing', 'duplicate', 'wrong', 'source', 'failed', 'hardware'):
            report = copy.deepcopy(fixture())
            if change == 'missing':
                report['checks'].pop()
            elif change == 'duplicate':
                report['checks'][-1] = report['checks'][0]
            elif change == 'wrong':
                report['checks'][0]['exact'] = False
            elif change == 'source':
                report['sources_after']['history_concat_lifetime.py'] = '0' * 64
            elif change == 'failed':
                report['closed_cleanly'] = False
            else:
                report['backend'] = 'hardware'
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate(report)

    def test_report_exit_and_runtime_pins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checksum = evidence(root)
            qualify(Path(__file__).parent, root, checksum)
            with self.assertRaises(ValueError):
                qualify(Path(__file__).parent, root, '0' * 64)
            (root / 'history-concat.exit-status').write_text('124')
            with self.assertRaises(ValueError):
                qualify(Path(__file__).parent, root, checksum)

    def test_full_window_staging_checks_real_source_fingerprints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts = root / 'evidence'
            artifacts.mkdir()
            checksum = evidence(artifacts)
            scripts = root / 'checkout/scripts/ci'
            scripts.mkdir(parents=True)
            for name in ('dspark_history.py', 'gdn_multitoken_conv.py'):
                (scripts / name).write_bytes(Path(__file__).with_name(name).read_bytes())
            (scripts / 'frozen_ladder_cache_scope.py').write_text(
                'def run():\n    with tail_scope() as tail, page_geometry(context) as evidence, audit_scope() as reference_audit:\n        pass\n')
            (scripts / 'dspark_history.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Staged history source'):
                stage(root / 'checkout', artifacts, checksum, root / 'manifest.json')
            (scripts / 'dspark_history.py').write_bytes(Path(__file__).with_name('dspark_history.py').read_bytes())
            stage(root / 'checkout', artifacts, checksum, root / 'manifest.json')
            self.assertIn('with concat_scope(directory)', (scripts / 'frozen_ladder_cache_scope.py').read_text())
            self.assertFalse(json.loads((root / 'manifest.json').read_text())['performance_qualified'])
            with self.assertRaises(ValueError):
                stage(root / 'checkout', artifacts, checksum, root / 'manifest.json')


if __name__ == '__main__':
    unittest.main()
