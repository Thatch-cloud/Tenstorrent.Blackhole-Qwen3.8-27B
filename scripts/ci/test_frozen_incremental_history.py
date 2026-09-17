import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from history_append_hardware_gate import qualify
from incremental_history_scope import incremental_history
from test_incremental_history_scope import History, writer


class FrozenIncrementalHistoryTests(unittest.TestCase):
    def test_32k_full_banks_match_after_commits_and_discard(self):
        for position in (32767, 32768):
            with self.subTest(position=position):
                history, records = History(), []
                history.position, history.capacity = position, 33024
                expected = list(range(1, position + 1)) + [0] * (33024 - position)
                history.layers = tuple((expected.copy(), expected.copy()) for layer in range(5))
                history.spare_layers = tuple((expected.copy(), expected.copy()) for layer in range(5))
                module = SimpleNamespace(prepare=None)
                projection = SimpleNamespace(operations=history.operations, mesh=history.mesh)
                with incremental_history(History, module, records, writer_factory=writer):
                    for ordinal, (prefix, commit) in enumerate(((3, True), (32, False), (1, True), (32, True), (16, True))):
                        delta = [100000 + ordinal] * prefix + [-999] * (32 - prefix)
                        projection.project = lambda features, tables: tuple((delta, delta) for layer in range(5))
                        publication = module.prepare(history, projection, (), (), prefix, position=history.position)
                        candidate = expected.copy()
                        candidate[history.position:history.position + prefix] = delta[:prefix]
                        for pair in history.layers:
                            for bank in pair:
                                self.assertEqual(bank, expected)
                        for pair in publication.layers:
                            for bank in pair:
                                self.assertEqual(bank, candidate)
                        if commit:
                            history.commit_publication(publication)
                            expected = candidate
                        else:
                            history.discard_publication(publication)
                self.assertTrue(records[0]['restored'])
                self.assertFalse(records[0]['failed'])
                self.assertLessEqual(records[0]['max_touched_rows'], 96)

    @unittest.skipUnless(os.environ.get('QWEN_HISTORY_REPORT'), 'Retained physical writer report required')
    def test_existing_hardware_evidence_matches_current_writer(self):
        import json
        path = Path(os.environ['QWEN_HISTORY_REPORT'])
        report = json.loads(path.read_bytes())
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in report['sources']:
                (directory / name).write_bytes(subprocess.check_output(['git', 'show', f'HEAD:scripts/ci/{name}']))
            result = qualify(directory, path)
            self.assertTrue(result['passed'])
            self.assertEqual(len(result['checks']), 84)


if __name__ == '__main__':
    unittest.main()
