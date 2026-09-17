import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from frozen_draft_tail_gate import SOURCES, RUNTIME, qualify, validate


class DraftTailGateTests(unittest.TestCase):
    def fixture(self):
        checks = []
        for position in (64, 96):
            for proposals in (7, 15):
                for chip in (0, 1):
                    for name in ('reference_eager', 'candidate_eager', 'reference_replay',
                            'candidate_replay', 'history_unchanged', 'queries_unchanged'):
                        for seed in ((0,) if name.endswith('_eager') else (1, 0, 2)):
                            checks.append(dict(position=position, proposals=proposals, chip=chip,
                                name=name, seed=seed, exact=True))
        return dict(passed=True, closed_cleanly=True, backend='simulator', performance_qualified=False,
            model_integrated=False, checks=checks, sources={name: 'x' for name in SOURCES})

    def test_complete_matrix(self):
        report = self.fixture()
        self.assertEqual(len(report['checks']), 112)
        self.assertIs(validate(report), report)

    def test_missing_duplicate_failed_or_changed_check_rejected(self):
        original = self.fixture()
        for mutate in (lambda checks: checks.pop(), lambda checks: checks.append(checks[0]),
                lambda checks: checks[0].update(exact=False), lambda checks: checks[0].update(chip=True),
                lambda checks: checks[0].update(seed=17)):
            report = copy.deepcopy(original)
            mutate(report['checks'])
            with self.assertRaises(ValueError):
                validate(report)

    def test_pin_runtime_status_and_every_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = self.fixture()
            for name in SOURCES:
                (root / name).write_text(name)
                report['sources'][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
            raw = json.dumps(report).encode()
            (root / 'draft-tail.json').write_bytes(raw)
            (root / 'draft-tail.exit-status').write_text('0')
            (root / 'simulator-runtime.txt').write_text(RUNTIME)
            digest = hashlib.sha256(raw).hexdigest()
            qualify(root, root, digest)
            for name in SOURCES:
                before = (root / name).read_bytes()
                (root / name).write_bytes(before + b'changed')
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    qualify(root, root, digest)
                (root / name).write_bytes(before)
            for name, value in (('draft-tail.exit-status', '124'), ('simulator-runtime.txt', 'other')):
                before = (root / name).read_bytes()
                (root / name).write_text(value)
                with self.assertRaises(ValueError):
                    qualify(root, root, digest)
                (root / name).write_bytes(before)
            with self.assertRaises(ValueError):
                qualify(root, root, '0' * 64)

    def test_failed_or_misclassified_report_rejected(self):
        for field, value in (('passed', False), ('closed_cleanly', False), ('backend', 'hardware'),
                ('performance_qualified', True), ('model_integrated', True)):
            report = self.fixture()
            report[field] = value
            with self.assertRaises(ValueError):
                validate(report)
