import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from gdn_native_slot_gate import SOURCES, qualify


class NativeSlotGateTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        root = Path(__file__).parent
        names = set(source for sources in SOURCES.values() for source in sources)
        names.update(f'gdn-native-{name}-simulator{suffix}' for name in SOURCES for suffix in ('.json', '.exit-status'))
        for name in names:
            shutil.copyfile(root / name, self.directory / name)

    def test_retained_real_evidence_qualifies_as_synthetic_only(self):
        result = qualify(self.directory)
        self.assertEqual(set(result['reports']), {'projected', 'zero'})
        self.assertIn('learned hardware audit still required', result['scope'])

    def test_duplicate_failed_missing_or_stale_evidence_rejected(self):
        for name in SOURCES:
            path = self.directory / f'gdn-native-{name}-simulator.json'
            original = path.read_text()
            for mutation in ('duplicate', 'missing', 'failed', 'source'):
                report = json.loads(original)
                if mutation == 'duplicate':
                    report['checks'][-1] = report['checks'][0]
                elif mutation == 'missing':
                    report['checks'].pop()
                elif mutation == 'failed':
                    report['checks'][0]['exact'] = False
                else:
                    report['sources'].pop(next(iter(report['sources'])))
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    qualify(self.directory)
            path.write_text(original)

    def test_changed_adapter_rejected(self):
        (self.directory / 'gdn_native_slot_projected.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            qualify(self.directory)
