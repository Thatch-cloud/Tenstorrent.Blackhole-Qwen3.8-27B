import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from gdn_slot_copy_gate import SOURCES, validate


class SlotCopyGateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        for name in SOURCES:
            (self.root / name).write_text(name)
        hashes = {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest() for name in SOURCES}
        self.report = dict(passed=True, closed_cleanly=True, backend='simulator',
            performance_qualified=False, model_integrated=False, sources=hashes, sources_after=dict(hashes),
            checks=[dict(slot=slot, direction=direction, phase=phase, role=role, tensor=tensor, chip=chip, exact=True)
                    for slot in range(8) for direction in ('load', 'store')
                    for phase in ('eager', 'replay', 'changed-replay')
                    for role in ('source', 'destination') for tensor in range(5) for chip in range(2)])

    def test_complete_matrix_qualifies_only_simulator_component(self):
        result = validate(self.report, self.root, '0\n')
        self.assertEqual(result['checks'], 960)
        self.assertFalse(result['hardware_qualified'])
        self.assertFalse(result['performance_qualified'])

    def test_missing_duplicate_inexact_and_wrong_geometry_rejected(self):
        for mutation in ('missing', 'duplicate', 'inexact', 'chip', 'slot', 'phase', 'boolean'):
            report = copy.deepcopy(self.report)
            if mutation == 'missing':
                report['checks'].pop()
            elif mutation == 'duplicate':
                report['checks'][-1] = report['checks'][0]
            else:
                field, value = {'inexact': ('exact', False), 'chip': ('chip', 2),
                    'slot': ('slot', 8), 'phase': ('phase', 'capture'), 'boolean': ('slot', False)}[mutation]
                report['checks'][0][field] = value
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate(report, self.root, 0)

    def test_timeout_failed_cleanup_and_changed_source_rejected(self):
        with self.assertRaises(ValueError):
            validate(self.report, self.root, 124)
        for field in ('passed', 'closed_cleanly', 'sources_after'):
            report = copy.deepcopy(self.report)
            report[field] = None
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(report, self.root, 0)
        (self.root / SOURCES[0]).write_text('changed')
        with self.assertRaises(ValueError):
            validate(self.report, self.root, 0)


if __name__ == '__main__':
    unittest.main()
