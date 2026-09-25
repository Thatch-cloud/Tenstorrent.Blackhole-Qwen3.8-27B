import unittest
import hashlib
from pathlib import Path
import tempfile

from draft_kv_slide import geometry, row_source
from draft_kv_slide_report import validate


class SlidingHistoryTests(unittest.TestCase):
    def test_every_prefix_at_tile_and_capacity_boundaries(self):
        for history in (1, 15, 16, 17, 31, 32, 33, 255, 256, 2016, 2031, 2047, 2048):
            for prefix in range(1, 33):
                active = [('active', row) for row in range(history)]
                delta = [('delta', row) for row in range(prefix)]
                expected = (active + delta)[-2048:]
                expected += [('zero', 0)] * (2048 - len(expected))
                with self.subTest(history=history, prefix=prefix):
                    self.assertEqual([row_source(history, prefix, row) for row in range(2048)], expected)

    def test_full_window_drops_only_the_accepted_prefix(self):
        self.assertEqual(geometry(2048, 16), dict(history_rows=2048, prefix=16, rows=2048, drop=16))
        self.assertEqual(row_source(2048, 16, 0), ('active', 16))
        self.assertEqual(row_source(2048, 16, 2032), ('delta', 0))

    def test_invalid_geometry_and_rows(self):
        for history, prefix in ((0, 1), (2049, 1), (2048, 0), (2048, 33), (True, 1), (1, 1.0)):
            with self.assertRaises(ValueError):
                geometry(history, prefix)
        for row in (-1, 2048, True):
            with self.assertRaises(ValueError):
                row_source(2048, 16, row)

    def test_report_rejects_missing_duplicate_and_changed_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ('history-append-probe.py', 'draft-kv-slide-probe.py', 'draft_kv_slide.py', 'draft_kv_slide.cpp')
            for name in names:
                (root / name).write_bytes(b'fixture')
            report = dict(passed=True, closed_cleanly=True, backend='simulator',
                performance_qualified=False, model_integrated=False,
                sources={name: hashlib.sha256(b'fixture').hexdigest() for name in names if name != 'draft-kv-slide-probe.py'},
                checks=[dict(history=history, prefix=prefix, ordinal=ordinal, name=name, chip=chip, exact=True)
                    for history, prefix in ((31, 2), (2047, 2), (2048, 1), (2048, 16), (2048, 32))
                    for ordinal in (-1, 0, 1, 2)
                    for name in ('active_unchanged', 'delta_unchanged', 'sliding_output') for chip in (0, 1)])
            self.assertEqual(validate(report, root)['checks'], 120)
            removed = report['checks'].pop()
            with self.assertRaises(ValueError):
                validate(report, root)
            report['checks'].append(report['checks'][0])
            with self.assertRaises(ValueError):
                validate(report, root)
            report['checks'][-1] = removed
            (root / 'draft_kv_slide.cpp').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                validate(report, root)
