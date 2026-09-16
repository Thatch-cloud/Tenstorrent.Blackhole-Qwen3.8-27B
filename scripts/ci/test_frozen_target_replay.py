import subprocess
import unittest
import hashlib
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from frozen_recipe_context import REVISION
from frozen_target_replay import SOURCES, adapt_target_probe, validate_target_report


class TargetReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = subprocess.check_output(['git', 'show',
            f'{REVISION}:scripts/ci/target-t16-attention-8k-probe.py'], text=True)

    def test_selected_geometry_and_runtime_kv_preserve_replay_checks(self):
        changed = adapt_target_probe(self.original)
        self.assertIn("for capacity in (selected_geometry()['capacity'],):", changed)
        self.assertIn("kv_dtype='bfloat16'", changed)
        self.assertNotIn('ttnn.bfloat8_b', changed)
        self.assertIn("'frozen_context_geometry.py'", changed)
        start = '                original_cache = '
        self.assertEqual(self.original.split(start, 1)[1], changed.split(start, 1)[1])
        self.assertIn('starts = [first, first + 17, capacity - rows, first]', changed)

    def test_source_drift_and_reapplication_rejected(self):
        with self.assertRaises(ValueError):
            adapt_target_probe(self.original.replace('for capacity in (8448,):', 'for capacity in (4352,):'))
        with self.assertRaises(ValueError):
            adapt_target_probe(adapt_target_probe(self.original))

    def test_context_dtype_identity_and_complete_replay_required(self):
        with TemporaryDirectory() as directory:
            hashes = {}
            for name in SOURCES | {'frozen_context_geometry.py'}:
                payload = name.encode()
                (Path(directory) / name).write_bytes(payload)
                hashes[name] = hashlib.sha256(payload).hexdigest()
            checks = [dict(capacity=33024, start=start, ticket=ticket, chip=chip, exact=True)
                for ticket, start in enumerate((32768, 32785, 33008, 32768)) for chip in range(2)]
            report = dict(passed=True, closed=True, backend='simulator', context=32768,
                kv_dtype='bfloat16', sources=hashes, sources_after=dict(hashes),
                checks=checks, mask_checks=checks * 2,
                source_checks=[dict(capacity=33024, chip=chip, exact=True) for chip in (0, 1, 0, 1)],
                unpoisoned_replay=[dict(chip=chip, exact=True, nonfinite=0, mismatches=0) for chip in range(2)],
                stale_controls=2, mask_poison_controls=8)
            result = validate_target_report(report, directory, 32768)
            self.assertTrue(result['component_qualified'])
            self.assertFalse(result['performance_qualified'])
            for mutate in (
                    lambda value: value.update(context=8192),
                    lambda value: value.update(kv_dtype='bfloat8_b'),
                    lambda value: value.update(stale_controls=0),
                    lambda value: value['checks'].pop(),
                    lambda value: value['mask_checks'].pop(),
                    lambda value: value['source_checks'][0].update(exact=False),
                    lambda value: value['sources_after'].clear()):
                changed = deepcopy(report)
                mutate(changed)
                with self.assertRaises(ValueError):
                    validate_target_report(changed, directory, 32768)
            (Path(directory) / 'attention_replay.py').write_text('changed')
            with self.assertRaises(ValueError):
                validate_target_report(report, directory, 32768)


if __name__ == '__main__':
    unittest.main()
