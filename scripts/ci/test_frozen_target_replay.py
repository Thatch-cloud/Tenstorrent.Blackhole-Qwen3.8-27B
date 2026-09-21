import subprocess
import unittest
import hashlib
import os
from unittest.mock import patch
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from frozen_recipe_context import REVISION
from frozen_target_replay import SOURCES, adapt_target_probe, adapt_target_mask, validate_target_report


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
        start = '                trace, output = capture_operation'
        self.assertEqual(self.original.split(start, 1)[1].split('    except BaseException', 1)[0],
            changed.split(start, 1)[1].split('    except BaseException', 1)[0])
        self.assertIn('starts = [first, first + 17, capacity - rows, first]', changed)
        self.assertLess(changed.index('allocation_check = reader('), changed.index('for start, ticket_query in zip('))
        self.assertIn('start == starts[0] and torch.equal(ticket_query, queries[0])', changed)
        self.assertIn('zip(allocation_host, gold[0], strict=True)', changed)
        self.assertNotIn('warm = reader(', changed)

    def test_failing_comparison_reports_magnitude_and_shape_before_raising(self):
        """The bare torch.equal told us only that T16 and B1 disagree, never by how
        much or on which rows, so a 2-ulp bf16 rounding difference and a wrong-rows
        structural bug looked identical in the artifact (docs/t16-vs-b1-65536.md).
        Every mismatching tensor now prints one JSON line first; the assertion that
        fails the probe is unchanged and still fires."""
        changed = adapt_target_probe(self.original)
        self.assertIn("stage='t16-b1-mismatch'", changed)
        for field in ('tensor=index', 'start=start', 'shape=list(actual.shape)',
                      'mismatching=int((actual != expected).sum())', 'max_abs=float(difference.max())',
                      'mean_abs=float(difference.mean())', 'rows_affected=int(rows.numel())',
                      'first_rows=[int(value) for value in rows[:8]]'):
            self.assertIn(field, changed)
        self.assertIn('nonfinite=int((~torch.isfinite(actual.to(torch.float32))).sum())', changed)
        # Diagnostics run first, then the same failure.
        self.assertLess(changed.index("stage='t16-b1-mismatch'"),
            changed.index("raise AssertionError('T16 long-context warm output differs from native B1')"))
        self.assertIn('if mismatched:', changed)
        compile(changed, 'target-probe', 'exec')

    def test_mismatch_report_runs_against_real_tensors(self):
        """Execute the emitted block itself, not just its text: a shape or dtype slip
        in the generated source would otherwise only surface on the rig."""
        import json
        try:
            import torch
        except ImportError:
            self.skipTest('torch not installed on this host')
        opening = '                mismatched = [index for index, (actual, expected)'
        changed = adapt_target_probe(self.original)
        body = opening + changed.split(opening, 1)[1].split('                if mismatched:', 1)[0]
        # zip(strict=) is 3.10, which the container has and this host does not; the
        # lengths are equal by construction here, so dropping it changes nothing the
        # block does. Exactly one occurrence, or the extraction moved.
        self.assertEqual(body.count(', strict=True'), 1)
        body = body.replace(', strict=True', '')
        block = '\n'.join(line[16:] if line.startswith(' ' * 16) else line for line in body.split('\n'))
        expected = [torch.zeros(4, 8, dtype=torch.bfloat16), torch.ones(2, 3, dtype=torch.bfloat16)]
        actual = [value.clone() for value in expected]
        actual[0][1][2] = 0.5
        printed = []
        namespace = dict(allocation_host=actual, gold=[expected], torch=torch, start=65536,
            json=json, print=lambda value, flush=False: printed.append(value))
        exec(compile(block, 'mismatch-block', 'exec'), namespace)
        self.assertEqual(namespace['mismatched'], [0])
        self.assertEqual(len(printed), 1)
        report = json.loads(printed[0])
        self.assertEqual(report['stage'], 't16-b1-mismatch')
        self.assertEqual((report['tensor'], report['start']), (0, 65536))
        self.assertEqual(report['shape'], [4, 8])
        self.assertEqual((report['mismatching'], report['elements']), (1, 32))
        self.assertEqual((report['max_abs'], report['nonfinite']), (0.5, 0))
        self.assertEqual((report['rows_affected'], report['first_rows']), (1, [1]))

    def test_source_drift_and_reapplication_rejected(self):
        with self.assertRaises(ValueError):
            adapt_target_probe(self.original.replace('for capacity in (8448,):', 'for capacity in (4352,):'))
        with self.assertRaises(ValueError):
            adapt_target_probe(adapt_target_probe(self.original))

    def test_selected_capacity_is_checked_before_reference_work(self):
        original = subprocess.check_output(['git', 'show',
            f'{REVISION}:scripts/ci/attention_mask_replay.py'], text=True)
        namespace = {}
        exec(adapt_target_mask(original), namespace)
        with patch.dict(os.environ, {'QWEN_DSPARK_REQUEST_CONTEXT': '32768'}):
            for start in (32768, 32785, 33008):
                namespace['validate_ticket'](start, 16, 33024)
            for start, capacity in ((32767, 33024), (33009, 33024), (65536, 65792)):
                with self.assertRaises(ValueError):
                    namespace['validate_ticket'](start, 16, capacity)
        changed = adapt_target_probe(self.original)
        self.assertLess(changed.index('validate_ticket(start, 16'), changed.index('    import torch'))

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
            with self.assertRaisesRegex(ValueError, 'variant differs'):
                validate_target_report(report, directory, 32768, compact_scratch=True)
            from sdpa_tree_scratch import HASHES, PATCHED_FACTORY_SHA256
            compact = deepcopy(report)
            native = dict(HASHES, **{'sdpa_decode_program_factory.cpp': PATCHED_FACTORY_SHA256})
            compact.update(compact_tree_scratch=True, native_sources=native,
                native_sources_after=dict(native), factory_build=dict(passed=True, import_passed=True,
                    binaries_after={name: 'a' * 64 for name in
                        ('build_Release/lib/_ttnncpp.so', 'build_Release/ttnn/_ttnncpp.so')}))
            self.assertTrue(validate_target_report(compact, directory, 32768,
                compact_scratch=True)['compact_tree_scratch'])
            compact['native_sources_after'].clear()
            with self.assertRaisesRegex(ValueError, 'native sources'):
                validate_target_report(compact, directory, 32768, compact_scratch=True)
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
