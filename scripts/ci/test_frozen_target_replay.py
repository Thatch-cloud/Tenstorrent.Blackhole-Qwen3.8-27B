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

    def test_bar_is_bit_exact_and_ulp_stats_are_the_diagnosis(self):
        """Run 35663000515 showed T16 and B1 agree EXACTLY at k_chunk_size 256, the
        value B1 itself picks, at all four start positions. The one-ulp disagreement run
        35662960713 measured came entirely from the 65536-only override to 128. Exactness
        is free, so the bar stays bit-exact; MAX_ULP only classifies a failure when one
        happens, and the stats print on every comparison so an exact run says max_ulp 0.0."""
        changed = adapt_target_probe(self.original)
        self.assertIn('MAX_ULP = 4.0', changed)
        self.assertIn("stage='t16-b1-ulp'", changed)
        for field in ('max_ulp=worst', 'mean_ulp=float(error.mean())', 'budget_ulp=MAX_ULP',
                      'nonfinite=nonfinite', 'rows_over_budget=int(rows.numel())'):
            self.assertIn(field, changed)
        # The failure condition is inequality and non-finites - not the budget.
        self.assertIn('if nonfinite or differing:', changed)
        self.assertNotIn('worst > MAX_ULP', changed)
        self.assertIn('MAX_ULP is NOT the pass condition', changed)
        self.assertNotIn('if not torch.equal(actual, expected)]', changed)
        # Shape and dtype stay exact - the budget is for values only.
        self.assertIn('if actual.shape != expected.shape or actual.dtype != expected.dtype:', changed)
        # The reference-reuse check compares INPUTS and must stay bit-exact.
        self.assertIn('torch.equal(ticket_query, queries[0])', changed)
        compile(changed, 'target-probe', 'exec')

    def _run_comparison(self, actual, expected, start=65536):
        """Execute the emitted comparison block itself against real tensors."""
        import json
        import torch
        opening = '                failures = []'
        changed = adapt_target_probe(self.original)
        body = opening + changed.split(opening, 1)[1].split('                if failures:', 1)[0]
        self.assertEqual(body.count(', strict=True'), 1)
        body = body.replace(', strict=True', '')   # 3.10 only; lengths equal by construction
        block = '\n'.join(line[16:] if line.startswith(' ' * 16) else line for line in body.split('\n'))
        printed = []
        namespace = dict(allocation_host=actual, gold=[expected], torch=torch, start=start,
            MAX_ULP=4.0, json=json, print=lambda value, flush=False: printed.append(value))
        exec(compile(block, 'comparison-block', 'exec'), namespace)
        return namespace['failures'], [json.loads(line) for line in printed]

    def test_bit_identical_tensors_pass_with_zero_ulp(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch not installed on this host')
        expected = [torch.full((4, 8), 0.005, dtype=torch.bfloat16)]
        failures, reports = self._run_comparison([expected[0].clone()], expected)
        self.assertEqual(failures, [])
        self.assertEqual(reports[0]['max_ulp'], 0.0)
        self.assertEqual(reports[0]['differing'], 0)

    def test_one_ulp_of_rounding_now_fails_but_is_diagnosed(self):
        """The k_chunk=128 case. It FAILS, because exactness is achievable at 256 and a
        tolerance would accept a regression for nothing - but the report still says it was
        one ulp with no rows over the structural threshold, which is what tells you at a
        glance that it is merge order and not a wrong row."""
        try:
            import torch
        except ImportError:
            self.skipTest('torch not installed on this host')
        expected = [torch.full((4, 8), 0.005, dtype=torch.bfloat16)]
        actual = expected[0].to(torch.float32)
        actual += 2.0 ** -15                      # exactly one ulp at this magnitude
        failures, reports = self._run_comparison([actual.to(torch.bfloat16)], expected)
        self.assertEqual(failures, [0])
        self.assertGreater(reports[0]['max_ulp'], 0.0)
        self.assertLessEqual(reports[0]['max_ulp'], 4.0)
        self.assertEqual(reports[0]['rows_over_budget'], 0)
        self.assertEqual(reports[0]['nonfinite'], 0)

    def test_near_zero_references_do_not_inflate_the_ulp_report(self):
        """Run 35665484092 reported max_ulp 91393 for a worst difference of 3.05e-05,
        because the per-element denominator collapsed on near-zero entries. The ulp is
        now taken at the reference TENSOR scale, so the number stays interpretable when
        most of the tensor is zero - which is the normal shape of attention output."""
        try:
            import torch
        except ImportError:
            self.skipTest('torch not installed on this host')
        expected = [torch.zeros(4, 8, dtype=torch.bfloat16)]
        expected[0][0][0] = 0.0028                  # one large entry, the rest zero
        actual = expected[0].to(torch.float32)
        actual[1][1] = 3.0517578125e-05             # a one-ulp-of-scale difference on a zero
        failures, reports = self._run_comparison([actual.to(torch.bfloat16)], expected)
        self.assertEqual(failures, [0])             # still fails: the bar is exactness
        report = reports[0]
        # 0.0028 rounds to this exactly in bfloat16; ulp at that scale is scale * 2**-7,
        # so the 3.05e-05 difference is about 1.4 ulp - the interpretable number.
        self.assertAlmostEqual(report['scale'], 0.0028076171875, places=9)
        self.assertAlmostEqual(report['max_ulp'], 1.39, places=1)
        self.assertLess(report['max_ulp'], 10.0)    # ~1.4, not 91393
        self.assertEqual(report['rows_over_budget'], 0)

    def test_a_structurally_wrong_row_still_fails(self):
        """Both kinds of failure are caught, and the report tells them apart: a wrong row
        puts rows_over_budget above zero, where merge-order rounding leaves it at zero."""
        try:
            import torch
        except ImportError:
            self.skipTest('torch not installed on this host')
        expected = [torch.full((4, 8), 0.005, dtype=torch.bfloat16)]
        actual = expected[0].clone()
        actual[2] = 0.05                          # one row an order of magnitude out
        failures, reports = self._run_comparison([actual], expected)
        self.assertEqual(failures, [0])
        self.assertGreater(reports[0]['max_ulp'], 4.0)
        self.assertEqual(reports[0]['rows_over_budget'], 1)
        self.assertEqual(reports[0]['first_rows'], [2])

    def test_non_finite_output_fails_even_within_budget(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch not installed on this host')
        expected = [torch.full((4, 8), 0.005, dtype=torch.bfloat16)]
        actual = expected[0].clone()
        actual[1][3] = float('nan')
        failures, reports = self._run_comparison([actual], expected)
        self.assertEqual(failures, [0])
        self.assertEqual(reports[0]['nonfinite'], 1)

    def test_shape_mismatch_is_refused_outright(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch not installed on this host')
        expected = [torch.full((4, 8), 0.005, dtype=torch.bfloat16)]
        with self.assertRaises(AssertionError):
            self._run_comparison([torch.full((4, 9), 0.005, dtype=torch.bfloat16)], expected)

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
