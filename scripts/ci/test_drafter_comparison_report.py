import copy
import unittest
from unittest.mock import patch

from drafter_comparison_report import SCHEDULE, summarize
from shared_qk_norm_scatter_gate import REPORT_SHA256 as NORM_SHA256
from gdn_direct_window_gate import REPORT_SHA256 as WINDOW_SHA256
from mlp_down_grid_gate import REPORT_SHA256 as DOWN_SHA256


class ComparisonReportTests(unittest.TestCase):
    def records(self):
        records = []
        for drafter, audit in SCHEDULE:
            records.append(dict(comparison_drafter=drafter, selected_drafter=drafter,
                instrumented_timing=audit, prompt_tokens=[10] * 4096, emitted=[20, 21],
                max_new_tokens=256, eos_ids=[21], vocab_size=248320, committed_decode_tokens=1,
                exact=True, state_exact=True, inactive_exact=True, commit_only_gdn=True,
                sampler_num_links=4, decode_ms=10, prefill_ms=100,
                blocks=[dict(rows=16, source=drafter, accepted=0, position=4096,
                             input_tokens=[20] + [22 if drafter == 'dspark' else 23] * 15, committed=1)],
                fused_t16_mlp=dict(hits=[1] * 64),
                gdn_shared_qk=dict(restored=True, released=True, admission=dict(report_sha256=NORM_SHA256),
                                  loads=[dict(rows=16, programs=3, retained_preparation_buffers=2)] * 48),
                norm_reader=dict(policy='scatter', restored=True, builds=48, report_sha256=NORM_SHA256),
                gdn_direct_window=dict(direct=True, restored=True, hits=48, report_sha256=WINDOW_SHA256),
                mlp_down_grid=dict(wider_down=True, restored=True, hits=[1] * 64, report_sha256=DOWN_SHA256)))
        return records

    def check(self, records):
        with patch('dspark_request_experiment.summarize', return_value=dict(committed_tg=100)) as dspark, \
                patch('full_dflash_request.summarize_dflash_requests',
                      return_value=dict(committed_tokens_per_second=110)) as dflash, \
                patch('cumulative_fusion_validation.validate_fusion_policy') as fusion:
            result = summarize(records)
            self.assertEqual(dspark.call_args.args[0], [records[index] for index in (0, 2, 5)])
            self.assertEqual(dflash.call_args.args[0], [records[index] for index in (1, 3, 4)])
            self.assertEqual(fusion.call_count, 6)
            return result

    def test_different_draft_paths_with_identical_target_outputs_allowed(self):
        result = self.check(self.records())
        self.assertAlmostEqual(result['committed_tg_change_percent'], 10)
        self.assertFalse(result['performance_promoted'])

    def test_workload_state_and_target_component_drift_rejected(self):
        changes = [('emitted', [20, 24]), ('state_exact', False), ('selected_drafter', 'dspark'),
                   ('sampler_num_links', 1), ('decode_ms', float('nan')),
                   ('norm_reader', {}), ('gdn_direct_window', {}), ('mlp_down_grid', {})]
        for name, value in changes:
            with self.subTest(name=name):
                records = self.records()
                records[3][name] = value
                with self.assertRaises(ValueError):
                    self.check(records)

    def test_same_drafter_must_repeat_audited_proposal(self):
        records = self.records()
        records[3]['blocks'][0]['input_tokens'][1] = 24
        with self.assertRaisesRegex(ValueError, 'audited proposals'):
            self.check(records)

    def test_missing_or_reordered_requests_rejected(self):
        for records in (self.records()[:-1], list(reversed(self.records()))):
            with self.assertRaises(ValueError):
                self.check(records)
