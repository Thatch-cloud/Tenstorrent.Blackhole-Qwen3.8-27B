from copy import deepcopy
import unittest

from dspark_draft_profile_report import attribute, validate
from dspark_intake import TAPS
from test_tensix_mlp_profile import device_rows


def fixture():
    positions = [4096, 4100, 4104]
    blocks = [dict(position=position, rows=16) for position in positions]
    records = []
    for index, position in enumerate([4096, 4096, *positions]):
        role = 'capture_warmup' if index == 0 else 'proposal'
        records.append(dict(block=index, position=position, rows=15, trace_id=23, trace_ordinal=index,
            first_replay=index == 0, role=role, instrumented_host_ms=1,
            label=f'qwen_draft_{index}_pos{position}_trace23_{role}'))
    request = dict(length=4096, lookup_max_rows=16, exact=True, state_exact=True, inactive_exact=True,
        instrumented_timing=True, commit_only_gdn=True, norm_batch=True, native_sampling_rows=True,
        committed_tokens_per_second=None, target_attention_t16=True, attention_replay=True, family_routing=True,
        blocks=blocks, gdn_verify_checks=[dict(position=position, rows=16, unchanged=True) for position in positions],
        dspark=dict(native_attention=True, proposal_trace=True, audit_features=True,
            proposal_checks=[dict(position=position, tensors=6, exact=True) for position in [4096, *positions]],
            feature_checks=[dict(position=position, tap=tap, chip=chip, exact=True)
                for position in positions for tap in TAPS for chip in range(2)]),
        draft_profile=dict(restored=True, records=records, trace_counts={'23': len(records)}))
    report = dict(passed=True, closed_cleanly=True, instrumented_timing=True, correctness_only=True,
        stage='complete', profile_family='dspark-draft', committed_tg=None, pp=None, eligible_for_serving=False,
        sources={'source': 'same'}, sources_after={'source': 'same'}, native_sources={'native': 'same'},
        native_sources_after={'native': 'same'}, request_checks=[request])
    console = '\n'.join(f'QWEN_DRAFT_PROFILE_BEGIN {record["label"]}\nQWEN_DRAFT_PROFILE_END {record["label"]}'
        for record in records)
    return report, console


class DraftReportTests(unittest.TestCase):
    def test_complete_trace_attribution_preserves_both_chips(self):
        report, console = fixture()
        request, records = validate(report, console)
        result = attribute(records, device_rows(records))
        self.assertEqual({entry['device'] for entry in result}, {'0', '1'})
        self.assertTrue(all(entry['steady_replays'] == 4 for entry in result))
        self.assertTrue(all(operation['samples'] == 4 for entry in result for operation in entry['captured_operations']))

    def test_missing_changed_or_promoted_evidence_rejected(self):
        original, console = fixture()
        for failure in ('count', 'order', 'position', 'trace', 'features', 'proposal', 'gdn', 'restored', 'timing', 'source'):
            report = deepcopy(original)
            request = report['request_checks'][0]
            records = request['draft_profile']['records']
            if failure == 'count':
                records.pop()
            elif failure == 'order':
                records[0], records[1] = records[1], records[0]
            elif failure == 'position':
                records[-1]['position'] += 1
            elif failure == 'trace':
                records[-1]['trace_id'] += 1
            elif failure == 'features':
                request['dspark']['feature_checks'][0]['exact'] = False
            elif failure == 'proposal':
                request['dspark']['proposal_checks'].pop()
            elif failure == 'gdn':
                request['gdn_verify_checks'].pop()
            elif failure == 'restored':
                request['draft_profile']['restored'] = False
            elif failure == 'timing':
                report['committed_tg'] = 200
            else:
                report['sources_after'] = {}
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                validate(report, console)
        for changed in (console.replace('QWEN_DRAFT_PROFILE_END', 'missing', 1),
                console.replace('QWEN_DRAFT_PROFILE_END', 'markers were dropped\nQWEN_DRAFT_PROFILE_END', 1)):
            with self.assertRaises(ValueError):
                validate(original, changed)

    def test_missing_device_or_operation_events_rejected(self):
        report, console = fixture()
        request, records = validate(report, console)
        for failure in ('missing', 'duplicate', 'chip', 'identity'):
            rows = device_rows(records)
            if failure == 'missing':
                rows.pop()
            elif failure == 'duplicate':
                rows.append(rows[-1])
            elif failure == 'chip':
                rows = [row for row in rows if row['DEVICE ID'] == '0']
            else:
                rows[-1]['GLOBAL CALL COUNT'] = '99999'
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                attribute(records, rows)


if __name__ == '__main__':
    unittest.main()
