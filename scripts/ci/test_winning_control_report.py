from copy import deepcopy
import unittest

from winning_control_report import ORIGINALS, compare


def fixture():
    request = dict(exact=True, state_exact=True, inactive_exact=True, prompt_tokens=[1] * 4096,
        max_new_tokens=256, committed_decode_tokens=1, decode_ms=10, prefill_ms=20,
        prompt_sha256='prompt', output_sha256='output', emitted=[2], proposed=1, accepted=0,
        native_attention_kernel={'hash': 'same'}, commit_only_gdn=True, target_attention_t16=True,
        norm_batch=True, capture_count=5,
        blocks=[dict(rows=16, source='dspark', position=4096, input_tokens=[1], accepted=0, committed=1)])
    requests = [dict(deepcopy(request), arm=arm, instrumented_timing=audit)
        for arm in ('control', 'publication') for audit in (True, False, False)]
    return dict(ctx_tokens=4096, drafter_history_rows=4096, proposal_rows=15, streams=1,
        source_revision=ORIGINALS[4096][0], target_index_sha256='index', target_config_sha256='config',
        parameter_sha256='parameter', target_cache_formats={'format': 'bf8'}, sampler_links=4,
        sources={'source': 'hash'}, sources_after={'source': 'hash'}, native_sources={'native': 'hash'},
        native_sources_after={'native': 'hash'}, comparison_axis='combined', passed=True,
        closed_cleanly=True, checkpoint_closed=True, stage='complete',
        device_parameter_checks=[dict(exact=True) for index in range(120)],
        request_checks=requests, committed_tg=100)


class WinningControlReportTests(unittest.TestCase):
    def test_recomputes_performance_instead_of_requiring_same_timing(self):
        original = fixture()
        replay = deepcopy(original)
        replay['committed_tg'] = 50
        for request in replay['request_checks']:
            request['decode_ms'] *= 2
        result = compare(original, replay)
        self.assertTrue(result['runtime_identity_and_outputs_match'])
        self.assertEqual(result['candidate_tg_change_percent'], -50)

    def test_rejects_changed_sources_state_proposals_and_reported_timing(self):
        original = fixture()
        mutations = (
            lambda value: value['sources'].update(source='changed'),
            lambda value: value['request_checks'][0].update(state_exact=False),
            lambda value: value['request_checks'][0]['blocks'][0].update(input_tokens=[2]),
            lambda value: value.update(committed_tg=200),
            lambda value: value['device_parameter_checks'][0].update(exact=False),
        )
        for mutate in mutations:
            replay = deepcopy(original)
            mutate(replay)
            with self.assertRaises(ValueError):
                compare(original, replay)


if __name__ == '__main__':
    unittest.main()
