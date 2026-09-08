from copy import deepcopy
import unittest

from full_request_pair import measure_requests, summarize_requests


class MatchedRequestTests(unittest.TestCase):
    def test_instrumented_diagnostics_cannot_enter_throughput_summary(self):
        for key in ('attention_audit', 'instrumented_timing'):
            with self.assertRaisesRegex(ValueError, 'not paired throughput'):
                summarize_requests([{key: True}])

    def test_short_attention_keeps_mtp_sampling_and_family_routing_fixed(self):
        def measure(*, mtp_short_attention):
            record = dict(self.record(True), mtp_short_attention=mtp_short_attention, native_sampling_rows=True,
                norm_batch=True, family_routing=True, short_context=True, attention_replay=mtp_short_attention,
                attention_mask_once=mtp_short_attention, replay_group_rows=4, lookup_max_rows=8,
                sampler_num_links=4, ended_with_eos=True, fabric_sources={'descriptor': 'audited'},
                selected_drafter='mtp', drafting_policy='neural-with-target-fallback', mtp_setup_ms=50,
                mtp=dict(max_drafts=7, head='native-full-vocabulary-force-argmax', native_sampling_rows=True,
                    mtp_weight_names=['mtp.fc.weight'], index_sha256='index', embedding_key='embedding',
                    prompt_alignment={'initialized_mtp_rows': 169}))
            record['blocks'][0]['source'] = 'mtp'
            return record
        records, summary = measure_requests(measure, arm_key='mtp_short_attention')
        self.assertTrue(summary['exact'])
        for key, value in (('family_routing', False), ('short_context', False), ('native_sampling_rows', False),
                ('attention_replay', False), ('attention_mask_once', False), ('lookup_max_rows', 32),
                ('replay_group_rows', 8), ('sampler_num_links', 1)):
            invalid = deepcopy(records)
            invalid[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                summarize_requests(invalid, arm_key='mtp_short_attention')

    def test_native_sampling_mtp_comparison_includes_setup_and_preserves_routing(self):
        def measure(*, native_sampling_rows):
            record = dict(self.record(native_sampling_rows), native_sampling_rows=native_sampling_rows,
                norm_batch=True, family_routing=False, attention_replay=False, attention_mask_once=False,
                replay_group_rows=4, lookup_max_rows=8, sampler_num_links=4, ended_with_eos=True,
                fabric_sources={'descriptor': 'audited'}, selected_drafter='mtp',
                drafting_policy='neural-with-target-fallback', mtp_setup_ms=50,
                mtp=dict(max_drafts=7, head='native-full-vocabulary-force-argmax',
                    native_sampling_rows=native_sampling_rows, mtp_weight_names=['mtp.fc.weight'],
                    index_sha256='index', embedding_key='embedding', prompt_alignment={'initialized_mtp_rows': 1}))
            record['blocks'][0]['source'] = 'mtp'
            return record
        records, summary = measure_requests(measure, arm_key='native_sampling_rows')
        self.assertEqual(summary['arms']['candidate']['mtp_setup_ms'], 100)
        self.assertEqual(summary['arms']['candidate']['prefill_setup_decode_ms'], 360)
        self.assertAlmostEqual(summary['arms']['candidate']['post_seed_including_setup_tokens_per_second'], 4000 / 300)
        for key, value in (('sampler_num_links', 1), ('lookup_max_rows', 32), ('selected_drafter', 'lookup'),
                ('accepted', 0), ('drafting_policy', 'lookup-first'), ('mtp_setup_ms', 0),
                ('mtp_setup_ms', float('nan')), ('ended_with_eos', False)):
            invalid = deepcopy(records)
            invalid[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                summarize_requests(invalid, arm_key='native_sampling_rows')
        for key, value in (('max_drafts', 15), ('native_sampling_rows', False), ('index_sha256', 'different'),
                           ('head', 'shortlisted'), ('prompt_alignment', {'initialized_mtp_rows': 0})):
            invalid = deepcopy(records)
            invalid[1]['mtp'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                summarize_requests(invalid, arm_key='native_sampling_rows')
        invalid = deepcopy(records)
        invalid[2]['blocks'][0]['input_tokens'] = [0, 9]
        with self.assertRaisesRegex(ValueError, 'changed proposal routing'):
            summarize_requests(invalid, arm_key='native_sampling_rows')

    def test_sampling_links_keeps_request_and_routing_identical(self):
        def measure(*, sampling_links):
            return dict(self.record(sampling_links), sampling_links=sampling_links, norm_batch=True,
                family_routing=False, attention_replay=False, attention_mask_once=False,
                replay_group_rows=4, lookup_max_rows=8, sampler_num_links=4 if sampling_links else 1,
                ended_with_eos=True, fabric_sources={'descriptor': 'audited'})
        records, summary = measure_requests(measure, arm_key='sampling_links')
        self.assertTrue(summary['exact'])
        for key, value in (('norm_batch', False), ('family_routing', True), ('lookup_max_rows', 32),
                ('sampler_num_links', 2), ('sampler_num_links', 4.0), ('ended_with_eos', False),
                ('fabric_sources', {}), ('fabric_sources', {'descriptor': 'changed'})):
            invalid = deepcopy(records)
            invalid[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                summarize_requests(invalid, arm_key='sampling_links')
        invalid = deepcopy(records)
        invalid[1]['blocks'][0]['input_tokens'] = [9]
        with self.assertRaises(ValueError):
            summarize_requests(invalid, arm_key='sampling_links')

    def test_lookup_cap_allows_only_between_arm_routing_changes(self):
        def measure(*, lookup_cap):
            record = dict(self.record(lookup_cap), lookup_cap=lookup_cap, norm_batch=True,
                family_routing=False, attention_replay=False, attention_mask_once=False,
                replay_group_rows=4, lookup_max_rows=8 if lookup_cap else 32)
            if lookup_cap:
                record.update(proposed=0, accepted=0, blocks=[dict(rows=1, source='target',
                    accepted=0, committed=1, match_length=0, position=position, input_tokens=[token])
                    for position, token in ((2, 0), (3, 1))])
            return record
        records, summary = measure_requests(measure, arm_key='lookup_cap')
        self.assertTrue(summary['exact'])
        for key, value in (('norm_batch', False), ('family_routing', True), ('lookup_max_rows', 32),
                ('lookup_max_rows', 8.0), ('proposed', 7), ('accepted', 1), ('emitted', [0, 2, 1])):
            invalid = deepcopy(records)
            invalid[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                summarize_requests(invalid, arm_key='lookup_cap')
        invalid = deepcopy(records)
        invalid[2]['blocks'][0]['input_tokens'] = [9]
        with self.assertRaisesRegex(ValueError, 'changed proposal routing'):
            summarize_requests(invalid, arm_key='lookup_cap')
    def test_width_comparison_keeps_replay_and_masks_fixed(self):
        def measure(*, attention_wide):
            return dict(self.record(attention_wide), attention_wide=attention_wide, norm_batch=True,
                family_routing=True, attention_replay=True, attention_mask_once=True,
                replay_group_rows=8 if attention_wide else 4)

        records, summary = measure_requests(measure, arm_key='attention_wide')
        self.assertEqual([entry['replay_group_rows'] for entry in records], [4, 8, 8, 4])
        self.assertEqual(summary['arm_key'], 'attention_wide')
        self.assertEqual(summary['decode_speedup'], 1.25)
        for key, value in (('norm_batch', False), ('family_routing', False), ('attention_replay', False),
                           ('attention_mask_once', False), ('replay_group_rows', 4), ('replay_group_rows', 8.0)):
            invalid = deepcopy(records)
            invalid[1][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                summarize_requests(invalid, arm_key='attention_wide')
        invalid = deepcopy(records)
        invalid[0]['replay_group_rows'] = 8
        with self.assertRaises(ValueError):
            summarize_requests(invalid, arm_key='attention_wide')

    def test_attention_comparison_keeps_norm_and_family_routing_fixed(self):
        def measure(*, attention_replay):
            return dict(self.record(attention_replay), norm_batch=True, attention_replay=attention_replay, family_routing=True)

        records, summary = measure_requests(measure, arm_key='attention_replay')
        self.assertEqual([entry['attention_replay'] for entry in records], [False, True, True, False])
        self.assertEqual(summary['arm_key'], 'attention_replay')
        self.assertEqual(summary['decode_speedup'], 1.25)
        for key in ('norm_batch', 'family_routing'):
            invalid = deepcopy(records)
            invalid[1][key] = False
            with self.assertRaisesRegex(ValueError, 'identical norm batching'):
                summarize_requests(invalid, arm_key='attention_replay')

    def record(self, enabled):
        return dict(norm_batch=enabled, prompt_tokens=[8, 9], emitted=[0, 1, 2], max_new_tokens=3,
            eos_ids=[], vocab_size=10, committed_decode_tokens=2, proposed=1, accepted=1,
            exact=True, state_exact=True, inactive_exact=True, decode_ms=80 if enabled else 100,
            engine_setup_ms=20, prefill_ms=30, setup_amortized=False, cross_request_trace_reuse=False,
            blocks=[dict(rows=2, source='lookup', accepted=1, committed=2, match_length=1,
                         position=2, input_tokens=[0, 1])])

    def test_actual_abba_order_and_separate_setup_accounting(self):
        observed = []

        def measure(*, norm_batch):
            observed.append(norm_batch)
            return self.record(norm_batch)

        requests, summary = measure_requests(measure)
        self.assertEqual(observed, [False, True, True, False])
        self.assertEqual(len(requests), 4)
        self.assertEqual(summary['decode_speedup'], 1.25)
        self.assertEqual(summary['arms']['candidate']['committed_tokens_per_second'], 25)
        self.assertEqual(summary['arms']['candidate']['post_seed_including_setup_tokens_per_second'], 20)
        self.assertEqual(summary['arms']['candidate']['prefill_setup_decode_ms'], 260)

    def test_failure_does_not_start_later_requests(self):
        observed = []

        def measure(*, norm_batch):
            observed.append(norm_batch)
            if norm_batch:
                raise RuntimeError('request failed after cleanup')
            return self.record(norm_batch)

        with self.assertRaises(RuntimeError):
            measure_requests(measure)
        self.assertEqual(observed, [False, True])

    def test_incomplete_nonfinite_or_different_work_is_rejected(self):
        baseline = [self.record(enabled) for enabled in (False, True, True, False)]
        for key, value in (('exact', False), ('state_exact', False), ('inactive_exact', False),
                           ('norm_batch', 1), ('decode_ms', 0), ('decode_ms', float('nan')),
                           ('engine_setup_ms', -1), ('committed_decode_tokens', 0),
                           ('emitted', [0, 1, 3]), ('proposed', 2), ('setup_amortized', True),
                           ('cross_request_trace_reuse', True)):
            records = deepcopy(baseline)
            records[1][key] = value
            with self.assertRaises(ValueError):
                summarize_requests(records)
        records = deepcopy(baseline)
        records[1]['blocks'][0]['input_tokens'] = [0, 3]
        with self.assertRaisesRegex(ValueError, 'routing'):
            summarize_requests(records)
        with self.assertRaises(ValueError):
            summarize_requests(baseline[:3])
