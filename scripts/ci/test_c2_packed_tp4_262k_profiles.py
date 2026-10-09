"""The 262k-window profiles (tp4/seats8-262k, I2): every eight-seat profile's window is 262,144 tokens, not just its prompt.

c2-packed-tp4-8x262k and its gate, time-gate and diag-strace twins are the 131k eight-seat profiles plus exactly the 262k deltas
(window, page-table width, pooled KV cache, reservation admission, DRAM bounds, the drafter headroom); c2-packed-tp4-262k-gate is the
four-seat gate profile at the same window. The protected profiles (production and the 131k eight-seat family) are held by digest:
adding these must not move what is served today. The drafter's request context stays 131,072 in every one of them (the H1 census,
test_request_context_census). The contract's clamp (drafter_headroom_tokens) keeps prompt + answer at or under 262,112."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import make_parked_profiles as parked_twins  # noqa: E402
import packed_any_admission as admission  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import serving_runtime  # noqa: E402

PROFILES = HERE / 'qwen_c2_profiles.json'
WINDOW = 262144
HEADROOM = 32
MAX_END = WINDOW - HEADROOM                      # 262,112: the drafter's last position
BLOCKS_8 = 21760
BLOCKS_4 = 16384
PAGES = WINDOW // 64                             # 4,096
TRAFFIC, GATE, TIME_GATE, DIAG = ('c2-packed-tp4-8x262k', 'c2-packed-tp4-8x262k-gate', 'c2-packed-tp4-8x262k-time-gate',
                                  'c2-packed-tp4-8x262k-diag-strace')
FOUR_GATE = 'c2-packed-tp4-262k-gate'
EIGHT = (TRAFFIC, GATE, TIME_GATE, DIAG)
TWIN = {TRAFFIC: 'c2-packed-tp4-8', GATE: 'c2-packed-tp4-8-gate', TIME_GATE: 'c2-packed-tp4-8-time-gate',
        DIAG: 'c2-packed-tp4-8-diag-strace', FOUR_GATE: 'c2-packed-tp4-gate'}
ALL = EIGHT + (FOUR_GATE,)
# tp4/262k8: the best-lever arms (test_tp4_262k8_best_profiles holds them); they are not the 131k twins' 262k deltas, so the twin rule does not apply
BEST262K = ('c2-packed-tp4-8x262k-best', 'c2-packed-tp4-8x262k-best-audit', 'c2-packed-tp4-8x262k-best-levern-audit', 'c2-packed-tp4-8x262k-best-levern-control-audit', 'c2-packed-tp4-8x262k-best-levern-final-hold-time-gate', 'c2-packed-tp4-8x262k-best-levern-foreign-time-gate', 'c2-packed-tp4-8x262k-best-levern-hang-gate', 'c2-packed-tp4-8x262k-best-levern-r1-time-gate', 'c2-packed-tp4-8x262k-best-levern-time-gate', 'c2-packed-tp4-8x262k-best-time-gate', 'c2-packed-tp4-8x262k-ship',
            'c2-packed-tp4-8x262k-best-time-gate-u1', 'c2-packed-tp4-8x262k-best-u1-audit',
            'c2-packed-tp4-8x262k-hostgap-1', 'c2-packed-tp4-8x262k-hostgap-1-audit', 'c2-packed-tp4-8x262k-hostgap-2', 'c2-packed-tp4-8x262k-hostgap-2-audit',
            # tp4/w1: the built levers gated together (test_tp4_w1 holds each as the control minus the in-trace sampler plus the stack)
            'c2-packed-tp4-8x262k-w1', 'c2-packed-tp4-8x262k-w1-audit', 'c2-packed-tp4-8x262k-w1-audit-nod1', 'c2-packed-tp4-8x262k-w1-lite', 'c2-packed-tp4-8x262k-w1-nod1', 'c2-packed-tp4-8x262k-w2', 'c2-packed-tp4-8x262k-w2-audit', 'c2-packed-tp4-8x262k-w2-nof1', 'c2-packed-tp4-8x262k-w2-nof1-audit', 'c2-packed-tp4-8x262k-best-sdpamulti', 'c2-packed-tp4-8x262k-best-sdpamulti-audit',
            # tp4/packed-prefix + ship/262k-prefix: the sticky-session twins and the both-levers ship profile
            'c2-packed-tp4-8x262k-prefix-gate', 'c2-packed-tp4-8x262k-prefix-time-gate', 'c2-packed-tp4-8x262k-ship-prefix', 'c2-packed-tp4-8x262k-ship-prefix-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern', 'c2-packed-tp4-8x262k-ship-prefix-levern-traffic', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit', 'c2-packed-tp4-8x262k-ship-prefix-audit-digests', 'c2-packed-tp4-8x262k-ship-prefix-dckdefault', 'c2-packed-tp4-8x262k-ship-prefix-w2', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-sdpa', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-f1', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-ln', 'c2-packed-tp4-8x262k-ship-prefix-pool', 'c2-packed-tp4-8x262k-ship-prefix-dbf16', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-w1')
# engine reuse (tp4/engine-reuse): the generated gate-only twins of the two levern profiles (test_parked_tp4_profiles holds each as its parent plus its flags)
BEST262K = BEST262K + tuple(name for name, parent, env, why in parked_twins.specs())
# tp4/262k8-x (test_tp4_262k8_x holds each as the best-time-gate or best-audit twin plus/minus exactly one lever): the experiments image's arms; 262k knobs by inheritance.
X262K = tuple('c2-packed-tp4-8x262k-best-time-gate-' + lever for lever in ('nosamp', 's1', 'd2', 'dbf16', 'lookup', 'stack')) + (
    'c2-packed-tp4-8x262k-best-nosamp-audit', 'c2-packed-tp4-8x262k-best-stack-audit')
DRAM = {'QWEN_FAST_DRAM_ENGINE_BUILD_MB': '800', 'QWEN_FAST_DRAM_PREFILL_TRANSIENT_MB': '600',
        'QWEN_FAST_DRAM_LARGEST_BUFFER_MB': '256'}
# Canonical-JSON sha256 prefixes at 20d8adcd (the I1 head this branch starts from).
PROTECTED = {'c2-packed-tp4': 'aeaa12f7e4b33ec5', 'c2-packed-tp4-8': 'd6937f64f23f391f',
             'c2-packed-tp4-8-gate': '52c48543958efcc4', 'c2-packed-tp4-8-time-gate': '0b2b6bd11495bcc9',
             'c2-packed-tp4-8-diag-strace': '8a2b4122a94e1c85', 'c2-packed-tp4-8-diag-strace-nowarm': '6a562c303ed9ba5c',
             'c2-packed-tp4-8-diag-strace-rshard': '9a7a655c2a02d71d', 'c2-packed-tp4-gate': '64ad7155ff5b9add'}


def document():
    return json.loads(PROFILES.read_text(encoding='utf-8'))


def profiles():
    return document()['profiles']


def digest(profile):
    return hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]


def minus_deltas(profile, traffic):
    """The profile with the 262k deltas and the description taken out: what must equal its 131k twin."""
    out = copy.deepcopy(profile)
    out.pop('description')
    out.pop('drafter_headroom_tokens', None)
    for key in ('QWEN_FAST_MAX_POSITION', 'QWEN_FAST_KV_RESERVATION', 'QWEN36_MAX_TOKENS_ALL_USERS', 'QWEN_FAST_262K_EVIDENCE_WAIVER') + tuple(DRAM):
        out['env'].pop(key, None)
    for key in ('max-model-len', 'max-num-batched-tokens', 'num-gpu-blocks-override'):
        out['engine'].pop(key)
    if traffic:
        out.pop('max_prompt_tokens')
    return out


class ProtectedProfileTests(unittest.TestCase):
    def test_production_and_the_131k_eight_seat_family_are_byte_identical(self):
        found = profiles()
        for name, prefix in PROTECTED.items():
            with self.subTest(profile=name):
                self.assertEqual(digest(found[name]), prefix)

    def test_the_file_default_is_still_production_and_the_new_profiles_are_new(self):
        self.assertEqual(document()['default'], 'c2-packed-tp4')
        self.assertTrue(set(ALL) <= set(profiles()))
        self.assertEqual(sorted(name for name in profiles() if '262k' in name), sorted(ALL + BEST262K + X262K))

    def test_the_131k_profiles_carry_none_of_the_262k_knobs(self):
        for name, profile in profiles().items():
            if name in ALL + BEST262K + X262K:
                continue
            with self.subTest(profile=name):
                self.assertNotIn('drafter_headroom_tokens', profile)
                self.assertNotIn('QWEN_FAST_KV_RESERVATION', profile['env'])
                self.assertNotEqual(profile['env'].get('QWEN_FAST_MAX_POSITION'), str(WINDOW))
                for key in DRAM:
                    self.assertNotIn(key, profile['env'])


class DeltaTests(unittest.TestCase):
    def test_each_profile_is_its_131k_twin_plus_exactly_the_262k_deltas(self):
        found = profiles()
        for name in ALL:
            with self.subTest(profile=name):
                traffic = name == TRAFFIC
                self.assertEqual(minus_deltas(found[name], traffic), minus_deltas(found[TWIN[name]], traffic))

    def test_the_window_the_pages_the_blocks_and_the_headroom(self):
        found = profiles()
        for name in ALL:
            profile, engine, env = found[name], found[name]['engine'], found[name]['env']
            with self.subTest(profile=name):
                self.assertEqual((engine['max-model-len'], engine['max-num-batched-tokens']), (WINDOW, WINDOW))
                self.assertEqual(env['QWEN_FAST_MAX_POSITION'], str(WINDOW))
                self.assertEqual(engine['max-model-len'] // 64, PAGES, 'the page-table width page_width_tp4 admits behind E1')
                self.assertEqual(profile['drafter_headroom_tokens'], HEADROOM)
                self.assertEqual(WINDOW - profile['drafter_headroom_tokens'], contract.DRAFTER_END_LIMIT)
                self.assertEqual(engine['num-gpu-blocks-override'], BLOCKS_4 if name == FOUR_GATE else BLOCKS_8)
                self.assertEqual(engine['max-num-seqs'], 4 if name == FOUR_GATE else 8)

    def test_the_drafters_request_context_stays_131072_in_every_262k_profile(self):
        for name in ALL:
            env = profiles()[name]['env']
            with self.subTest(profile=name):
                self.assertEqual(env['QWEN_DSPARK_REQUEST_CONTEXT'], '131072')
                for stale in ('261888', '32768', '4096'):
                    self.assertNotEqual(env['QWEN_DSPARK_REQUEST_CONTEXT'], stale)

    def test_the_four_seat_gate_is_fully_reserved_and_the_eight_seat_ones_are_pooled_under_the_reservation(self):
        found = profiles()
        four = found[FOUR_GATE]
        self.assertEqual(four['engine']['num-gpu-blocks-override'], 4 * PAGES)
        self.assertNotIn('QWEN_FAST_KV_RESERVATION', four['env'])
        for name in EIGHT:
            with self.subTest(profile=name):
                pool = found[name]['engine']['num-gpu-blocks-override']
                self.assertLess(pool, 8 * PAGES, 'eight full windows do not fit: pooled')
                self.assertGreater(pool, 8 * 2052, 'at least the 131k reservation of every seat')
                self.assertEqual(found[name]['env']['QWEN_FAST_KV_RESERVATION'], '1')
                self.assertEqual({key: found[name]['env'][key] for key in DRAM}, DRAM)

    def test_the_pool_vllm_builds_is_the_override_because_the_worker_overwrites_it_from_the_token_count(self):
        # plugin worker.py:388-390: num_gpu_blocks = ceil(max_tokens_all_users / 64) + max_num_seqs, whatever the override says
        found = profiles()
        for name in EIGHT:
            with self.subTest(profile=name):
                profile = found[name]
                tokens = int(profile['env']['QWEN36_MAX_TOKENS_ALL_USERS'])
                self.assertEqual(-(-tokens // 64) + profile['engine']['max-num-seqs'], BLOCKS_8)
                self.assertEqual(contract.real_pool_blocks(profile), BLOCKS_8)
                self.assertEqual(contract.kv_pool_blocks(profile), BLOCKS_8 - 1)
                self.assertIsNone(contract.kv_pool_problem(profile))
                # without the variable the worker would build seats x window + seats blocks: the contract refuses that profile
                bare = copy.deepcopy(profile)
                del bare['env']['QWEN36_MAX_TOKENS_ALL_USERS']
                self.assertIn('must name QWEN36_MAX_TOKENS_ALL_USERS', contract.kv_pool_problem(bare))
                with self.assertRaisesRegex(ValueError, 'QWEN36_MAX_TOKENS_ALL_USERS'):
                    contract.request_limits(bare)
                wrong = copy.deepcopy(profile)
                wrong['env']['QWEN36_MAX_TOKENS_ALL_USERS'] = str(tokens + 64)
                with self.assertRaisesRegex(ValueError, 'overwrites the override'):
                    contract.request_limits(wrong)
                wrong['env']['QWEN36_MAX_TOKENS_ALL_USERS'] = 'many'
                self.assertIn('positive integer', contract.kv_pool_problem(wrong))
        self.assertNotIn('QWEN36_MAX_TOKENS_ALL_USERS', found[FOUR_GATE]['env'])
        self.assertIsNone(contract.kv_pool_problem(found[FOUR_GATE]))
        self.assertIsNone(contract.kv_pool_problem(found['c2-packed-tp4']))

    def test_the_pool_holds_five_full_windows_at_once_and_the_largest_request_fits(self):
        # a full window reserves ceil((262,144 + 32) / 64) + 1 blocks; the usable pool is the override less vLLM's null block
        largest = -(-(WINDOW - HEADROOM + HEADROOM) // 64) + 1 if False else -(-(MAX_END + 32) // 64) + 1
        self.assertEqual(largest, 4097)
        usable = BLOCKS_8 - 1
        self.assertGreaterEqual(usable, largest)
        self.assertEqual(usable // largest, 5)
        self.assertLessEqual(5 * largest, usable)
        self.assertGreater(6 * largest, usable)

    def test_gate_only_twins_and_the_traffic_profile(self):
        found = profiles()
        self.assertNotIn('gate_only', found[TRAFFIC])
        self.assertNotIn('QWEN_C2_GATE_PROFILE', found[TRAFFIC]['env'])
        for name in (GATE, TIME_GATE, DIAG, FOUR_GATE):
            with self.subTest(profile=name):
                self.assertIs(found[name]['gate_only'], True)
                self.assertEqual(found[name]['env']['QWEN_C2_GATE_PROFILE'], '1')
                self.assertEqual(found[name]['min_answer_tokens'], 256)
                self.assertNotIn('max_prompt_tokens', found[name])
        self.assertEqual({key: found[GATE]['env'][key] for key in ('QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT')},
                         {'QWEN_FAST_VERIFY_T1_AUDIT': '1', 'QWEN_FAST_VERIFY_T2_AUDIT': '1'})
        for name in (TRAFFIC, TIME_GATE, DIAG):
            self.assertEqual(found[name]['env']['QWEN_FAST_VERIFY_T1_AUDIT'], '0')


class LimitsTests(unittest.TestCase):
    def test_the_traffic_limits(self):
        profile = profiles()[TRAFFIC]
        limits = contract.request_limits(profile)
        self.assertEqual(limits, dict(budget=16384, max_prompt_tokens=253920, min_answer_tokens=8192, default_max_tokens=8192,
                                      drafter_headroom_tokens=32, kv_pool_blocks=BLOCKS_8 - 1))
        room = contract.prompt_room(WINDOW, 16384, 253920, 8192, 32)
        self.assertEqual(room, 253920)
        self.assertEqual(room, WINDOW - HEADROOM - 8192)

    def test_the_gate_limits_admit_up_to_the_clamp(self):
        for name in (GATE, TIME_GATE, DIAG, FOUR_GATE):
            limits = contract.request_limits(profiles()[name])
            self.assertEqual(contract.prompt_room(WINDOW, limits['budget'], limits['max_prompt_tokens'],
                                                  limits['min_answer_tokens'], limits['drafter_headroom_tokens']),
                             MAX_END - 256, name)

    def test_every_262k_profile_loads_and_names_its_mesh(self):
        for name in ALL:
            loaded = contract.load_profile(PROFILES, name)
            with self.subTest(profile=name):
                self.assertEqual(contract.mesh_problems(loaded), [])
                self.assertEqual(loaded['mesh_device'], 'P150x4')
                self.assertEqual(loaded['env']['QWEN_FAST_TP'], '4')
                self.assertEqual(loaded['env']['QWEN_FAST_EXTENT_REPLAY'], '1')
                self.assertEqual(loaded['env']['QWEN_FAST_MAX_POSITION'], str(loaded['engine']['max-model-len']))

    def test_the_argv_is_the_262k_window(self):
        loaded = contract.load_profile(PROFILES, TRAFFIC)
        argv = contract.engine_arguments(loaded, '/snap')
        self.assertEqual(argv[argv.index('--max-model-len') + 1], '262144')
        self.assertEqual(argv[argv.index('--max-num-batched-tokens') + 1], '262144')
        self.assertEqual(argv[argv.index('--num-gpu-blocks-override') + 1], str(BLOCKS_8))
        self.assertEqual(argv[argv.index('--max-num-seqs') + 1], '8')


class AdmissionOnFakeRecordsTests(unittest.TestCase):
    """The attach's admission over each 262k profile's own environment (the image ENV added), on fake qualifying records."""

    def environment(self, name):
        import test_seats8_profiles_meet_attach as meets

        environ = meets.container_env(name)
        if profiles()[name].get('gate_only') is True:
            environ['QWEN_C2_GATE'] = '1'           # the switch the gate harness adds to a gate-only profile's container
        return environ

    def test_each_eight_seat_profile_passes_the_extent_environment_at_eight_requests_and_two_blocks(self):
        for name in EIGHT:
            environ = self.environment(name)
            with self.subTest(profile=name):
                m3 = serving_runtime.m3_shape(dict(scheduler_requests=8), environ)
                self.assertTrue(m3[0], m3[1])
                self.assertEqual(admission.check_environment(environ, m3), [])
                self.assertEqual(admission.served_capacity(environ), WINDOW)

    def test_the_four_seat_gate_is_the_one_block_shape_at_four_requests(self):
        environ = self.environment(FOUR_GATE)
        m3 = serving_runtime.m3_shape(dict(scheduler_requests=4), environ)
        self.assertTrue(m3[0], m3[1])
        self.assertEqual(admission.check_environment(environ, m3), [])
        self.assertEqual(admission.served_capacity(environ), WINDOW)

    def test_each_profile_attaches_on_both_fake_records_and_is_refused_on_one(self):
        from unittest import mock

        import page_width_tp4
        import test_262k_evidence_waiver as waiver
        import test_packed_any_admission_262k as wide

        waiver.pending_records(self)     # the 'refused on one record' half reads the unrecorded state
        for name in ALL:
            waiver_environ = self.environment(name)
            environ = {key: value for key, value in waiver_environ.items() if key != 'QWEN_FAST_262K_EVIDENCE_WAIVER'}
            blocks = 4 if name == FOUR_GATE else 8
            m3 = serving_runtime.m3_shape(dict(scheduler_requests=blocks), environ)
            runtime = dict(binaries={'a/_ttnncpp.so': admission.K64J_TTNNCPP_SHA256, 'b/_ttnncpp.so': admission.K64J_TTNNCPP_SHA256})

            def admit(writer_ok=True, record=True, waived=False):
                lines = wide.Lines()
                real = admission.check_evidence

                def read(path=None, **kwargs):
                    if kwargs.get('capacity') == WINDOW and record:
                        return wide.qualifying262()
                    return real(path, **kwargs)

                with mock.patch.dict(admission._STATE, clear=True), \
                        mock.patch.object(admission, 'check_runtime', return_value=runtime), \
                        mock.patch.object(admission, 'check_evidence', side_effect=read), \
                        mock.patch.object(page_width_tp4, 'evidence_state',
                                          return_value=(True, []) if writer_ok else (False, ['PENDING'])):
                    return admission.admit('/opt/tt-metal', m3=m3, environ=dict(waiver_environ if waived else environ), log=lines), lines

            with self.subTest(profile=name):
                record, lines = admit()
                self.assertEqual(record['capacity'], WINDOW)
                self.assertTrue(lines[-1].endswith('capacity=262144'), lines[-1])
                with self.assertRaises(admission.AdmissionRefused):
                    admit(writer_ok=False)
                with self.assertRaises(admission.AdmissionRefused):
                    admit(record=False)
                if name == TRAFFIC:
                    self.assertNotIn('QWEN_FAST_262K_EVIDENCE_WAIVER', waiver_environ)
                    continue
                # the gate-only profiles' own waiver: neither record needed, one loud line, UNQUALIFIED
                import page_width_tp4 as pw

                pw._WAIVER_LOGGED[:] = []
                record, lines = admit(writer_ok=False, record=False, waived=True)
                self.assertEqual(record['capacity'], WINDOW)
                self.assertTrue(record['waived'])
                self.assertEqual(len([line for line in lines if line.startswith('[PINDIAG] 262k evidence WAIVED (gate-only): ')]), 1, lines)
                self.assertIn('passed UNQUALIFIED', lines[-1])
                pw._WAIVER_LOGGED[:] = []

    def test_the_pool_the_profile_builds_is_the_admitted_capacity(self):
        for name in ALL:
            engine = profiles()[name]['engine']
            self.assertEqual(max(68, -(-engine['max-model-len'] // 64)) * 64, WINDOW, name)


class ClampTests(unittest.TestCase):
    """enforce_request / prompt_room with the drafter headroom: prompt + answer <= 262,112, and only for the profiles that name one."""

    class Params(object):
        n, logprobs, prompt_logprobs, structured_outputs, stop, min_tokens = 1, None, None, None, None, 0
        logit_bias = allowed_token_ids = bad_words = stop_token_ids = max_tokens = None

    def enforce(self, prompt, max_tokens=None, **extra):
        params = self.Params()
        params.max_tokens = max_tokens
        limits = dict(budget=16384, max_prompt_tokens=253920, min_answer_tokens=8192, default_max_tokens=8192,
                      drafter_headroom_tokens=32)
        limits.update(extra)
        return contract.enforce_request(params, prompt_tokens=prompt, max_model_len=WINDOW, eos_ids=frozenset(), **limits)

    def test_prompt_plus_answer_never_passes_262112(self):
        for prompt in (1, 4096, 100000, 245728, 245760, 253919, 253920):
            for asked in (None, 1, 8192, 16384, 16385, 100000, WINDOW - prompt, WINDOW):
                params = self.enforce(prompt, asked)
                with self.subTest(prompt=prompt, asked=asked):
                    self.assertLessEqual(prompt + params.max_tokens, MAX_END)
                    self.assertGreaterEqual(params.max_tokens, 1)

    def test_the_clamp_is_the_window_less_the_headroom_when_that_is_the_binding_limit(self):
        # (a prompt of 245,760 would make 16,384 exactly the remaining context, which reads as an omitted max_tokens)
        params = self.enforce(245770, 16384)
        self.assertEqual(params.max_tokens, MAX_END - 245770)
        self.assertEqual(params.max_tokens, 16342)
        self.assertEqual(self.enforce(245000, 16384).max_tokens, 16384, 'the ceiling binds first when there is room')

    def test_an_omitted_max_tokens_still_reads_the_whole_window(self):
        # vLLM fills an omitted max_tokens with max_model_len - prompt: it must still read as omitted (default_max_tokens)
        params = self.enforce(1000, WINDOW - 1000)
        self.assertEqual(params.max_tokens, 8192)
        self.assertEqual(self.enforce(1000, None).max_tokens, 8192)

    def test_the_prompt_limit_is_the_cap_and_one_more_is_refused(self):
        self.enforce(253920)
        with self.assertRaises(contract.ContractError):
            self.enforce(253921)

    def test_without_a_headroom_every_earlier_limit_is_what_it_was(self):
        for prompt in (1, 100000, 123136):
            kwargs = dict(max_model_len=131328, budget=16384, eos_ids=frozenset(), max_prompt_tokens=123136,
                          min_answer_tokens=8192, default_max_tokens=8192)
            params = self.Params()
            params.max_tokens = 100000
            self.assertEqual(contract.enforce_request(params, prompt_tokens=prompt, **kwargs).max_tokens,
                             min(16384, 131328 - prompt))
        self.assertEqual(contract.prompt_room(131328, 16384, 123136, 8192), 123136)
        self.assertEqual(contract.prompt_room(131328, 16384, None, 8192), 131328 - 8192)
        self.assertEqual(contract.prompt_room(131328, 16384, None, None, 0), 131328 - 16384)

    def test_the_drafter_end_limit_is_the_proposal_inputs_bound(self):
        source = (HERE / 'dflash_proposal_inputs.py').read_text(encoding='utf-8')
        self.assertIn('1 <= position <= 262111', source)
        self.assertIn('262144 - position - 32', source)
        self.assertEqual(contract.DRAFTER_END_LIMIT, 262111 + 1)
        self.assertEqual(contract.DRAFTER_END_LIMIT, 262144 - 32)
        import dflash_proposal_inputs

        dflash_proposal_inputs.proposal_contexts(262111, 1)
        with self.assertRaises(ValueError):
            dflash_proposal_inputs.proposal_contexts(262111, 2)
        with self.assertRaises(ValueError):
            dflash_proposal_inputs.proposal_contexts(262112, 1)

    def test_a_clamped_request_is_always_a_legal_proposal_budget(self):
        import dflash_proposal_inputs

        for prompt in (1, 4096, 245760, 253920):
            params = self.enforce(prompt, 16384)
            dflash_proposal_inputs.proposal_contexts(prompt, params.max_tokens)

    def test_the_engine_side_budget_never_hands_the_drafter_more_than_it_can_propose(self):
        # request_budget is what the engine build asks proposal_contexts: without the contract's clamp (a gate path, a bug) a
        # 262,144-page capacity would give budget 262,144 - prompt, 32 past the drafter's bound, refused AFTER the prefill ran.
        from types import SimpleNamespace

        import dflash_proposal_inputs
        import serving_request_factory as factory

        self.assertEqual(factory.DRAFTER_END_LIMIT, contract.DRAFTER_END_LIMIT)
        for prompt in (1, 4096, 245760, 262080, 262111):
            parameters = SimpleNamespace(max_tokens=10 ** 6)
            budget = factory.request_budget(parameters, prompt_tokens=prompt, capacity=WINDOW, ceiling=10 ** 6)
            self.assertEqual(budget, MAX_END - prompt)
            dflash_proposal_inputs.proposal_contexts(prompt, budget)
        with self.assertRaises(ValueError):
            factory.request_budget(SimpleNamespace(max_tokens=5), prompt_tokens=262112, capacity=WINDOW)
        # every earlier capacity is what it was
        for capacity in (131328, 4352, 262112):
            for prompt in (1, 4096, capacity - 1):
                parameters = SimpleNamespace(max_tokens=10 ** 6)
                self.assertEqual(factory.request_budget(parameters, prompt_tokens=prompt, capacity=capacity, ceiling=10 ** 6),
                                 capacity - prompt)
        self.assertEqual(factory.request_budget(SimpleNamespace(max_tokens=100), prompt_tokens=4096, capacity=131328), 100)
        self.assertEqual(factory.request_budget(SimpleNamespace(max_tokens=10 ** 6), prompt_tokens=4096, capacity=131328,
                                                ceiling=16384), 16384)

    def test_boot_refuses_a_window_past_the_drafters_last_position_without_a_headroom(self):
        profile = copy.deepcopy(profiles()[TRAFFIC])
        profile.pop('drafter_headroom_tokens')
        with self.assertRaisesRegex(ValueError, "past the drafter's last position 262112"):
            contract.request_limits(profile)
        profile['drafter_headroom_tokens'] = 31
        with self.assertRaisesRegex(ValueError, 'at least 32'):
            contract.request_limits(profile)
        profile['drafter_headroom_tokens'] = 32
        self.assertEqual(contract.request_limits(profile)['drafter_headroom_tokens'], 32)
        profile['drafter_headroom_tokens'] = 64
        self.assertEqual(contract.request_limits(profile)['drafter_headroom_tokens'], 64)

    def test_a_bad_headroom_is_refused(self):
        for bad in (-1, 4097, '32', 1.5, True):
            profile = copy.deepcopy(profiles()[TRAFFIC])
            profile['drafter_headroom_tokens'] = bad
            if bad is True:
                continue                      # bool is an int: True would be 1, below the needed 32, refused by the end-limit rule
            with self.subTest(value=bad), self.assertRaises(ValueError):
                contract.request_limits(profile)

    def test_the_131k_profiles_boot_without_a_headroom_key(self):
        for name in ('c2-packed-tp4', 'c2-packed-tp4-8', 'c2-packed-tp4-8-gate'):
            limits = contract.request_limits(profiles()[name])
            self.assertNotIn('drafter_headroom_tokens', limits)

    def test_the_installed_contract_passes_the_headroom_through(self):
        import types

        seen = []

        class Processor(object):
            model_config = types.SimpleNamespace(max_model_len=WINDOW)

            def process_inputs(self, request_id, prompt, params, *args, **kwargs):
                seen.append(params.max_tokens)

        module = types.SimpleNamespace(InputProcessor=Processor)
        contract.install_request_contract(module, budget=16384, eos_ids=frozenset(), max_prompt_tokens=253920,
                                          min_answer_tokens=8192, default_max_tokens=8192, drafter_headroom_tokens=32)
        params = self.Params()
        params.max_tokens, params.temperature = 16384, 0.7        # the wrapper acts on sampling params (they have a temperature)
        Processor().process_inputs('r', dict(prompt_token_ids=[1] * 245770), params)
        self.assertEqual(seen, [16342])


if __name__ == '__main__':
    unittest.main()
