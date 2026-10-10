import json
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import serving_c2_contract as contract

PROFILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qwen_c2_profiles.json')


class _ParkedTwins(object):
    """The generated engine-reuse gate twins by name (this file ships in the image, where make_parked_profiles does not: test_parked_tp4_profiles holds the list)."""

    def __contains__(self, name):
        # (and the round-host gate twins of the Lever N traffic profile: test_tp4_round_host holds the list)
        return ('-ship-prefix-levern-parked' in name or name.endswith('-ship-prefix-levern-audit-r2')
                or '-ship-prefix-levern-traffic-roundhost' in name)


PARKED_TWINS = _ParkedTwins()


class Params(object):
    def __init__(self, **values):
        self.n, self.logprobs, self.prompt_logprobs = 1, None, None
        self.structured_outputs, self.logit_bias, self.allowed_token_ids, self.bad_words = None, None, None, None
        self.stop, self.stop_token_ids, self.min_tokens = None, None, 0
        self.temperature, self.top_p, self.top_k, self.min_p = 1.0, 0.95, 20, 0.0
        self.presence_penalty, self.frequency_penalty, self.repetition_penalty = 0.0, 0.0, 1.05
        self.seed, self.max_tokens = 7, None
        for key, value in values.items():
            setattr(self, key, value)


def enforce(params, prompt_tokens=1000, max_model_len=131328, budget=4096, max_prompt_tokens=61440, **c2):
    return contract.enforce_request(params, prompt_tokens=prompt_tokens, max_model_len=max_model_len,
                                    budget=budget, eos_ids=frozenset((248046, 248044)),
                                    max_prompt_tokens=max_prompt_tokens, **c2)


class ProfileTest(unittest.TestCase):
    def test_profiles_hold_the_measured_c2_argv(self):
        exact = contract.load_profile(PROFILES, 'exact')
        snapshot = exact['snapshots'][1]
        argv = contract.engine_arguments(exact, snapshot)
        # The v235 command (run 36087022223, m3native-gate.json .command) after --port 8000,
        # with the gate's own --model, --served-model-name and --host left to the platform.
        self.assertEqual(argv[:2], ['--model', snapshot])
        self.assertIn('--max-model-len', argv)
        self.assertEqual(argv[argv.index('--max-model-len') + 1], '131328')
        self.assertEqual(argv[argv.index('--num-gpu-blocks-override') + 1], '8208')
        additional = json.loads(argv[argv.index('--additional-config') + 1])
        self.assertEqual(additional['tt'], {'trace_mode': 'decode_only', 'trace_region_size': 268435456,
                                            'l1_small_size': 24576})
        self.assertIs(additional['qwen_fast_t16'], True)
        self.assertEqual(additional['qwen_fast_runtime']['target_snapshot'], snapshot)
        speculative = json.loads(argv[argv.index('--speculative-config') + 1])
        self.assertEqual(speculative['num_speculative_tokens'], 15)
        self.assertEqual(exact['env']['QWEN_FAST_OUTPUT_BUDGET'], '256')

    def test_profile_geometry_is_consistent(self):
        for name in ('exact', 'coding', 'c2', 'c2-gate', 'c2-packed', 'c2-packed-gate'):
            profile = contract.load_profile(PROFILES, name)
            engine, env = profile['engine'], profile['env']
            self.assertIs(engine['additional-config']['qwen_fast_t16'], True, name)
            budget = int(env['QWEN_FAST_OUTPUT_BUDGET'])
            capacity = int(env['QWEN_DSPARK_REQUEST_CONTEXT']) + 256
            self.assertEqual(engine['max-model-len'], capacity, name)
            self.assertEqual(int(env['QWEN_FAST_MAX_POSITION']), capacity, name)
            # ordered_cache admits page tables up to 1024 pages or exactly 2052 (131,328 positions).
            self.assertIn(capacity // 64, (2052,) + tuple(range(1, 1025)), name)
            self.assertEqual(budget % 256, 0, name)
            # Every admitted request fits the KV cache at once: no preemption on the fast path.
            # The longest request is the longest admitted prompt plus the most the contract lets
            # it answer, which is the budget clamped to the context left after the prompt.
            room = contract.prompt_room(capacity, budget, profile.get('max_prompt_tokens'),
                                        profile.get('min_answer_tokens'))
            longest = min(room + budget, capacity)
            self.assertGreaterEqual(engine['num-gpu-blocks-override'],
                                    engine['max-num-seqs'] * -(-longest // 64), name)
            # The request contract's numbers load and check at boot.
            self.assertEqual(contract.request_limits(profile)['budget'], budget, name)

    def test_the_c2_profile_is_the_exact_geometry_serving_any_request(self):
        exact, c2 = contract.load_profile(PROFILES, 'exact'), contract.load_profile(PROFILES, 'c2')
        # The same engine argv, byte for byte: 131328 positions, 2052-page tables, 8208 blocks.
        self.assertEqual(contract.engine_arguments(c2, '/snap'), contract.engine_arguments(exact, '/snap'))
        self.assertEqual(c2['engine']['num-gpu-blocks-override'], 4 * 2052)
        env = c2['env']
        self.assertEqual(env['QWEN_FAST_OUTPUT_BUDGET'], '16384')
        self.assertEqual((env['QWEN_FAST_MAX_POSITION'], env['QWEN_DSPARK_REQUEST_CONTEXT']), ('131328', '131072'))
        self.assertEqual(env['QWEN_FAST_ANY_REQUEST'], '1')
        self.assertEqual(env['QWEN_FAST_FAULTHANDLER'], '0')
        # No packed block: it cannot engage under the 123136-token cap and its 3.84 GB per chip is what
        # the fourth per-request engine needs (run 36218104858).
        self.assertEqual(env['QWEN_FAST_PACKED_STEP'], '0')
        # ...and so no padded block, which is admitted at the 64-row block only (run 36219636175).
        self.assertEqual(env['QWEN_FAST_PADDED_BLOCK'], '0')
        limits = contract.request_limits(c2)
        self.assertEqual(limits, dict(budget=16384, max_prompt_tokens=123136, min_answer_tokens=8192,
                                      default_max_tokens=8192))
        room = contract.prompt_room(131328, 16384, 123136, 8192)
        self.assertEqual(room, 123136)
        self.assertGreaterEqual(131328 - room, 8192, 'every admitted prompt keeps at least 8k of answer room')

    def test_the_c2_gate_profile_is_c2_at_exacts_prompt_limit(self):
        """The plan's bring-up runs c2 at v235's shape, 4 x 131072, and compares byte for byte. Under
        c2 itself the edge refuses every 131072-token prompt (its cap is 123136), so the packed
        block never engages and the one direct A/B against v235 could not run: c2-gate is c2's
        environment with exact's prompt limit, and exact's engine."""
        exact, c2, gate = (contract.load_profile(PROFILES, name) for name in ('exact', 'c2', 'c2-gate'))
        for snapshot in ('/snap', exact['snapshots'][1]):
            self.assertEqual(contract.engine_arguments(gate, snapshot), contract.engine_arguments(exact, snapshot))
        self.assertEqual(gate['engine'], exact['engine'])
        # c2's code paths and ceiling, but WITH the packed block: the bring-up's 4 x 131072 users are the
        # one shape it serves, and the image's own QWEN_FAST_PACKED_STEP=1 builds it.
        block_only = ('QWEN_FAST_PACKED_STEP', 'QWEN_FAST_PADDED_BLOCK')
        self.assertEqual(gate['env'], {key: value for key, value in c2['env'].items() if key not in block_only},
                         'the c2 code paths, the c2 ceiling')
        for key in block_only:
            self.assertNotIn(key, gate['env'])
        for key in ('eos_ids', 'snapshots', 'mesh_graph_descriptor', 'default_max_tokens'):
            self.assertEqual(gate.get(key), c2.get(key), key)
        limits = contract.request_limits(gate)
        self.assertEqual(limits, dict(budget=16384, max_prompt_tokens=None, min_answer_tokens=256,
                                      default_max_tokens=8192))
        room = contract.prompt_room(131328, 16384, None, 256)
        self.assertEqual(room, 131072)
        self.assertEqual(room, contract.prompt_room(131328, int(exact['env']['QWEN_FAST_OUTPUT_BUDGET'])),
                         "exact's prompt limit")
        gate_limits = dict(max_model_len=131328, max_prompt_tokens=None, min_answer_tokens=256,
                           budget=16384, default_max_tokens=8192)
        # v235's requests: 131072 tokens, max_tokens 256 - admitted, and the budget is exact's
        self.assertEqual(enforce(Params(max_tokens=256), prompt_tokens=131072, **gate_limits).max_tokens, 256)
        self.assertEqual(enforce(Params(max_tokens=16384), prompt_tokens=131072, **gate_limits).max_tokens, 256)
        self.assertEqual(enforce(Params(max_tokens=131328 - 131072), prompt_tokens=131072,
                                 **gate_limits).max_tokens, 256)
        with self.assertRaises(contract.ContractError):
            enforce(Params(max_tokens=256), prompt_tokens=131073, **gate_limits)
        # ...which c2 refuses outright
        with self.assertRaisesRegex(contract.ContractError, 'exceeds the 123136-token prompt limit'):
            enforce(Params(max_tokens=256), prompt_tokens=131072, max_model_len=131328,
                    **dict(contract.request_limits(c2)))

    def test_no_other_profile_turns_the_any_request_path_on(self):
        for name in ('exact', 'coding', 'general'):
            profile = contract.load_profile(PROFILES, name)
            self.assertNotIn('QWEN_FAST_ANY_REQUEST', profile['env'], name)
            self.assertNotIn('QWEN_FAST_FAULTHANDLER', profile['env'], name)
            for key in ('min_answer_tokens', 'default_max_tokens'):
                self.assertNotIn(key, profile, name)
        exact = contract.load_profile(PROFILES, 'exact')
        self.assertEqual(exact['env'], {'QWEN_FAST_OUTPUT_BUDGET': '256', 'QWEN_FAST_MAX_POSITION': '131328',
                                        'QWEN_DSPARK_REQUEST_CONTEXT': '131072'})

    @unittest.skipUnless(os.path.isfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docker',
                                                     'qwen-c2-serving.Dockerfile')), 'repository checkout only')
    def test_the_image_does_not_bake_the_any_request_flag(self):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docker',
                            'qwen-c2-serving.Dockerfile')
        with open(path, encoding='utf-8') as handle:
            self.assertNotIn('QWEN_FAST_ANY_REQUEST', handle.read(), 'only the c2 profiles may set it')

    def test_default_profile_is_the_four_card_traffic_profile(self):
        # serving/tp4-s2: the image serves c2-packed-tp4 when the platform names no profile (the agent forwards none).
        environ = dict(os.environ)
        os.environ.pop('QWEN_C2_PROFILE', None)
        try:
            self.assertEqual(contract.load_profile(PROFILES)['name'], 'c2-packed-tp4')
        finally:
            os.environ.clear()
            os.environ.update(environ)


DOCKERFILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docker', 'qwen-c2-serving.Dockerfile')
EXTENT_FLAG = 'QWEN_FAST_EXTENT_REPLAY'
# S2's gate-only knobs (design 1.2, W3/W4/W11): the gate sets them per arm; no profile and no image may.
GATE_ONLY = ('QWEN_FAST_EXTENT_AUDIT', 'QWEN_FAST_PACKED_CAPTURE_POSITION', 'QWEN_FAST_GATE_FORCE_CAP')


class PackedAnyProfileTest(unittest.TestCase):
    """S2 (design W8, 1.2): c2-packed is c2 with the packed block and the extent flag; c2-packed-gate is
    c2-gate with the flag. exact, c2 and c2-gate keep their bytes, and are the rollback and the flag-off arms."""

    def load(self, name):
        return contract.load_profile(PROFILES, name)

    def test_c2_packed_is_c2_less_its_block_overrides_plus_the_flag(self):
        c2, packed = self.load('c2'), self.load('c2-packed')
        block_only = ('QWEN_FAST_PACKED_STEP', 'QWEN_FAST_PADDED_BLOCK')
        self.assertEqual(packed['env'], dict({key: value for key, value in c2['env'].items() if key not in block_only},
                                             **{EXTENT_FLAG: '1'}))
        for key in block_only:
            self.assertIn(key, c2['env'])
            self.assertNotIn(key, packed['env'], 'the image\'s own value (1) builds the block')
        for key in set(c2) | set(packed):
            if key not in ('env', 'description', 'name'):
                self.assertEqual(packed.get(key), c2.get(key), key)
        self.assertEqual(contract.request_limits(packed), dict(budget=16384, max_prompt_tokens=123136,
                                                               min_answer_tokens=8192, default_max_tokens=8192))
        self.assertIs(contract.parser_rechunk(packed), True)

    def test_c2_packed_gate_is_c2_gate_plus_the_flag(self):
        gate, packed_gate = self.load('c2-gate'), self.load('c2-packed-gate')
        self.assertEqual(packed_gate['env'], dict(gate['env'], **{EXTENT_FLAG: '1'}))
        for key in set(gate) | set(packed_gate):
            if key not in ('env', 'description', 'name'):
                self.assertEqual(packed_gate.get(key), gate.get(key), key)
        self.assertEqual(contract.request_limits(packed_gate), contract.request_limits(gate))

    def test_the_s2_engines_are_exacts(self):
        exact = self.load('exact')
        for name in ('c2-packed', 'c2-packed-gate'):
            for snapshot in ('/snap', exact['snapshots'][1]):
                with self.subTest(profile=name, snapshot=snapshot):
                    self.assertEqual(contract.engine_arguments(self.load(name), snapshot),
                                     contract.engine_arguments(exact, snapshot))

    def test_only_the_s2_profiles_set_the_flag_and_none_sets_a_gate_only_knob(self):
        with open(PROFILES, encoding='utf-8') as handle:
            names = sorted(json.load(handle)['profiles'])
        # ...and the sticky-session profiles, c2-packed and c2-packed-gate with prefix reuse.
        # ...and the four-card twins, c2-packed-tp4 and its gate profile (plan S2-TP4), and the two gate arms of the S2 window
        # (the ring fabric, the bfloat16 drafter), and the four batched-draft profiles (tp4/draft).
        # ...and the generated engine-reuse twins of the Lever N profiles (tp4/engine-reuse; test_parked_tp4_profiles holds each as its parent plus its flags).
        self.assertEqual([name for name in names if EXTENT_FLAG in self.load(name)['env'] and name not in PARKED_TWINS],
                         ['c2-packed', 'c2-packed-gate', 'c2-packed-prefix', 'c2-packed-prefix-gate', 'c2-packed-tp4',
                          'c2-packed-tp4-262k-gate', 'c2-packed-tp4-8', 'c2-packed-tp4-8-best', 'c2-packed-tp4-8-best-quad', 'c2-packed-tp4-8-best-quad-dbf16', 'c2-packed-tp4-8-best-quad-gate', 'c2-packed-tp4-8-diag-strace', 'c2-packed-tp4-8-diag-strace-nowarm', 'c2-packed-tp4-8-diag-strace-rshard', 'c2-packed-tp4-8-gate', 'c2-packed-tp4-8-time-gate', 'c2-packed-tp4-8x262k', 'c2-packed-tp4-8x262k-best', 'c2-packed-tp4-8x262k-best-audit', 'c2-packed-tp4-8x262k-best-levern-audit', 'c2-packed-tp4-8x262k-best-levern-control-audit', 'c2-packed-tp4-8x262k-best-levern-final-hold-time-gate', 'c2-packed-tp4-8x262k-best-levern-foreign-time-gate', 'c2-packed-tp4-8x262k-best-levern-hang-gate', 'c2-packed-tp4-8x262k-best-levern-r1-time-gate', 'c2-packed-tp4-8x262k-best-levern-time-gate', 'c2-packed-tp4-8x262k-best-nosamp-audit', 'c2-packed-tp4-8x262k-best-sdpamulti', 'c2-packed-tp4-8x262k-best-sdpamulti-audit', 'c2-packed-tp4-8x262k-best-stack-audit', 'c2-packed-tp4-8x262k-best-time-gate', 'c2-packed-tp4-8x262k-best-time-gate-d2', 'c2-packed-tp4-8x262k-best-time-gate-dbf16', 'c2-packed-tp4-8x262k-best-time-gate-lookup', 'c2-packed-tp4-8x262k-best-time-gate-nosamp', 'c2-packed-tp4-8x262k-best-time-gate-s1', 'c2-packed-tp4-8x262k-best-time-gate-stack', 'c2-packed-tp4-8x262k-best-time-gate-u1', 'c2-packed-tp4-8x262k-best-u1-audit', 'c2-packed-tp4-8x262k-diag-strace', 'c2-packed-tp4-8x262k-gate', 'c2-packed-tp4-8x262k-hostgap-1', 'c2-packed-tp4-8x262k-hostgap-1-audit', 'c2-packed-tp4-8x262k-hostgap-2', 'c2-packed-tp4-8x262k-hostgap-2-audit', 'c2-packed-tp4-8x262k-prefix-gate', 'c2-packed-tp4-8x262k-prefix-time-gate', 'c2-packed-tp4-8x262k-ship', 'c2-packed-tp4-8x262k-ship-prefix', 'c2-packed-tp4-8x262k-ship-prefix-audit', 'c2-packed-tp4-8x262k-ship-prefix-audit-digests', 'c2-packed-tp4-8x262k-ship-prefix-dbf16', 'c2-packed-tp4-8x262k-ship-prefix-dckdefault', 'c2-packed-tp4-8x262k-ship-prefix-levern', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-traffic', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-f1', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-ln', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-sdpa', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-w1', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-pool', 'c2-packed-tp4-8x262k-ship-prefix-w2', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit-pool', 'c2-packed-tp4-8x262k-time-gate', 'c2-packed-tp4-8x262k-w1', 'c2-packed-tp4-8x262k-w1-audit', 'c2-packed-tp4-8x262k-w1-audit-nod1', 'c2-packed-tp4-8x262k-w1-lite', 'c2-packed-tp4-8x262k-w1-nod1', 'c2-packed-tp4-8x262k-w2', 'c2-packed-tp4-8x262k-w2-audit', 'c2-packed-tp4-8x262k-w2-nof1', 'c2-packed-tp4-8x262k-w2-nof1-audit', 'c2-packed-tp4-best', 'c2-packed-tp4-best-d2', 'c2-packed-tp4-best-dbf16', 'c2-packed-tp4-best-gate', 'c2-packed-tp4-best-gate-d2', 'c2-packed-tp4-best-gate-dbf16', 'c2-packed-tp4-best-gate-glue', 'c2-packed-tp4-best-gate-lookup', 'c2-packed-tp4-best-gate-samp', 'c2-packed-tp4-best-gate-tpub', 'c2-packed-tp4-best-lookup', 'c2-packed-tp4-best-rshard', 'c2-packed-tp4-best-samp', 'c2-packed-tp4-best-sdpa', 'c2-packed-tp4-best-ship', 'c2-packed-tp4-best-ship-glue', 'c2-packed-tp4-best-ship-tpub', 'c2-packed-tp4-best-ship-warm4', 'c2-packed-tp4-best-strace', 'c2-packed-tp4-best-strace-glue', 'c2-packed-tp4-best-strace-tpub', 'c2-packed-tp4-best-v5', 'c2-packed-tp4-best-v5-gate', 'c2-packed-tp4-diag', 'c2-packed-tp4-diag-rshard', 'c2-packed-tp4-diag-sprewarm',
                          'c2-packed-tp4-diag-strace', 'c2-packed-tp4-diag-t1', 'c2-packed-tp4-diag-t1-rshard-audit', 'c2-packed-tp4-diag-t2', 'c2-packed-tp4-f12',
                          'c2-packed-tp4-f2', 'c2-packed-tp4-gate', 'c2-packed-tp4-gate-bf16', 'c2-packed-tp4-gate-draftwide', 'c2-packed-tp4-gate-fcommit',
                          'c2-packed-tp4-gate-fcommit-live', 'c2-packed-tp4-gate-fcommit-quad', 'c2-packed-tp4-gate-noslide',
                          'c2-packed-tp4-gate-pairs', 'c2-packed-tp4-gate-pairslice', 'c2-packed-tp4-gate-quad', 'c2-packed-tp4-gate-ring', 'c2-packed-tp4-gate-rshard-audit', 'c2-packed-tp4-gate-vglue',
                          'c2-packed-tp4-lanes-gate', 'c2-packed-tp4-lanes-time-gate', 'c2-packed-tp4-solo-gate',
                          'c2-packed-tp4-solo-time-gate', 'c2-packed-tp4-speed', 'c2-packed-tp4-speed-fcommit',
                          'c2-packed-tp4-speed-fcommit-oop', 'c2-packed-tp4-speed-fcommit-quad', 'c2-packed-tp4-speed-fix',
                          'c2-packed-tp4-speed-noslide', 'c2-packed-tp4-speed-pairs', 'c2-packed-tp4-speed-quad',
                          'c2-packed-tp4-speed-rshard', 'c2-packed-tp4-speed-sprewarm', 'c2-packed-tp4-speed-strace', 'c2-packed-tp4-speed-strace-dispatchdiag', 'c2-packed-tp4-speed-strace-draftwide', 'c2-packed-tp4-speed-strace-fcommit', 'c2-packed-tp4-speed-strace-pairslice', 'c2-packed-tp4-speed-strace-ring', 'c2-packed-tp4-speed-strace-v5', 'c2-packed-tp4-speed-vglue',
                          'c2-packed-tp4-speed-vglue-c1a', 'c2-packed-tp4-speed-vglue-v1', 'c2-packed-tp4-speed-vglue-v2',
                          'c2-packed-tp4-speed-vglue-v3a', 'c2-packed-tp4-speed-vglue-v4a', 'c2-packed-tp4-speed-warm4', 'c2-packed-tp4-time-gate', 'c2-packed-tp4-warm4-control', 'c2-packed-tp4-warm4-diag', 'c2-packed-tp4-warm4-diag-oldtail',
                          'c2-packed-tp4-warm4-even-diag', 'c2-packed-tp4-warm4-gate'])
        for name in names:
            env = self.load(name)['env']
            with self.subTest(profile=name):
                self.assertIn(env.get(EXTENT_FLAG, '0'), ('0', '1'))
                for knob in GATE_ONLY:
                    self.assertNotIn(knob, env)
        # The rollback and the flag-off arms are untouched by S2 (design 1.2: W6 no longer edits them).
        self.assertEqual(self.load('exact')['env'], {'QWEN_FAST_OUTPUT_BUDGET': '256', 'QWEN_FAST_MAX_POSITION': '131328',
                                                     'QWEN_DSPARK_REQUEST_CONTEXT': '131072'})
        self.assertEqual(self.load('c2')['env']['QWEN_FAST_PACKED_STEP'], '0')

    def test_apply_environment_exports_the_tail_caps_and_the_diagnostics_and_the_diag_profiles_need_the_gate(self):
        # tp4-serve-3: the traffic profile carries the tail caps and the watchdog; the diag arms carry the host-only instruments
        # with the caps off and, being gate only, refuse to boot without QWEN_C2_GATE=1.
        caps = {'QWEN_FAST_BUDGET_CAP': '1', 'QWEN_FAST_SEQ_DEADLINE_S': '120'}
        instruments = ('QWEN_FAST_SEQ_STAGE_LOG', 'QWEN_FAST_TRACE_CENSUS', 'QWEN_FAST_CCL_HANDLE_LOG', 'QWEN_FAST_MEMORY_LEDGER_L1')
        for name in ('c2-packed-tp4', 'c2-packed-tp4-speed'):
            exported = contract.apply_environment(self.load(name), {})
            for key, value in caps.items():
                self.assertEqual(exported[key], value, (name, key))
            for key in instruments:
                self.assertNotIn(key, exported, (name, key))
        for name in ('c2-packed-tp4-diag', 'c2-packed-tp4-diag-rshard', 'c2-packed-tp4-diag-sprewarm', 'c2-packed-tp4-diag-strace', 'c2-packed-tp4-diag-t1', 'c2-packed-tp4-diag-t1-rshard-audit', 'c2-packed-tp4-diag-t2'):
            profile = dict(self.load(name), name=name)
            exported = contract.apply_environment(profile, {})
            self.assertEqual((exported['QWEN_FAST_BUDGET_CAP'], exported['QWEN_FAST_SEQ_DEADLINE_S']), ('0', '120'), name)
            for key in instruments:
                self.assertEqual(exported[key], '1', (name, key))
            self.assertEqual(len(contract.gate_problems(profile, {})), 1, name)
            self.assertEqual(contract.gate_problems(profile, {contract.GATE_SWITCH: '1'}), [], name)
        self.assertEqual(contract.gate_problems(dict(self.load('c2-packed-tp4'), name='c2-packed-tp4'), {}), [])
        self.assertEqual(len(contract.gate_problems(dict(self.load('c2-packed-tp4-speed'), name='c2-packed-tp4-speed'), {})), 1)

    @unittest.skipUnless(os.path.isfile(DOCKERFILE), 'repository checkout only')
    def test_the_image_bakes_neither_the_flag_nor_a_gate_only_knob(self):
        with open(DOCKERFILE, encoding='utf-8') as handle:
            text = handle.read()
        for name in (EXTENT_FLAG,) + GATE_ONLY:
            self.assertNotIn(name, text, 'only the S2 profiles (the flag) or the gate (the knobs) may set %s' % name)

    @unittest.skipUnless(os.path.isfile(DOCKERFILE), 'repository checkout only')
    def test_the_s2_profiles_boot_into_an_environment_the_admission_takes(self):
        """The image's ENV, then the profile (the contract's own apply_environment): exactly what the attach's
        packed_any_admission.check_environment reads - the M3 shape at max-num-seqs, ANY_REQUEST, eight-row
        groups, the tree-scratch patch, tail without extent - and the runtime pin names K64j. Under c2 the
        same boot fails the shape: it builds no block."""
        import packed_any_admission
        import serving_runtime

        with open(DOCKERFILE, encoding='utf-8') as handle:
            joined = handle.read().replace(chr(92) + chr(10), ' ')
        # c2_image_provenance.dockerfile_env's parse (that module is not in the image this test also runs in).
        image = dict(token.partition('=')[::2] for line in joined.split(chr(10)) if line.startswith('ENV ')
                     for token in line[4:].split() if '=' in token)
        for name, admitted in (('c2-packed', True), ('c2-packed-gate', True), ('c2', False)):
            profile = self.load(name)
            env = contract.apply_environment(profile, dict(image))
            m3 = serving_runtime.m3_shape(dict(scheduler_requests=profile['engine']['max-num-seqs']), env)
            problems = packed_any_admission.check_environment(env, m3)
            with self.subTest(profile=name):
                if admitted:
                    self.assertEqual(problems, [])
                    self.assertTrue(packed_any_admission.extent_replay_enabled(env))
                    self.assertEqual(env[packed_any_admission.RUNTIME_BINARY_ENV],
                                     packed_any_admission.K64J_TTNNCPP_SHA256)
                else:
                    self.assertEqual(len(problems), 1, problems)
                    self.assertIn('PACKED_STEP=0', problems[0])
                    self.assertFalse(packed_any_admission.extent_replay_enabled(env))


class GeneralProfileTest(unittest.TestCase):
    def test_general_turns_the_fast_path_off(self):
        profile = contract.load_profile(PROFILES, 'general')
        argv = contract.engine_arguments(profile, '/snap')
        self.assertNotIn('--speculative-config', argv)
        additional = json.loads(argv[argv.index('--additional-config') + 1])
        self.assertNotIn('qwen_fast_t16', additional)
        self.assertIs(profile['request_contract'], False)
        environ = contract.apply_environment(profile, {'QWEN36_BATCHED_DECODE_MODE': 'host'})
        self.assertEqual(environ['QWEN36_BATCHED_DECODE_MODE'], 'host')


class ArgvTest(unittest.TestCase):
    def test_platform_engine_flags_are_replaced_and_its_own_kept(self):
        profile = contract.load_profile(PROFILES, 'coding')
        platform = ['-m', '--model', 'Qwen/Qwen3.8-27B', '--served-model-name', 'Qwen/Qwen3.8-27B',
                    '--port', '8001', '--max-model-len', '65536', '--max_num_seqs=2', '--block-size', '64',
                    '--no-enable-prefix-caching', '--reasoning-parser', 'qwen3', '--tool-call-parser', 'qwen3_xml',
                    '--enable-auto-tool-choice', '--additional-config', '{"tt": {"fabric_config": "FABRIC_1D"}}']
        argv = contract.rewrite_argv(platform, profile, '/snap')
        self.assertEqual(argv[0], '-m')
        for kept in ('--served-model-name', '--port', '--reasoning-parser', '--tool-call-parser',
                     '--enable-auto-tool-choice'):
            self.assertEqual(argv.count(kept), 1, kept)
        self.assertEqual(argv[argv.index('--served-model-name') + 1], 'Qwen/Qwen3.8-27B')
        self.assertEqual(argv.count('--model'), 1)
        self.assertEqual(argv[argv.index('--model') + 1], '/snap')
        self.assertEqual(argv.count('--max-model-len'), 1)
        self.assertEqual(argv[argv.index('--max-model-len') + 1], '131328')
        self.assertNotIn('--max_num_seqs=2', argv)
        self.assertEqual(argv.count('--additional-config'), 1)
        self.assertNotIn('FABRIC_1D', ' '.join(argv))
        self.assertNotIn('65536', argv)

    def test_api_server_detection(self):
        self.assertTrue(contract.is_api_server(['/opt/venv/bin/python3', '-m', contract.API_SERVER, '--port', '1']))
        self.assertFalse(contract.is_api_server(['/opt/venv/bin/python3', '-m', 'serving.server']))
        self.assertFalse(contract.is_api_server(None))


class PathAndEnvironmentTest(unittest.TestCase):
    def test_fast_tree_goes_after_the_thatch_runtime(self):
        path = contract.fix_sys_path(['', '/opt/thatch/py', '/usr/lib/python310.zip', '/opt/venv/lib/site-packages'])
        self.assertEqual(path[:6], ['', '/opt/thatch/py'] + list(contract.FAST_PATHS))
        self.assertEqual(contract.fix_sys_path(list(path)), path)

    def test_environment_overrides_the_agent(self):
        profile = contract.load_profile(PROFILES, 'coding')
        environ = contract.apply_environment(profile, {
            'TT_MESH_GRAPH_DESC_PATH': '/opt/tt-metal/p300_mesh_graph_descriptor.textproto',
            'QWEN36_BATCHED_DECODE_MODE': 'host'})
        self.assertTrue(environ['TT_MESH_GRAPH_DESC_PATH'].endswith('p150_x2_mesh_graph_descriptor.textproto'))
        self.assertNotIn('QWEN36_BATCHED_DECODE_MODE', environ)
        self.assertEqual(environ['QWEN_FAST_OUTPUT_BUDGET'], '4096')

    def test_snapshot_must_be_mounted(self):
        profile = contract.load_profile(PROFILES, 'coding')
        self.assertEqual(contract.resolve_snapshot(profile, exists=lambda path: '/hub/' in path),
                         profile['snapshots'][1])
        with self.assertRaises(ValueError):
            contract.resolve_snapshot(profile, exists=lambda path: False)


class RequestTest(unittest.TestCase):
    def test_sampling_is_coerced_to_greedy_and_max_tokens_clamped(self):
        params = enforce(Params())
        self.assertEqual((params.temperature, params.top_p, params.top_k, params.min_p), (0.0, 1.0, 0, 0.0))
        self.assertEqual(params.repetition_penalty, 1.0)
        self.assertIsNone(params.seed)
        self.assertEqual(params.max_tokens, 4096)
        self.assertEqual(enforce(Params(max_tokens=100)).max_tokens, 100)
        self.assertEqual(enforce(Params(max_tokens=100000), prompt_tokens=61000).max_tokens, 4096)
        self.assertEqual(enforce(Params(max_tokens=None), prompt_tokens=None).max_tokens, 4096)

    def test_prompt_limits(self):
        enforce(Params(), prompt_tokens=61440)
        with self.assertRaises(contract.ContractError):
            enforce(Params(), prompt_tokens=61441)
        enforce(Params(), prompt_tokens=65792 - 4096, max_model_len=65792, max_prompt_tokens=None)
        with self.assertRaises(contract.ContractError):
            enforce(Params(), prompt_tokens=65792 - 4095, max_model_len=65792, max_prompt_tokens=None)

    def test_the_prompt_cap_can_rise_to_leave_only_the_minimum_answer_room(self):
        """room = max_model_len - budget blocked a 123,136-token cap under a 16,384 ceiling
        (114,944 was the most); with min_answer_tokens the cap is what leaves 8,192."""
        c2 = dict(max_model_len=131328, budget=16384, max_prompt_tokens=123136)
        self.assertEqual(contract.prompt_room(131328, 16384, 123136), 131328 - 16384, 'the old formula')
        enforce(Params(), prompt_tokens=123136, min_answer_tokens=8192, **c2)
        with self.assertRaisesRegex(contract.ContractError, 'at least 8192 tokens of answer room'):
            enforce(Params(), prompt_tokens=123137, min_answer_tokens=8192, **c2)
        # max_prompt_tokens still lowers it, never raises it past the answer room
        self.assertEqual(contract.prompt_room(131328, 16384, 100000, 8192), 100000)
        self.assertEqual(contract.prompt_room(131328, 16384, 130000, 8192), 123136)
        self.assertEqual(contract.prompt_room(131328, 16384, None, 8192), 123136)
        for bad in (0, 16385, 8192.0, '8192'):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'min_answer_tokens'):
                contract.prompt_room(131328, 16384, None, bad)

    def test_max_tokens_is_clamped_to_the_room_left_and_never_below_the_minimum_answer(self):
        c2 = dict(max_model_len=131328, budget=16384, max_prompt_tokens=123136, min_answer_tokens=8192)
        self.assertEqual(enforce(Params(max_tokens=16384), prompt_tokens=123136, **c2).max_tokens, 8192)
        self.assertEqual(enforce(Params(max_tokens=16384), prompt_tokens=60, **c2).max_tokens, 16384)
        self.assertEqual(enforce(Params(max_tokens=100000), prompt_tokens=60, **c2).max_tokens, 16384)
        self.assertEqual(enforce(Params(max_tokens=10000), prompt_tokens=60, **c2).max_tokens, 10000)
        self.assertEqual(enforce(Params(max_tokens=1), prompt_tokens=60, **c2).max_tokens, 1)
        for prompt in (1, 60, 2048, 65536, 114945, 123136):
            limit = enforce(Params(max_tokens=100000), prompt_tokens=prompt, **c2).max_tokens
            self.assertGreaterEqual(limit, 8192, prompt)
            self.assertLessEqual(prompt + limit, 131328, prompt)

    def test_an_omitted_max_tokens_defaults_to_the_profiles_value_within_the_clamp(self):
        c2 = dict(max_model_len=131328, budget=16384, max_prompt_tokens=123136, min_answer_tokens=8192,
                  default_max_tokens=8192)
        # the engine API leaves it None; vLLM's OpenAI server fills max_model_len - prompt
        self.assertEqual(enforce(Params(max_tokens=None), prompt_tokens=60, **c2).max_tokens, 8192)
        self.assertEqual(enforce(Params(max_tokens=131328 - 60), prompt_tokens=60, **c2).max_tokens, 8192)
        self.assertEqual(enforce(Params(max_tokens=None), prompt_tokens=None, **c2).max_tokens, 8192)
        # an explicit value is the client's, within the clamp
        self.assertEqual(enforce(Params(max_tokens=12000), prompt_tokens=60, **c2).max_tokens, 12000)
        self.assertEqual(enforce(Params(max_tokens=131328 - 61), prompt_tokens=60, **c2).max_tokens, 16384)
        # without a default nothing changes: omitted means the clamp, as before
        self.assertEqual(enforce(Params(max_tokens=None), prompt_tokens=60, max_model_len=131328, budget=16384,
                                 max_prompt_tokens=123136, min_answer_tokens=8192).max_tokens, 16384)
        self.assertTrue(contract.omitted_max_tokens(None, prompt_tokens=5, max_model_len=10))
        self.assertTrue(contract.omitted_max_tokens(5, prompt_tokens=5, max_model_len=10))
        self.assertFalse(contract.omitted_max_tokens(4, prompt_tokens=5, max_model_len=10))
        self.assertFalse(contract.omitted_max_tokens(5, prompt_tokens=None, max_model_len=10))

    def test_request_limits_refuse_a_default_outside_the_budget(self):
        profile = contract.load_profile(PROFILES, 'c2')
        for bad in (0, 16385, '8192', 8192.0):
            broken = dict(profile, default_max_tokens=bad)
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'default_max_tokens'):
                contract.request_limits(broken)
        with self.assertRaisesRegex(ValueError, 'min_answer_tokens'):
            contract.request_limits(dict(profile, min_answer_tokens=20000))
        coding = contract.request_limits(contract.load_profile(PROFILES, 'coding'))
        self.assertEqual(coding, dict(budget=4096, max_prompt_tokens=61440, min_answer_tokens=None,
                                      default_max_tokens=None))

    def test_the_installed_contract_applies_the_c2_limits(self):
        calls = []

        class InputProcessor(object):
            model_config = types.SimpleNamespace(max_model_len=131328)

            def process_inputs(self, request_id, prompt, params, *args, **kwargs):
                calls.append((request_id, params.max_tokens))
                return 'request'

        module = types.ModuleType('fake_input_processor_c2')
        module.InputProcessor = InputProcessor
        contract.install_request_contract(module, budget=16384, eos_ids=frozenset((248046,)), max_prompt_tokens=123136,
                                          min_answer_tokens=8192, default_max_tokens=8192)
        InputProcessor().process_inputs('r1', {'prompt_token_ids': [1] * 123136}, Params(max_tokens=16384))
        InputProcessor().process_inputs('r2', {'prompt_token_ids': [1] * 60}, Params(max_tokens=131328 - 60))
        InputProcessor().process_inputs('r3', {'prompt_token_ids': [1] * 60}, Params(max_tokens=12000))
        self.assertEqual(calls, [('r1', 8192), ('r2', 8192), ('r3', 12000)])
        with self.assertRaises(contract.ContractError):
            InputProcessor().process_inputs('r4', {'prompt_token_ids': [1] * 123137}, Params())

    def test_what_the_fast_path_cannot_serve_is_refused(self):
        for values in (dict(n=2), dict(logprobs=1), dict(prompt_logprobs=0), dict(structured_outputs=object()),
                       dict(logit_bias={1: 1.0}), dict(allowed_token_ids=[1]), dict(bad_words=['x']),
                       dict(stop=['\n\n']), dict(min_tokens=5), dict(stop_token_ids=[5])):
            with self.assertRaises(contract.ContractError, msg=repr(values)):
                enforce(Params(**values))
        enforce(Params(stop_token_ids=[248046]))

    def test_contract_wraps_process_inputs_once(self):
        calls = []

        class InputProcessor(object):
            model_config = types.SimpleNamespace(max_model_len=131328)

            def process_inputs(self, request_id, prompt, params, *args, **kwargs):
                calls.append((request_id, params.temperature, params.max_tokens, args, kwargs))
                return 'request'

        module = types.ModuleType('fake_input_processor')
        module.InputProcessor = InputProcessor
        contract.install_request_contract(module, budget=4096, eos_ids=frozenset((248046,)), max_prompt_tokens=61440)
        contract.install_request_contract(module, budget=4096, eos_ids=frozenset((248046,)), max_prompt_tokens=61440)
        result = InputProcessor().process_inputs('r1', {'type': 'token', 'prompt_token_ids': [1] * 1000},
                                                 Params(), ('generate',), arrival_time=1.0)
        self.assertEqual(result, 'request')
        self.assertEqual(calls, [('r1', 0.0, 4096, (('generate',),), {'arrival_time': 1.0})])
        with self.assertRaises(contract.ContractError):
            InputProcessor().process_inputs('r2', {'prompt_token_ids': [1] * 10}, Params(n=3), ())
        with self.assertRaises(contract.ContractError):
            InputProcessor().process_inputs('r3', {'prompt_token_ids': [1] * 61441}, Params(), ())

    def test_prompt_length_reads_token_prompts(self):
        self.assertEqual(contract.prompt_length({'type': 'token', 'prompt_token_ids': [1, 2]}), 2)
        self.assertEqual(contract.prompt_length({'decoder': {'prompt_token_ids': [1]}}), 1)
        self.assertIsNone(contract.prompt_length('text'))


class PostImportHookTest(unittest.TestCase):
    def test_callback_runs_after_the_module_executes(self):
        directory = tempfile.mkdtemp()
        with open(os.path.join(directory, 'c2_hook_target.py'), 'w') as handle:
            handle.write('VALUE = 1\n')
        seen = []
        sys.path.insert(0, directory)
        hook = contract.PostImportHook('c2_hook_target', lambda module: seen.append(module.VALUE))
        sys.meta_path.insert(0, hook)
        try:
            import c2_hook_target  # noqa: F401
        finally:
            sys.path.remove(directory)
            if hook in sys.meta_path:
                sys.meta_path.remove(hook)
            sys.modules.pop('c2_hook_target', None)
        self.assertEqual(seen, [1])
        self.assertNotIn(hook, sys.meta_path)


class TeardownSkipTest(unittest.TestCase):
    def test_only_an_engine_child_with_ttnn_exits_early(self):
        exits = []
        self.assertFalse(contract.exit_without_device_teardown(modules={}, parent=object(), exit=exits.append))
        self.assertFalse(contract.exit_without_device_teardown(modules={'ttnn': 1}, parent=None, exit=exits.append,
                                                               streams=()))
        self.assertEqual(exits, [])
        self.assertTrue(contract.exit_without_device_teardown(modules={'ttnn': 1}, parent=object(),
                                                              exit=exits.append, streams=()))
        self.assertEqual(exits, [0])


class BootTest(unittest.TestCase):
    def test_off_unless_enabled(self):
        self.assertIsNone(contract.boot(environ={}, orig_argv=['python3', '-m', contract.API_SERVER]))


TP4_PROFILES = ('general-tp4', 'general-prefix-tp4', 'general-tp4-131k', 'general-prefix-tp4-131k', 'general-tp4-bench',
                'general-tp4-mmrs', 'general-tp4-ring-mmrs')
# The S2 window's G1 defaults arm: the ring fabric and the fused prefill out-projection on together.
RING_FABRIC_PROFILES = ('general-tp4-ring-mmrs', 'c2-packed-tp4-gate-ring')
MMRS_PROFILES = ('general-tp4-mmrs', 'general-tp4-ring-mmrs')
# The four-card fast-path (S2) profiles: mesh_device P150x4 with the fast path on, under QWEN_FAST_TP=4.
FAST_TP4_PROFILES = ('c2-packed-tp4', 'c2-packed-tp4-262k-gate', 'c2-packed-tp4-8', 'c2-packed-tp4-8-best', 'c2-packed-tp4-8-best-quad', 'c2-packed-tp4-8-best-quad-dbf16', 'c2-packed-tp4-8-best-quad-gate', 'c2-packed-tp4-8-diag-strace', 'c2-packed-tp4-8-diag-strace-nowarm', 'c2-packed-tp4-8-diag-strace-rshard', 'c2-packed-tp4-8-gate', 'c2-packed-tp4-8-time-gate', 'c2-packed-tp4-8x262k', 'c2-packed-tp4-8x262k-best', 'c2-packed-tp4-8x262k-best-audit', 'c2-packed-tp4-8x262k-best-levern-audit', 'c2-packed-tp4-8x262k-best-levern-control-audit', 'c2-packed-tp4-8x262k-best-levern-final-hold-time-gate', 'c2-packed-tp4-8x262k-best-levern-foreign-time-gate', 'c2-packed-tp4-8x262k-best-levern-hang-gate', 'c2-packed-tp4-8x262k-best-levern-r1-time-gate', 'c2-packed-tp4-8x262k-best-levern-time-gate', 'c2-packed-tp4-8x262k-best-nosamp-audit', 'c2-packed-tp4-8x262k-best-sdpamulti', 'c2-packed-tp4-8x262k-best-sdpamulti-audit', 'c2-packed-tp4-8x262k-best-stack-audit', 'c2-packed-tp4-8x262k-best-time-gate', 'c2-packed-tp4-8x262k-best-time-gate-d2', 'c2-packed-tp4-8x262k-best-time-gate-dbf16', 'c2-packed-tp4-8x262k-best-time-gate-lookup', 'c2-packed-tp4-8x262k-best-time-gate-nosamp', 'c2-packed-tp4-8x262k-best-time-gate-s1', 'c2-packed-tp4-8x262k-best-time-gate-stack', 'c2-packed-tp4-8x262k-best-time-gate-u1', 'c2-packed-tp4-8x262k-best-u1-audit', 'c2-packed-tp4-8x262k-diag-strace', 'c2-packed-tp4-8x262k-gate', 'c2-packed-tp4-8x262k-hostgap-1', 'c2-packed-tp4-8x262k-hostgap-1-audit', 'c2-packed-tp4-8x262k-hostgap-2', 'c2-packed-tp4-8x262k-hostgap-2-audit', 'c2-packed-tp4-8x262k-prefix-gate', 'c2-packed-tp4-8x262k-prefix-time-gate', 'c2-packed-tp4-8x262k-ship', 'c2-packed-tp4-8x262k-ship-prefix', 'c2-packed-tp4-8x262k-ship-prefix-audit', 'c2-packed-tp4-8x262k-ship-prefix-audit-digests', 'c2-packed-tp4-8x262k-ship-prefix-dbf16', 'c2-packed-tp4-8x262k-ship-prefix-dckdefault', 'c2-packed-tp4-8x262k-ship-prefix-levern', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-traffic', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-f1', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-ln', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-sdpa', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-w1', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-pool', 'c2-packed-tp4-8x262k-ship-prefix-w2', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit-pool', 'c2-packed-tp4-8x262k-time-gate', 'c2-packed-tp4-8x262k-w1', 'c2-packed-tp4-8x262k-w1-audit', 'c2-packed-tp4-8x262k-w1-audit-nod1', 'c2-packed-tp4-8x262k-w1-lite', 'c2-packed-tp4-8x262k-w1-nod1', 'c2-packed-tp4-8x262k-w2', 'c2-packed-tp4-8x262k-w2-audit', 'c2-packed-tp4-8x262k-w2-nof1', 'c2-packed-tp4-8x262k-w2-nof1-audit', 'c2-packed-tp4-best', 'c2-packed-tp4-best-d2', 'c2-packed-tp4-best-dbf16', 'c2-packed-tp4-best-gate', 'c2-packed-tp4-best-gate-d2', 'c2-packed-tp4-best-gate-dbf16', 'c2-packed-tp4-best-gate-glue', 'c2-packed-tp4-best-gate-lookup', 'c2-packed-tp4-best-gate-samp', 'c2-packed-tp4-best-gate-tpub', 'c2-packed-tp4-best-lookup', 'c2-packed-tp4-best-rshard', 'c2-packed-tp4-best-samp', 'c2-packed-tp4-best-sdpa', 'c2-packed-tp4-best-ship', 'c2-packed-tp4-best-ship-glue', 'c2-packed-tp4-best-ship-tpub', 'c2-packed-tp4-best-ship-warm4', 'c2-packed-tp4-best-strace', 'c2-packed-tp4-best-strace-glue', 'c2-packed-tp4-best-strace-tpub', 'c2-packed-tp4-best-v5', 'c2-packed-tp4-best-v5-gate', 'c2-packed-tp4-diag',
                     'c2-packed-tp4-diag-rshard', 'c2-packed-tp4-diag-sprewarm', 'c2-packed-tp4-diag-strace', 'c2-packed-tp4-diag-t1', 'c2-packed-tp4-diag-t1-rshard-audit', 'c2-packed-tp4-diag-t2',
                     'c2-packed-tp4-f12', 'c2-packed-tp4-f2', 'c2-packed-tp4-gate', 'c2-packed-tp4-gate-bf16', 'c2-packed-tp4-gate-draftwide',
                     'c2-packed-tp4-gate-fcommit', 'c2-packed-tp4-gate-fcommit-live', 'c2-packed-tp4-gate-fcommit-quad',
                     'c2-packed-tp4-gate-noslide', 'c2-packed-tp4-gate-pairs', 'c2-packed-tp4-gate-pairslice', 'c2-packed-tp4-gate-quad',
                     'c2-packed-tp4-gate-ring', 'c2-packed-tp4-gate-rshard-audit', 'c2-packed-tp4-gate-vglue', 'c2-packed-tp4-lanes-gate',
                     'c2-packed-tp4-lanes-time-gate', 'c2-packed-tp4-solo-gate', 'c2-packed-tp4-solo-time-gate',
                     'c2-packed-tp4-speed', 'c2-packed-tp4-speed-fcommit', 'c2-packed-tp4-speed-fcommit-oop',
                     'c2-packed-tp4-speed-fcommit-quad', 'c2-packed-tp4-speed-fix', 'c2-packed-tp4-speed-noslide',
                     'c2-packed-tp4-speed-pairs', 'c2-packed-tp4-speed-quad', 'c2-packed-tp4-speed-rshard', 'c2-packed-tp4-speed-sprewarm',
                     'c2-packed-tp4-speed-strace', 'c2-packed-tp4-speed-strace-dispatchdiag', 'c2-packed-tp4-speed-strace-draftwide', 'c2-packed-tp4-speed-strace-fcommit', 'c2-packed-tp4-speed-strace-pairslice', 'c2-packed-tp4-speed-strace-ring', 'c2-packed-tp4-speed-strace-v5', 'c2-packed-tp4-speed-vglue', 'c2-packed-tp4-speed-vglue-c1a',
                     'c2-packed-tp4-speed-vglue-v1', 'c2-packed-tp4-speed-vglue-v2', 'c2-packed-tp4-speed-vglue-v3a',
                     'c2-packed-tp4-speed-vglue-v4a', 'c2-packed-tp4-speed-warm4', 'c2-packed-tp4-time-gate', 'c2-packed-tp4-warm4-control', 'c2-packed-tp4-warm4-diag', 'c2-packed-tp4-warm4-diag-oldtail',
                          'c2-packed-tp4-warm4-even-diag', 'c2-packed-tp4-warm4-gate')
# The bring-up switches every TP4 profile carries in its env (see the profiles' descriptions): the fused prefill
# out-projection off (general-tp4-mmrs is the arm that turns it on) and the prefill conv audited for four chunks.
TP4_BRINGUP_ENV = {'QWEN_GDN_PREFILL_MMRS': '0', 'QWEN_FAST_GDN_PREFILL_CONV_AUDIT': '4'}
# KV bytes per token per chip at TP4: 16 attention layers x (K, V) x one KV head x 256 x 1.0625 B (bf8).
TP4_KV_BYTES_PER_TOKEN = 16 * 2 * 1 * 256 * 17 // 16


def document():
    with open(PROFILES, encoding='utf-8') as handle:
        return json.load(handle)


class MeshTest(unittest.TestCase):
    """Item 8: the pair's profiles name no mesh and serve as before; the TP4 family opens all four cards."""

    def test_every_profile_opens_a_mesh_its_descriptor_and_sampling_agree_with(self):
        for name in document()['profiles']:
            profile = contract.load_profile(PROFILES, name)
            self.assertEqual(contract.mesh_problems(profile), [], name)

    def test_the_pairs_profiles_are_untouched(self):
        for name, profile in document()['profiles'].items():
            if name in TP4_PROFILES or name in FAST_TP4_PROFILES or name == 'general-2link' or name in PARKED_TWINS:
                continue
            self.assertNotIn('mesh_device', profile, name)
            self.assertEqual(profile['mesh_graph_descriptor'], contract.PAIR_DESCRIPTOR, name)
            tt = profile['engine']['additional-config']['tt']
            self.assertNotIn('sample_on_device_mode', tt, name)
            self.assertNotIn('fabric_config', tt, name)
            self.assertFalse(contract.ring_mesh(profile), name)
            environ = contract.apply_environment(dict(profile, name=name), {'MESH_DEVICE': 'P300'})
            self.assertEqual(environ['MESH_DEVICE'], 'P300', 'a pair profile leaves MESH_DEVICE as it found it')

    def test_the_two_link_pair_is_general_under_a_two_channel_descriptor(self):
        import tp4_mesh

        self.assertEqual(contract.PAIR_2LINK_DESCRIPTOR, tp4_mesh.PAIR_DESCRIPTOR_PATH)
        self.assertEqual(contract.MESHES['P300']['shape'], (1, 2))
        mine, general = contract.load_profile(PROFILES, 'general-2link'), contract.load_profile(PROFILES, 'general')
        self.assertEqual(mine['mesh_device'], 'P300')
        self.assertEqual(mine['mesh_graph_descriptor'], contract.PAIR_2LINK_DESCRIPTOR)
        for key in ('engine', 'env', 'eos_ids', 'snapshots', 'request_contract', 'drop_batched_decode_mode'):
            self.assertEqual(mine[key], general[key], key)
        self.assertEqual(contract.mesh_problems(mine), [])
        self.assertFalse(contract.ring_mesh(mine))
        environ = contract.apply_environment(mine, {'MESH_DEVICE': 'P300', 'TT_MESH_GRAPH_DESC_PATH': 'p300'})
        self.assertEqual((environ['MESH_DEVICE'], environ['TT_MESH_GRAPH_DESC_PATH']),
                         ('P300', contract.PAIR_2LINK_DESCRIPTOR))

    def test_the_fast_path_is_refused_under_the_two_link_pair(self):
        fast = json.loads(json.dumps(contract.load_profile(PROFILES, 'general-2link')))
        fast['engine']['additional-config']['qwen_fast_t16'] = True
        problems = contract.mesh_problems(fast)
        self.assertTrue(any('fast path' in problem and 'p150_x2' in problem for problem in problems), problems)
        pinned = contract.load_profile(PROFILES, 'c2')
        self.assertEqual(contract.mesh_problems(pinned), [], 'the fast path stays on the four-channel pair')

    def test_the_tp4_family_opens_the_four_card_ring(self):
        import tp4_mesh

        self.assertEqual(contract.RING_DESCRIPTOR, tp4_mesh.DESCRIPTOR_PATH)
        self.assertEqual(contract.MESHES['P150x4']['shape'], tp4_mesh.MESH_SHAPE)
        for name in TP4_PROFILES:
            profile = contract.load_profile(PROFILES, name)
            self.assertEqual(profile['mesh_device'], tp4_mesh.MESH_DEVICE, name)
            self.assertEqual(profile['mesh_graph_descriptor'], tp4_mesh.DESCRIPTOR_PATH, name)
            tt = contract.tt_config(profile)
            self.assertEqual(tt['fabric_config'], 'FABRIC_1D_RING' if name in RING_FABRIC_PROFILES else tp4_mesh.FABRIC_CONFIG, name)
            self.assertEqual(tt['sample_on_device_mode'], 'decode_only', name)
            self.assertTrue(contract.ring_mesh(profile), name)
            self.assertNotIn('qwen_fast_t16', profile['engine']['additional-config'], name)
            self.assertIs(profile['request_contract'], False, name)
            environ = contract.apply_environment(profile, {'MESH_DEVICE': 'P300', 'TT_MESH_GRAPH_DESC_PATH': 'p300'})
            self.assertEqual((environ['MESH_DEVICE'], environ['TT_MESH_GRAPH_DESC_PATH']),
                             ('P150x4', tp4_mesh.DESCRIPTOR_PATH), name)

    def test_each_tp4_profile_is_its_pair_twin_at_its_own_seats_and_context(self):
        twins = {'general-tp4': 'general', 'general-prefix-tp4': 'general-prefix', 'general-tp4-131k': 'general',
                 'general-prefix-tp4-131k': 'general-prefix', 'general-tp4-bench': 'general',
                 'general-tp4-mmrs': 'general', 'general-tp4-ring-mmrs': 'general'}
        sized = ('max-model-len', 'max-num-batched-tokens', 'max-num-seqs')
        for name, twin in twins.items():
            mine, theirs = contract.load_profile(PROFILES, name), contract.load_profile(PROFILES, twin)
            self.assertEqual({key: value for key, value in mine['engine'].items() if key not in sized + ('additional-config',)},
                             {key: value for key, value in theirs['engine'].items() if key not in sized + ('additional-config',)},
                             name)
            tt = dict(contract.tt_config(mine))
            self.assertEqual((tt.pop('fabric_config'), tt.pop('sample_on_device_mode')),
                             ('FABRIC_1D_RING' if name in RING_FABRIC_PROFILES else 'FABRIC_1D', 'decode_only'))
            self.assertEqual(tt, contract.tt_config(theirs), name)
            context = mine['engine']['max-model-len']
            self.assertEqual(mine['engine']['max-num-batched-tokens'], context, 'whole-prompt prefill')
            expected_env = dict(theirs['env'], QWEN_FAST_MAX_POSITION=str(context), QWEN_DSPARK_REQUEST_CONTEXT=str(context),
                                **dict(TP4_BRINGUP_ENV, **({'QWEN_GDN_PREFILL_MMRS': '1'} if name in MMRS_PROFILES else {})))
            self.assertEqual(mine['env'], expected_env, name)
            for key in ('eos_ids', 'snapshots', 'request_contract', 'drop_batched_decode_mode'):
                self.assertEqual(mine.get(key), theirs.get(key), (name, key))
            self.assertEqual(contract.prefix_reuse(mine), contract.prefix_reuse(theirs), name)
            self.assertEqual(contract.prefix_reuse_problems(mine), [], name)

    def test_the_pools_follow_the_dram_arithmetic(self):
        pools = {}
        for name in TP4_PROFILES:
            engine = contract.load_profile(PROFILES, name)['engine']
            pools[name] = engine['max-model-len'] * engine['max-num-seqs']
        serving = [name for name in TP4_PROFILES if name != 'general-tp4-bench']
        for gated in MMRS_PROFILES:
            self.assertIs(contract.load_profile(PROFILES, gated).get('gate_only'), True, gated)
        for name in serving:
            self.assertEqual(pools[name], 524288, name)
            # the same KV per chip as general's 4 x 65,536 on the pair (17,408 B per token per chip there)
            self.assertEqual(pools[name] * TP4_KV_BYTES_PER_TOKEN, 4 * 65536 * 17408, name)
        self.assertEqual(TP4_KV_BYTES_PER_TOKEN, 8704)
        self.assertEqual(pools['general-tp4-bench'], 8 * 131072)
        self.assertLess(pools['general-tp4-bench'] * TP4_KV_BYTES_PER_TOKEN, 9.2e9)
        self.assertIs(contract.load_profile(PROFILES, 'general-tp4-bench').get('gate_only'), True,
                      'the 9.13 GB-per-chip pool is unmeasured: gate only until G5 at TP4')
        seats = {name: contract.load_profile(PROFILES, name)['engine']['max-num-seqs'] for name in TP4_PROFILES}
        self.assertEqual(seats, {'general-tp4': 8, 'general-prefix-tp4': 8, 'general-tp4-131k': 4,
                                 'general-prefix-tp4-131k': 4, 'general-tp4-bench': 8,
                                 'general-tp4-mmrs': 8, 'general-tp4-ring-mmrs': 8})

    def test_what_a_mesh_cannot_serve_is_refused(self):
        ring = contract.load_profile(PROFILES, 'general-tp4')
        fast = dict(contract.load_profile(PROFILES, 'c2'), mesh_device='P150x4',
                    mesh_graph_descriptor=contract.RING_DESCRIPTOR)
        self.assertTrue(any('fast path' in problem for problem in contract.mesh_problems(fast)))
        pair_sampling = json.loads(json.dumps(contract.load_profile(PROFILES, 'general')))
        pair_sampling['engine']['additional-config']['tt']['sample_on_device_mode'] = 'decode_only'
        self.assertEqual(contract.mesh_problems(pair_sampling),
                         ['on-device sampling needs at most 65536 logits per device; a (1, 2) mesh has 124160'])
        wrong_descriptor = dict(ring, mesh_graph_descriptor=contract.PAIR_DESCRIPTOR)
        self.assertTrue(contract.mesh_problems(wrong_descriptor)[0].startswith('mesh P150x4 needs the descriptor'))
        pair_on_ring = dict(contract.load_profile(PROFILES, 'general'), mesh_graph_descriptor=contract.RING_DESCRIPTOR)
        self.assertTrue(contract.mesh_problems(pair_on_ring)[0].startswith('mesh P300 (the pair) needs'))
        self.assertEqual(contract.mesh_problems(dict(ring, mesh_device='P150x8')),
                         ["mesh_device 'P150x8' is not one of P150x4, P300"])
        env_mesh = dict(ring, env=dict(ring['env'], MESH_DEVICE='P150x4'))
        self.assertIn("MESH_DEVICE is the profile's mesh_device, never an env value", contract.mesh_problems(env_mesh))
        bad_mode = json.loads(json.dumps(ring))
        bad_mode['engine']['additional-config']['tt']['sample_on_device_mode'] = 'prefill'
        self.assertTrue(any('not one of' in problem for problem in contract.mesh_problems(bad_mode)))

    def test_host_sampling_drops_device_sampling_and_nothing_else(self):
        ring = contract.load_profile(PROFILES, 'general-tp4')
        self.assertFalse(contract.host_sampling_forced({}, exists=lambda path: False))
        self.assertTrue(contract.host_sampling_forced({contract.HOST_SAMPLING_ENV: '1'}, exists=lambda path: False))
        self.assertTrue(contract.host_sampling_forced({}, exists=lambda path: path == contract.HOST_SAMPLING_FILE))
        host = contract.without_device_sampling(ring)
        self.assertNotIn('sample_on_device_mode', contract.tt_config(host))
        self.assertIn('sample_on_device_mode', contract.tt_config(ring), 'the input profile is kept')
        before, after = contract.engine_arguments(ring, '/snap'), contract.engine_arguments(host, '/snap')
        self.assertEqual([token for token in before if not token.startswith('{')],
                         [token for token in after if not token.startswith('{')])
        additional = json.loads(after[after.index('--additional-config') + 1])
        self.assertEqual(additional['tt'], {'trace_region_size': 1073741824, 'l1_small_size': 24576,
                                            'fabric_config': 'FABRIC_1D'})

    def boot(self, name, environ, argv):
        """contract.boot under a TP4 profile in the API server, with the process-wide effects patched out."""
        import unittest.mock as mock

        environ = dict(environ, QWEN_C2_SERVING='1', QWEN_C2_PROFILE=name, QWEN_C2_PROFILES=PROFILES)
        before = list(sys.meta_path)
        saved_argv = list(sys.argv)
        sys.argv[:] = list(argv)
        try:
            # load_profile reads the profile NAME from the process environment (the image's boot passes none).
            with mock.patch.dict(os.environ, {'QWEN_C2_PROFILE': name}),                     mock.patch.object(contract, 'fix_sys_path'), mock.patch.object(contract, 'install_teardown_skip'), \
                    mock.patch.object(contract, 'resolve_snapshot', return_value='/snap'), \
                    mock.patch.object(contract, 'install_prefix_metrics'), \
                    mock.patch.object(contract, 'read_salt_key', return_value=(None, 'none')), \
                    mock.patch.object(contract, 'log'):
                contract.boot(environ=environ, orig_argv=['python3', '-m', contract.API_SERVER])
            hooks = [hook for hook in sys.meta_path if hook not in before]
            return list(sys.argv), environ, hooks
        finally:
            sys.argv[:] = saved_argv
            sys.meta_path[:] = before

    def test_boot_serves_the_ring_samples_on_device_and_arms_the_ring_check(self):
        platform = ['api_server', '--model', 'Qwen/Qwen3.8-27B', '--port', '8000', '--max-model-len', '65536',
                    '--max-num-seqs', '2', '--additional-config', '{"tt": {"fabric_config": "FABRIC_1D"}}']
        with mock_exists(False):
            argv, environ, hooks = self.boot('general-tp4', {'MESH_DEVICE': 'P300'}, platform)
        self.assertEqual(environ['MESH_DEVICE'], 'P150x4')
        self.assertEqual(argv[argv.index('--max-num-seqs') + 1], '8')
        additional = json.loads(argv[argv.index('--additional-config') + 1])
        self.assertEqual(additional['tt']['sample_on_device_mode'], 'decode_only')
        self.assertEqual([hook.name for hook in hooks if isinstance(hook, contract.PostImportHook)
                          and hook.name == contract.PLUGIN_WORKER], [contract.PLUGIN_WORKER])
        with mock_exists(False):
            argv, _, hooks = self.boot('general-tp4', {'MESH_DEVICE': 'P300', contract.HOST_SAMPLING_ENV: '1',
                                                       contract.RING_CHECK_ENV: '0'}, platform)
        additional = json.loads(argv[argv.index('--additional-config') + 1])
        self.assertNotIn('sample_on_device_mode', additional['tt'])
        self.assertFalse([hook for hook in hooks if getattr(hook, 'name', None) == contract.PLUGIN_WORKER])

    def test_boot_under_a_pair_profile_arms_no_ring_check_and_keeps_its_mesh(self):
        with mock_exists(False):
            argv, environ, hooks = self.boot('general', {'MESH_DEVICE': 'P300'}, ['api_server', '--port', '8000'])
        self.assertEqual(environ['MESH_DEVICE'], 'P300')
        self.assertFalse([hook for hook in hooks if getattr(hook, 'name', None) == contract.PLUGIN_WORKER])
        additional = json.loads(argv[argv.index('--additional-config') + 1])
        self.assertNotIn('sample_on_device_mode', additional['tt'])


def mock_exists(value):
    import unittest.mock as mock

    return mock.patch.object(contract.os.path, 'exists', return_value=value)


if __name__ == '__main__':
    unittest.main()
