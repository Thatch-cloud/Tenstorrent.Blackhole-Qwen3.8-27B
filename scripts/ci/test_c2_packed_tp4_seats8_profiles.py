"""The eight-seat S2 profiles (tp4/seats8): two 64-row M3 blocks (QWEN_FAST_M3_BLOCKS=2), eight seats, 131,328 positions per seat.

Each profile is defined as a delta from an existing one, and these tests hold the delta exact:
  c2-packed-tp4-8              = c2-packed-tp4 + QWEN_FAST_M3_BLOCKS=2, max-num-seqs 8, 16,416 KV blocks, a 512 MiB trace region
  c2-packed-tp4-8-gate         = c2-packed-tp4-8 + both verify audits, the waiver marker, the gate limits, gate_only
  c2-packed-tp4-8-time-gate    = c2-packed-tp4-8 + the waiver marker, the gate limits, gate_only (audits stay off)
  c2-packed-tp4-8-diag-strace  = c2-packed-tp4-8-time-gate + the stall watch (build deadline 150 s), the handle log and the handle guard (log)
  c2-packed-tp4-8-diag-strace-nowarm = the diag arm with QWEN_FAST_M3_REQUEST_WARM=0 (the R2 control)
  c2-packed-tp4-8-diag-strace-rshard = the diag arm + the request-shard argmax arm and its audit (the R3 discriminator)
Every profile with QWEN_FAST_M3_BLOCKS=2 carries QWEN_FAST_M3_REQUEST_WARM=1 (the request widths warm before the block captures) except the control.
The production profile c2-packed-tp4 is untouched, and only the gate-only profiles carry the admission waiver."""

import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import packed_any_admission as admission  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

ROOT = HERE.parent.parent
PROFILES = HERE / 'qwen_c2_profiles.json'
DOCKERFILE = ROOT / 'docker' / 'qwen-c2-serving.Dockerfile'
M3 = (True, 'users=4 FOUR_AS_TWO=0 PACKED_STEP=1')

BASE = 'c2-packed-tp4'
TRAFFIC, GATE, TIME_GATE, DIAG = ('c2-packed-tp4-8', 'c2-packed-tp4-8-gate', 'c2-packed-tp4-8-time-gate',
                                  'c2-packed-tp4-8-diag-strace')
SEATS = 8
WINDOW = 131328
PAGE = 64
BLOCKS = 16416
TRACE_REGION = 536870912

NOWARM, RSHARD = 'c2-packed-tp4-8-diag-strace-nowarm', 'c2-packed-tp4-8-diag-strace-rshard'
TRAFFIC_ENV = {'QWEN_FAST_M3_BLOCKS': '2', 'QWEN_FAST_M3_REQUEST_WARM': '1'}
RSHARD_ENV = {'QWEN_FAST_REQUEST_SHARD_ARGMAX': '1', 'QWEN_FAST_REQUEST_SHARD_AUDIT': '1'}
GATE_MARKER = {'QWEN_C2_GATE_PROFILE': '1'}
AUDITS_ON = {'QWEN_FAST_VERIFY_T1_AUDIT': '1', 'QWEN_FAST_VERIFY_T2_AUDIT': '1'}
DIAG_ENV = {'QWEN_FAST_SEQ_STAGE_LOG': '1', 'QWEN_FAST_TRACE_CENSUS': '1', 'QWEN_FAST_TRACE_CENSUS_GRAPH': '0', 'QWEN_FAST_STALL_DEADLINE_S': '120', 'QWEN_FAST_STALL_BUILD_S': '150', 'QWEN_FAST_CCL_HANDLE_LOG': '1', 'QWEN_FAST_CCL_HANDLE_GUARD': 'log'}
# The production profile, held by digest: adding the eight-seat twins must not move what is served today.
PRODUCTION_ENV_KEYS = {
    'QWEN_FAST_VERIFY_T1_AUDIT': '0', 'QWEN_FAST_VERIFY_T2_AUDIT': '0', 'QWEN_FAST_PACKED_SAMPLER_IN_TRACE': '1',
    'QWEN_FAST_BUDGET_CAP': '1', 'QWEN_FAST_SEQ_DEADLINE_S': '120', 'QWEN_GDN_PREFILL_MMRS': '0'}


def profiles():
    return json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']


def image_env():
    """The C2 image's own ENV (the first ENV of the Dockerfile's final stage), as a dict."""
    text = DOCKERFILE.read_text(encoding='utf-8')
    block = text[text.index('ENV QWEN_ATTN_PREP=1'):]
    blank = chr(10) * 2
    block = block[:block.index(blank)] if blank in block else block
    lines = []
    for line in block.splitlines():
        lines.append(line.rstrip(chr(92)).strip())
        if not line.rstrip().endswith(chr(92)):
            break
    return dict(re.findall(r'(QWEN_[A-Z0-9_]+)=(\S+)', ' '.join(lines)))


def without(profile, *dotted):
    """A deep copy with the named keys removed (a dotted path into nested dicts)."""
    result = copy.deepcopy(profile)
    for path in dotted:
        node = result
        parts = path.split('.')
        for part in parts[:-1]:
            node = node[part]
        node.pop(parts[-1], None)
    return result


def strip_description(profile):
    result = copy.deepcopy(profile)
    result.pop('description')
    return result


def with_deltas(profile, *, env=None, engine=None, trace_region=None, top=None, drop=()):
    result = copy.deepcopy(profile)
    result.pop('description')
    result['env'].update(env or {})
    result['engine'].update(engine or {})
    if trace_region is not None:
        result['engine']['additional-config']['tt']['trace_region_size'] = trace_region
    result.update(top or {})
    for key in drop:
        result.pop(key)
    return result


class EightSeatProfileTests(unittest.TestCase):
    def test_each_profile_is_its_base_plus_exactly_the_named_deltas(self):
        data = profiles()
        traffic_deltas = dict(env=TRAFFIC_ENV, engine={'max-num-seqs': SEATS, 'num-gpu-blocks-override': BLOCKS},
                              trace_region=TRACE_REGION)
        expected_traffic = with_deltas(data[BASE], **traffic_deltas)
        self.assertEqual(strip_description(data[TRAFFIC]), expected_traffic)

        gate_limits = dict(top={'min_answer_tokens': 256, 'gate_only': True}, drop=('max_prompt_tokens',))
        expected_gate = with_deltas(data[TRAFFIC], env=dict(AUDITS_ON, **GATE_MARKER), **gate_limits)
        self.assertEqual(strip_description(data[GATE]), expected_gate)

        expected_time = with_deltas(data[TRAFFIC], env=GATE_MARKER, **gate_limits)
        self.assertEqual(strip_description(data[TIME_GATE]), expected_time)

        expected_diag = with_deltas(data[TIME_GATE], env=DIAG_ENV)
        self.assertEqual(strip_description(data[DIAG]), expected_diag)

        self.assertEqual(strip_description(data[NOWARM]), with_deltas(data[DIAG], env={'QWEN_FAST_M3_REQUEST_WARM': '0'}))
        self.assertEqual(strip_description(data[RSHARD]), with_deltas(data[DIAG], env=RSHARD_ENV))

    def test_the_deltas_from_production_are_literally_these_keys(self):
        data = profiles()
        production = data[BASE]
        traffic = data[TRAFFIC]
        self.assertEqual({key for key in traffic['env'] if traffic['env'][key] != production['env'].get(key)},
                         set(TRAFFIC_ENV))
        changed_engine = {key for key in traffic['engine'] if traffic['engine'][key] != production['engine'].get(key)}
        self.assertEqual(changed_engine, {'max-num-seqs', 'num-gpu-blocks-override', 'additional-config'})
        self.assertEqual(traffic['engine']['max-num-seqs'], SEATS)
        self.assertEqual(traffic['engine']['num-gpu-blocks-override'], BLOCKS)
        self.assertEqual(traffic['engine']['additional-config']['tt']['trace_region_size'], TRACE_REGION)
        for key in ('max_prompt_tokens', 'min_answer_tokens', 'default_max_tokens', 'parser_rechunk'):
            self.assertEqual(traffic[key], production[key], key)
        self.assertNotIn('gate_only', traffic)
        self.assertEqual(data[GATE]['env'].keys() - traffic['env'].keys(), {'QWEN_C2_GATE_PROFILE'})
        self.assertEqual(data[TIME_GATE]['env'].keys() - traffic['env'].keys(), {'QWEN_C2_GATE_PROFILE'})
        self.assertEqual(data[DIAG]['env'].keys() - data[TIME_GATE]['env'].keys(), set(DIAG_ENV))

    def test_the_kv_pool_reserves_every_seat_at_the_full_window(self):
        per_seat = -(-WINDOW // PAGE)
        self.assertEqual(per_seat, 2052)
        for name in (TRAFFIC, GATE, TIME_GATE, DIAG):
            engine = profiles()[name]['engine']
            with self.subTest(profile=name):
                self.assertEqual(engine['max-num-seqs'], SEATS)
                self.assertGreaterEqual(engine['num-gpu-blocks-override'], SEATS * -(-WINDOW // PAGE))
                self.assertEqual(engine['num-gpu-blocks-override'], SEATS * 2052)
                self.assertEqual(engine['max-model-len'], WINDOW)
                self.assertEqual(engine['max-num-batched-tokens'], WINDOW)

    def test_the_longest_admitted_request_fits_the_pool_at_once(self):
        """The fast path cannot preempt, so seats x the longest request must fit the blocks."""
        for name in (TRAFFIC, GATE, TIME_GATE, DIAG):
            profile = contract.load_profile(PROFILES, name)
            capacity = int(profile['env']['QWEN_DSPARK_REQUEST_CONTEXT']) + 256
            budget = int(profile['env']['QWEN_FAST_OUTPUT_BUDGET'])
            room = contract.prompt_room(capacity, budget, profile.get('max_prompt_tokens'),
                                        profile.get('min_answer_tokens'))
            longest = min(room + budget, capacity)
            with self.subTest(profile=name):
                self.assertEqual(profile['engine']['max-model-len'], capacity)
                self.assertGreaterEqual(profile['engine']['num-gpu-blocks-override'],
                                        profile['engine']['max-num-seqs'] * -(-longest // 64))

    def test_the_trace_region_is_half_a_gibibyte_and_nothing_else_in_the_tt_config_moves(self):
        data = profiles()
        for name in (TRAFFIC, GATE, TIME_GATE, DIAG):
            tt = data[name]['engine']['additional-config']['tt']
            reference = dict(data[BASE]['engine']['additional-config']['tt'], trace_region_size=TRACE_REGION)
            with self.subTest(profile=name):
                self.assertEqual(tt, reference)
                self.assertEqual(tt['trace_region_size'], 512 * 1024 * 1024)

    def test_production_is_untouched(self):
        production = profiles()[BASE]
        self.assertEqual(production['engine']['max-num-seqs'], 4)
        self.assertEqual(production['engine']['num-gpu-blocks-override'], 8208)
        self.assertEqual(production['engine']['additional-config']['tt']['trace_region_size'], 268435456)
        self.assertNotIn('QWEN_FAST_M3_BLOCKS', production['env'])
        self.assertNotIn('gate_only', production)
        self.assertNotIn('QWEN_C2_GATE_PROFILE', production['env'])
        for key, value in PRODUCTION_ENV_KEYS.items():
            self.assertEqual(production['env'][key], value, key)
        self.assertEqual(json.loads(PROFILES.read_text(encoding='utf-8'))['default'], BASE)

    def test_the_production_profile_is_byte_identical_to_the_commit_before_the_seat_profiles(self):
        digest = hashlib.sha256(json.dumps(profiles()[BASE], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(digest, PRODUCTION_DIGEST)

    def test_only_the_gate_only_profiles_carry_the_waiver(self):
        data = profiles()
        self.assertNotIn('QWEN_C2_GATE_PROFILE', data[TRAFFIC]['env'])
        self.assertNotIn('gate_only', data[TRAFFIC])
        for name in (GATE, TIME_GATE, DIAG):
            with self.subTest(profile=name):
                self.assertEqual(data[name]['env']['QWEN_C2_GATE_PROFILE'], '1')
                self.assertIs(data[name]['gate_only'], True)
                self.assertEqual(data[name]['min_answer_tokens'], 256)
                self.assertNotIn('max_prompt_tokens', data[name])

    def test_audits_only_on_the_gate_profile_and_the_sampler_stays_in_the_trace(self):
        data = profiles()
        for name in (TRAFFIC, TIME_GATE, DIAG):
            env = data[name]['env']
            with self.subTest(profile=name):
                self.assertEqual(env['QWEN_FAST_VERIFY_T1_AUDIT'], '0')
                self.assertEqual(env['QWEN_FAST_VERIFY_T2_AUDIT'], '0')
                self.assertEqual(env['QWEN_FAST_PACKED_SAMPLER_IN_TRACE'], '1')
        env = data[GATE]['env']
        self.assertEqual((env['QWEN_FAST_VERIFY_T1_AUDIT'], env['QWEN_FAST_VERIFY_T2_AUDIT']), ('1', '1'))

    def test_the_diag_twin_carries_the_hang_instruments_with_their_real_names(self):
        env = profiles()[DIAG]['env']
        for key, value in DIAG_ENV.items():
            self.assertEqual(env[key], value, key)
        for flag in DIAG_ENV:
            self.assertTrue(any(flag in path.read_text(encoding='utf-8', errors='replace')
                                for path in (HERE / 'stall_watch.py', HERE / 'trace_census.py')), flag)
        for flag in ('QWEN_FAST_PACKED_SAMPLER_IN_TRACE', 'QWEN_FAST_VERIFY_T1_AUDIT', 'QWEN_FAST_VERIFY_T2_AUDIT'):
            self.assertIn(flag, env)
        # No capture plug, no request-shard arm (the -rshard twin is a discriminator, not a fallback).
        self.assertFalse([key for key in env if key.startswith('QWEN_FAST_CAPTURE_PLUG')])
        self.assertNotIn('QWEN_FAST_REQUEST_SHARD_ARGMAX', env)

    def test_the_request_warm_flag_and_the_discriminator_arms(self):
        data = profiles()
        for name in (TRAFFIC, GATE, TIME_GATE, DIAG, RSHARD):
            with self.subTest(profile=name):
                self.assertEqual(data[name]['env']['QWEN_FAST_M3_REQUEST_WARM'], '1')
        self.assertEqual(data[NOWARM]['env']['QWEN_FAST_M3_REQUEST_WARM'], '0')
        self.assertEqual(data[DIAG]['env']['QWEN_FAST_STALL_BUILD_S'], '150')
        for name in (NOWARM, RSHARD):
            with self.subTest(profile=name):
                self.assertIs(data[name]['gate_only'], True)
                self.assertEqual(data[name]['env']['QWEN_C2_GATE_PROFILE'], '1')
                self.assertEqual(data[name]['env']['QWEN_FAST_M3_BLOCKS'], '2')
                self.assertEqual(data[name]['env']['QWEN_FAST_VERIFY_T1_AUDIT'], '0')
                self.assertIn('UNVERIFIED', data[name]['description'])
        self.assertNotIn('QWEN_FAST_REQUEST_SHARD_ARGMAX', data[NOWARM]['env'])
        for flag in RSHARD_ENV:
            self.assertIn(flag, (HERE / 'verifier_engine_tp.py').read_text(encoding='utf-8'))
        self.assertIn('QWEN_FAST_M3_REQUEST_WARM', (HERE / 'serving_runtime.py').read_text(encoding='utf-8'))
        self.assertNotIn('QWEN_FAST_M3_REQUEST_WARM', data[BASE]['env'])

    def test_the_names_are_new_and_the_four_seat_twins_keep_theirs(self):
        data = profiles()
        for name in (TRAFFIC, GATE, TIME_GATE, DIAG):
            self.assertIn(name, data)
        self.assertEqual(data['c2-packed-tp4-diag-strace']['engine']['max-num-seqs'], 4)
        self.assertEqual(data['c2-packed-tp4-time-gate']['engine']['max-num-seqs'], 4)

    def test_the_flag_is_in_no_four_seat_profile(self):
        for name, profile in profiles().items():
            if name in (TRAFFIC, GATE, TIME_GATE, DIAG, NOWARM, RSHARD):
                continue
            with self.subTest(profile=name):
                self.assertNotIn('QWEN_FAST_M3_BLOCKS', profile.get('env', {}))
                self.assertNotIn('QWEN_FAST_M3_REQUEST_WARM', profile.get('env', {}))

    def test_the_mesh_contract_and_the_gate_rules_accept_each_profile(self):
        for name in (TRAFFIC, GATE, TIME_GATE, DIAG, NOWARM, RSHARD):
            profile = contract.load_profile(PROFILES, name)
            with self.subTest(profile=name):
                self.assertEqual(contract.mesh_problems(profile), [])
                self.assertEqual(profile['mesh_device'], 'P150x4')
                self.assertEqual(profile['env']['QWEN_FAST_TP'], '4')
                self.assertEqual(contract.request_limits(profile)['budget'], 16384)

    def test_the_admission_environment_accepts_each_profile_over_the_image_environment(self):
        image = image_env()
        for name in (TRAFFIC, GATE, TIME_GATE, DIAG):
            environ = dict(image, **profiles()[name]['env'])
            with self.subTest(profile=name):
                self.assertEqual(admission.width(environ), 4)
                self.assertEqual(environ['QWEN_FAST_M3_BLOCKS'], '2')
                self.assertEqual(admission.check_environment(environ, M3), [])

    def test_every_profile_has_a_description_that_names_the_delta_and_no_host_or_card(self):
        banned = ('10.', '192.168', 'sha256:', 'ghcr', 'zot', '/home/', 'C:\\', 'D:\\', 'serial', 'laya')
        for name in (TRAFFIC, GATE, TIME_GATE, DIAG):
            description = profiles()[name]['description']
            with self.subTest(profile=name):
                self.assertIn('QWEN_FAST_M3_BLOCKS=2', description)
                self.assertIn('UNVERIFIED', description)
                for word in banned:
                    self.assertNotIn(word, description)


PRODUCTION_DIGEST = 'aeaa12f7e4b33ec55c1b7405825d3d1472ee3ea78b863bba8a05bd0809252b82'

if __name__ == '__main__':
    unittest.main()
