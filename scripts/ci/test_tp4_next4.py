"""tp4/next-4: the profiles that stack the levers on the traffic recipe, and the four-seat SHIP window's templates.

c2-packed-tp4-best-ship is production's recipe plus exactly the nine round-time levers c2-packed-tp4-best-strace proved on hardware, at production
limits and without the gate marker; c2-packed-tp4-best-ship-warm4 is that with the in-trace sampler off and the request-width warm on (only for after
the warm4 window proves out); c2-packed-tp4-8-best is the eight-seat timed twin plus the levers (gate only). c2-packed-tp4-best, the older gate-only
combined-best control of the tp4/next windows, is unchanged. The templates are public, so they name no rig, card, address, registry or digest."""

import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c2_serving_job as job  # noqa: E402
import serving_runtime  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FOLDER = os.path.join(HERE, 'references', 'tp4-next4-ship-jobs')
PROFILES_PATH = os.path.join(HERE, 'qwen_c2_profiles.json')
BANNED = re.compile(r'blackhole-[A-Za-z0-9]{8,}|thatch\.local|\d{1,3}(\.\d{1,3}){3}|sha256:[0-9a-f]{16}|[0-9a-f]{40,}|'
                    r'/dev/tenstorrent|home/|zot\.')
PLACEHOLDERS = {'@THIN_LAYER_IMAGE@': 'thin-layer-image-ref'}
IMAGE = 'tp4-serve-9'
PRODUCTION, BEST, BEST_STRACE = 'c2-packed-tp4', 'c2-packed-tp4-best', 'c2-packed-tp4-best-strace'
SHIP, SHIP_WARM4 = 'c2-packed-tp4-best-ship', 'c2-packed-tp4-best-ship-warm4'
EIGHT_TIME, EIGHT_BEST = 'c2-packed-tp4-8-time-gate', 'c2-packed-tp4-8-best'
LEVERS = ('QWEN_FAST_QUAD_DRAFT', 'QWEN_FAST_FUSED_COMMIT', 'QWEN_FAST_FUSED_COMMIT_INPLACE', 'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS',
          'QWEN_FAST_TP4_COMMIT_LANES', 'QWEN_FAST_TP4_SHARD_VALUES', 'QWEN_FAST_TP4_GDN_GLUE', 'QWEN_FAST_TP4_GDN_BLOCK_CONV',
          'QWEN_FAST_TP4_ATTN_FOLD')
IN_TRACE, WARM, MARKER = 'QWEN_FAST_PACKED_SAMPLER_IN_TRACE', 'QWEN_FAST_M3_REQUEST_WARM', 'QWEN_C2_GATE_PROFILE'
HANG = tuple('BH%d-hang-shapes-audits-off' % number for number in range(1, 6))
ORDERED = ('B0-build', 'A0-agentstop') + HANG + ('M-matrix-traffic', 'SR-platform-replay', 'H-handback-reset-fabric-agentstart')
ACTIONS = {'B0-build': 'build', 'A0-agentstop': 'agentstop unserve', 'M-matrix-traffic': 'reset gate', 'SR-platform-replay': 'reset replay',
           'H-handback-reset-fabric-agentstart': 'status reset fabric agentstart', **{name: 'reset smoke' for name in HANG}}


def profiles():
    with open(PROFILES_PATH, encoding='utf-8') as handle:
        return json.load(handle)['profiles']


def differences(mine, base, keys=('env',)):
    """(env added or changed, env removed) of mine against base, and the other keys that differ (description aside)."""
    env, other = mine['env'], base['env']
    changed = {key: value for key, value in env.items() if other.get(key) != value}
    removed = sorted(set(other) - set(env))
    rest = sorted(key for key in set(mine) | set(base) if key not in ('env', 'description') and mine.get(key) != base.get(key))
    return changed, removed, rest


class ProfileTests(unittest.TestCase):
    def test_the_ship_candidate_is_production_plus_exactly_the_nine_levers(self):
        found = profiles()
        changed, removed, rest = differences(found[SHIP], found[PRODUCTION])
        self.assertEqual(changed, {key: '1' for key in LEVERS})
        self.assertEqual((removed, rest), ([], []))
        self.assertEqual(found[SHIP]['env'][IN_TRACE], '1')
        self.assertNotIn(MARKER, found[SHIP]['env'])
        self.assertNotIn('gate_only', found[SHIP])
        self.assertEqual((found[SHIP]['max_prompt_tokens'], found[SHIP]['min_answer_tokens']), (123136, 8192))

    def test_the_ship_candidate_is_the_proven_strace_arm_less_the_marker_and_gate_limits(self):
        found = profiles()
        strace, ship = found[BEST_STRACE], found[SHIP]
        self.assertEqual(strace['env'], dict(ship['env'], **{MARKER: '1'}))
        for key in set(strace) | set(ship):
            if key not in ('description', 'env', 'gate_only', 'min_answer_tokens', 'max_prompt_tokens'):
                self.assertEqual(ship.get(key), strace.get(key), key)

    def test_the_warm4_twin_is_the_ship_candidate_with_two_deltas(self):
        found = profiles()
        changed, removed, rest = differences(found[SHIP_WARM4], found[SHIP])
        self.assertEqual((changed, removed, rest), ({WARM: '1'}, [IN_TRACE], []))
        self.assertNotIn('gate_only', found[SHIP_WARM4])
        self.assertTrue(found[SHIP_WARM4]['description'].startswith('C2-packed-any'))

    def test_the_eight_seat_levers_arm_is_gate_only_and_the_time_twin_plus_the_levers(self):
        found = profiles()
        changed, removed, rest = differences(found[EIGHT_BEST], found[EIGHT_TIME])
        self.assertEqual(changed, {key: '1' for key in LEVERS})
        self.assertEqual((removed, rest), ([], []))
        self.assertIs(found[EIGHT_BEST]['gate_only'], True)
        self.assertTrue(found[EIGHT_BEST]['description'].startswith('GATE ONLY'))
        env = found[EIGHT_BEST]['env']
        self.assertEqual((env['QWEN_FAST_M3_BLOCKS'], env[WARM], env[IN_TRACE], env[MARKER]), ('2', '1', '1', '1'))

    def test_production_the_default_and_the_older_best_are_untouched(self):
        with open(PROFILES_PATH, encoding='utf-8') as handle:
            found = json.load(handle)
        self.assertEqual(found['default'], PRODUCTION)
        env = found['profiles'][PRODUCTION]['env']
        for key in LEVERS + (WARM, MARKER, 'QWEN_FAST_M3_BLOCKS'):
            self.assertNotEqual(env.get(key), '1', key)
        self.assertIs(found['profiles'][BEST]['gate_only'], True)
        self.assertNotIn(IN_TRACE, found['profiles'][BEST]['env'])
        self.assertEqual(found['profiles'][BEST]['env']['QWEN_FAST_QUAD_DRAFT'], '1')

    def test_the_warm_flag_is_admitted_for_the_warm4_twin_at_four_seats_and_at_two_blocks_for_the_others(self):
        found = profiles()
        four = {'scheduler_requests': 4}
        image = {'QWEN_FAST_FOUR_AS_TWO': '0', 'QWEN_FAST_PACKED_STEP': '1'}
        env = dict(image, **found[SHIP_WARM4]['env'])
        self.assertEqual(serving_runtime.m3_request_warm(1, four, env), (1, 2, 4))
        self.assertIsNone(serving_runtime.m3_request_warm(1, four, found[SHIP]['env']))
        eight = {'scheduler_requests': 8}
        self.assertEqual(serving_runtime.m3_request_warm(2, eight, found[EIGHT_BEST]['env']), (1, 2, 4))

    def test_every_new_profile_carries_the_serving_shape_of_its_base(self):
        found = profiles()
        for name, base in ((SHIP, PRODUCTION), (SHIP_WARM4, PRODUCTION), (EIGHT_BEST, EIGHT_TIME)):
            with self.subTest(profile=name):
                self.assertEqual(found[name]['engine'], found[base]['engine'])
                self.assertEqual(found[name]['mesh_device'], 'P150x4')
                self.assertIs(found[name]['parser_rechunk'], True)


def read_order():
    with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
        return [line.split() for line in handle.read().splitlines() if line.strip() and not line.startswith('#')]


def text_of(name):
    with open(os.path.join(FOLDER, name + '.env'), encoding='utf-8') as handle:
        return handle.read()


def parsed(name):
    text = text_of(name)
    for placeholder, value in PLACEHOLDERS.items():
        text = text.replace(placeholder, value)
    return job.read_job(job.parse_env(text), sorted(profiles()), root=ROOT)


class ShipPackTests(unittest.TestCase):
    def test_every_template_is_in_the_order_once_in_the_ship_order(self):
        rows = read_order()
        self.assertTrue(all(len(row) == 4 for row in rows))
        on_disk = sorted(name[:-4] for name in os.listdir(FOLDER) if name.endswith('.env'))
        self.assertEqual(sorted(row[0] for row in rows), on_disk)
        self.assertEqual([row[0] for row in rows], list(ORDERED))

    def test_the_modes_the_one_image_and_the_actions(self):
        for name, mode, image, minutes in read_order():
            with self.subTest(job=name):
                self.assertEqual(image, IMAGE)
                outputs = parsed(name)
                self.assertEqual((outputs['tag'], outputs['cards']), (IMAGE, 'quad'))
                self.assertEqual(outputs['actions'], ACTIONS[name])
                self.assertEqual(mode, 'pre' if name == 'B0-build' else 'stop')
                self.assertTrue(minutes.isdigit() and 5 <= int(minutes) <= 180, minutes)

    def test_only_the_build_bakes_and_it_bakes_the_traffic_candidate(self):
        for name in ORDERED:
            outputs = parsed(name)
            self.assertEqual(outputs['bake_default_profile'], SHIP if name == 'B0-build' else '', name)
        self.assertEqual(parsed('B0-build')['profile'], SHIP)
        with open(PROFILES_PATH, encoding='utf-8') as handle:
            self.assertEqual(json.load(handle)['default'], PRODUCTION)

    def test_a_gate_arm_can_never_be_baked_and_the_warm4_twin_is_held_back_only_by_the_order_file(self):
        for name in (EIGHT_BEST, BEST_STRACE, BEST):
            text = text_of('B0-build').replace('=' + SHIP, '=' + name)
            with self.subTest(profile=name), self.assertRaises(job.JobError):
                job.read_job(job.parse_env(text), sorted(profiles()), root=ROOT)
        # the warm4 twin CAN be baked by read_job: only ORDER.txt and the profile description hold it back until warm4 proves out
        twin = text_of('B0-build').replace('=' + SHIP, '=' + SHIP_WARM4)
        self.assertEqual(job.read_job(job.parse_env(twin), sorted(profiles()), root=ROOT)['bake_default_profile'], SHIP_WARM4)

    def test_the_five_hang_runs_name_the_traffic_candidate_and_the_four_user_shapes(self):
        for name in HANG:
            outputs = parsed(name)
            with self.subTest(job=name):
                self.assertEqual(outputs['profile'], SHIP)
                self.assertEqual(outputs['tests'], 'warmup,coding,concurrent4,concurrent4_steady,steady_resend,replay_concurrent4,'
                                                   'concurrent4_code_equal')
        self.assertEqual(len({text_of(name).split('C2_CARDS=')[1] for name in HANG}), 1)

    def test_the_matrix_and_the_replay_run_the_candidate(self):
        matrix, replay = parsed('M-matrix-traffic'), parsed('SR-platform-replay')
        self.assertEqual((matrix['profile'], matrix['gate_plan']), (SHIP, 'matrix'))
        self.assertIn('C2_GATE_LENGTHS=4096,16384,32768,60000', text_of('M-matrix-traffic').splitlines())
        self.assertEqual(replay['replay_profile'], SHIP)
        self.assertEqual(replay['replay_budget_smoke'], '1')
        self.assertEqual(replay['platform_image'], 'thin-layer-image-ref')
        self.assertIn('@THIN_LAYER_IMAGE@', text_of('SR-platform-replay'))

    def test_the_agent_stops_first_after_the_build_and_starts_last(self):
        names = [row[0] for row in read_order()]
        self.assertEqual(names[:2], ['B0-build', 'A0-agentstop'])
        self.assertEqual(names[-1], 'H-handback-reset-fabric-agentstart')
        for name in ORDERED:
            actions = parsed(name)['actions'].split()
            self.assertEqual('agentstop' in actions, name == 'A0-agentstop', name)
            self.assertEqual('agentstart' in actions, name == 'H-handback-reset-fabric-agentstart', name)
            self.assertEqual('build' in actions, name == 'B0-build', name)

    def test_the_templates_name_no_host_address_registry_or_digest(self):
        for name in ORDERED + ('ORDER',):
            with open(os.path.join(FOLDER, name + ('.txt' if name == 'ORDER' else '.env')), encoding='utf-8') as handle:
                text = handle.read()
            with self.subTest(file=name):
                self.assertIsNone(BANNED.search(text), BANNED.search(text))

    def test_the_order_states_the_go_rule_the_naming_and_the_untouched_default(self):
        with open(os.path.join(FOLDER, 'ORDER.txt'), encoding='utf-8') as handle:
            text = handle.read()
        for word in ('THE GO RULE FOR MOVING :latest', 'NO-GO', 'ROLLBACK', 'c2-packed-tp4-best is the older GATE-ONLY',
                     'qwen_c2_profiles.json\'s default is still c2-packed-tp4', 'five CONSECUTIVE completions',
                     'THATCH_SERVING_SESSION_CAP=4', 'best-ship-warm4', 'c2-packed-tp4-8-best'):
            self.assertIn(word, text)


if __name__ == '__main__':
    unittest.main()
