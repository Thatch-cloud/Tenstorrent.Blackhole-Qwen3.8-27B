"""The octo-T8 and lone-user profile twins (make_octo_profiles): each is its parent plus exactly the documented env and, for the octo ones, the memory plan; gate only; no other
profile names the flags; the memory plan fits; and each twin passes the admission's environment checks over the image's ENV (the device pieces are the only reasons left)."""

import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import make_octo_profiles as twins  # noqa: E402
import make_parked_profiles as parked  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import serving_octo as octo  # noqa: E402
from test_c2_packed_tp4_profiles import image_env, profiles  # noqa: E402
import test_octo as host_tests  # noqa: E402

P = twins.PARENT
PROFILES_TEXT = (HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8')
ALL_PIECES = frozenset(key for key, text in octo.DEVICE_PIECES)
OCTO = twins.octo_twin_names()
SOLO = tuple(name for name in twins.twin_names() if name not in OCTO)


def spec(name):
    return next(item for item in twins.specs() if item[0] == name)


def container_env(name):
    environ = dict(image_env())
    contract.apply_environment(dict(profiles()[name], name=name), environ)
    environ['QWEN_C2_GATE'] = '1'                                    # what the gate workflow adds to a gate-only profile's container
    return environ


class GeneratorTests(unittest.TestCase):
    def test_the_checked_in_twins_are_what_the_generator_makes(self):
        self.assertEqual(twins.render(twins.generate(json.loads(PROFILES_TEXT))), PROFILES_TEXT)

    def test_the_engine_reuse_generator_leaves_the_octo_twins_alone_and_still_agrees(self):
        data = json.loads(PROFILES_TEXT)
        self.assertEqual(parked.render(parked.generate(data)), PROFILES_TEXT)
        for name in twins.twin_names():
            self.assertFalse(name.startswith(parked.CANDIDATE_PARENT + '-parked'), 'a name the engine-reuse generator would drop and regenerate: ' + name)

    def test_the_twins_follow_the_engine_reuse_family_and_nothing_else_moved(self):
        names = list(json.loads(PROFILES_TEXT)['profiles'])
        last_parked = max(index for index, name in enumerate(names) if name.startswith(twins.PARKED_PARENT))
        self.assertEqual(names[last_parked + 1:last_parked + 1 + len(twins.twin_names())], list(twins.twin_names()))

    def test_the_generator_is_idempotent_and_refuses_a_moved_parent(self):
        data = json.loads(PROFILES_TEXT)
        self.assertEqual(twins.generate(twins.generate(data)), twins.generate(data))
        for edit, text in (({'num-gpu-blocks-override': 19200}, 'num-gpu-blocks-override 19200, not 19968'),
                           ({'max-num-seqs': 4}, 'max-num-seqs 4, not 8 seats')):
            broken = copy.deepcopy(data)
            broken['profiles'][twins.PARENT]['engine'].update(edit)
            with self.assertRaisesRegex(ValueError, text):
                twins.generate(broken)
        broken = copy.deepcopy(data)
        broken['profiles'][twins.PARENT]['engine']['additional-config']['tt']['trace_region_size'] = 1 << 30
        with self.assertRaisesRegex(ValueError, 'trace_region_size'):
            twins.generate(broken)
        broken = copy.deepcopy(data)
        broken['profiles'][twins.PARENT]['env']['QWEN_FAST_OCTO'] = 'live'
        with self.assertRaisesRegex(ValueError, 'already names QWEN_FAST_OCTO'):
            twins.generate(broken)
        broken = copy.deepcopy(data)
        del broken['profiles'][twins.PARKED_PARENT]
        with self.assertRaisesRegex(ValueError, 'make_parked_profiles.py --write first'):
            twins.generate(broken)


class TwinTests(unittest.TestCase):
    def test_the_twins_are_the_documented_seven_and_all_exist(self):
        self.assertEqual(twins.twin_names(), (P + '-octo', P + '-octo-live', P + '-octo-audit', P + '-octo-parked', P + '-octo-parked-live',
                                              P + '-solopacked', P + '-solopacked-parked'))
        self.assertEqual(OCTO, twins.twin_names()[:5])
        self.assertEqual(SOLO, twins.twin_names()[5:])
        for name in twins.twin_names():
            self.assertIn(name, profiles())

    def test_each_is_its_parent_plus_exactly_the_documented_env(self):
        for name, parent_name, env, third, why in twins.specs():
            with self.subTest(profile=name):
                profile, parent = profiles()[name], profiles()[parent_name]
                differing = {key for key in set(profile['env']) | set(parent['env']) if profile['env'].get(key) != parent['env'].get(key)}
                expected = set(env) | ({'QWEN36_MAX_TOKENS_ALL_USERS'} if third else set())
                self.assertEqual(differing, expected)
                for key, value in env.items():
                    self.assertEqual(profile['env'][key], value)
                self.assertTrue(profile['gate_only'])
                self.assertEqual({key for key in set(profile) | set(parent) if profile.get(key) != parent.get(key)},
                                 {'env', 'description'} | ({'engine'} if third else set()))
                self.assertIn(parent_name, profile['description'])
                self.assertIn('make_octo_profiles.py', profile['description'])
                self.assertIn('GATE ONLY', profile['description'])

    def test_the_octo_twins_edit_the_engine_in_exactly_the_pool_and_the_trace_region(self):
        for name in OCTO:
            with self.subTest(profile=name):
                parent = profiles()[spec(name)[1]]['engine']
                engine = copy.deepcopy(profiles()[name]['engine'])
                self.assertEqual(engine['num-gpu-blocks-override'], 16500)
                self.assertEqual(engine['additional-config']['tt']['trace_region_size'], 640 << 20)
                engine['num-gpu-blocks-override'] = parent['num-gpu-blocks-override']
                engine['additional-config']['tt']['trace_region_size'] = parent['additional-config']['tt']['trace_region_size']
                self.assertEqual(engine, parent)

    def test_the_lone_user_twins_leave_the_engine_and_the_pool_alone(self):
        for name in SOLO:
            with self.subTest(profile=name):
                parent = profiles()[spec(name)[1]]
                self.assertEqual(profiles()[name]['engine'], parent['engine'])
                self.assertEqual(profiles()[name]['env']['QWEN36_MAX_TOKENS_ALL_USERS'], parent['env']['QWEN36_MAX_TOKENS_ALL_USERS'])

    def test_the_modes_are_the_documented_ones(self):
        modes = {name: profiles()[name]['env'].get('QWEN_FAST_OCTO') for name in OCTO}
        self.assertEqual(modes, {P + '-octo': 'alternate', P + '-octo-live': 'live', P + '-octo-audit': 'alternate',
                                 P + '-octo-parked': 'alternate', P + '-octo-parked-live': 'live'})
        self.assertEqual({profiles()[name]['env']['QWEN_FAST_OCTO_MIN_LIVE'] for name in OCTO}, {'6'})
        self.assertEqual({profiles()[name]['env']['QWEN_FAST_SOLO_PACKED'] for name in SOLO}, {'1'})
        for name in SOLO:
            self.assertNotIn('QWEN_FAST_OCTO', profiles()[name]['env'])
        for name in OCTO:
            self.assertNotIn('QWEN_FAST_SOLO_PACKED', profiles()[name]['env'])

    def test_the_audited_twin_carries_the_audits_and_the_timed_ones_do_not(self):
        audited = profiles()[P + '-octo-audit']['env']
        timed = profiles()[P + '-octo']['env']
        self.assertEqual((audited['QWEN_FAST_VERIFY_T1_AUDIT'], audited['QWEN_FAST_VERIFY_T2_AUDIT']), ('1', '1'))
        self.assertEqual((timed['QWEN_FAST_VERIFY_T1_AUDIT'], timed['QWEN_FAST_VERIFY_T2_AUDIT']), ('0', '0'))
        for name in OCTO:
            if name != P + '-octo-audit':
                self.assertEqual(profiles()[name]['env']['QWEN_FAST_VERIFY_T1_AUDIT'], '0', name)

    def test_the_parked_twins_keep_the_engine_reuse_flags_and_the_others_carry_none(self):
        for name in (P + '-octo-parked', P + '-octo-parked-live', P + '-solopacked-parked'):
            env = profiles()[name]['env']
            self.assertEqual((env['QWEN_FAST_PARKED_ENGINES'], env['QWEN_FAST_PARKED_DRAFTS'], env['QWEN_FAST_LEVERN_BUILD_MS']), ('1', '1', 'learned'), name)
        for name in (P + '-octo', P + '-octo-live', P + '-octo-audit', P + '-solopacked'):
            self.assertNotIn('QWEN_FAST_PARKED_ENGINES', profiles()[name]['env'], name)


class NoOtherProfileTests(unittest.TestCase):
    def test_no_other_profile_and_not_the_image_names_a_flag(self):
        for name, profile in profiles().items():
            if name in twins.twin_names():
                continue
            with self.subTest(profile=name):
                for flag in twins.LEVERS:
                    self.assertNotIn(flag, profile.get('env', {}))
        for flag in twins.LEVERS:
            self.assertNotIn(flag, image_env())

    def test_the_production_and_traffic_profiles_are_untouched(self):
        for name in ('c2-packed-tp4', 'c2-packed-tp4-8x262k-ship-prefix', P + '-traffic', P, P + '-audit', P + '-parked'):
            profile = profiles()[name]
            with self.subTest(profile=name):
                self.assertEqual(profile['engine']['num-gpu-blocks-override'] if name != 'c2-packed-tp4' else 19968, 19968 if name != 'c2-packed-tp4' else 19968)
                if name != 'c2-packed-tp4':
                    self.assertEqual(profile['engine']['additional-config']['tt']['trace_region_size'], 512 << 20)
                    self.assertEqual(profile['env']['QWEN36_MAX_TOKENS_ALL_USERS'], '1277440')
                self.assertNotIn('QWEN_C2_GATE_PROFILE', profile['env'])
        for name in ('c2-packed-tp4', 'c2-packed-tp4-8x262k-ship-prefix', P + '-traffic'):
            self.assertIsNot(profiles()[name].get('gate_only'), True)

    def test_the_gate_marker_is_carried_by_gate_only_profiles(self):
        for name in twins.twin_names():
            self.assertEqual(profiles()[name]['env']['QWEN_C2_GATE_PROFILE'], '1')
            self.assertIs(profiles()[name]['gate_only'], True)


class MemoryPlanTests(unittest.TestCase):
    def test_the_pool_gives_back_what_the_third_block_and_the_larger_trace_region_take(self):
        plan = twins.memory_plan()
        self.assertEqual(plan['freed_blocks'], 3468)
        self.assertEqual(plan['freed_bytes'], 3468 * 557056)
        self.assertEqual(plan['third_block_bytes'], 224 * 10 ** 6 * 8)
        self.assertEqual(plan['trace_region_bytes'], (640 - 512) << 20)
        self.assertGreaterEqual(plan['margin_bytes'], 0)
        self.assertLess(plan['margin_bytes'], 16 * 10 ** 6, 'a tight plan: nothing is given back that the block does not use')
        self.assertGreater(plan['freed_bytes'], plan['third_block_bytes'] + plan['trace_region_bytes'])

    def test_one_block_less_would_not_fit(self):
        plan = twins.memory_plan()
        self.assertLess(plan['freed_bytes'] - 557056 * 11, plan['third_block_bytes'] + plan['trace_region_bytes'], 'the margin is under 11 blocks')

    def test_the_numbers_the_plan_was_derived_from_are_the_repos_own(self):
        import test_c2_packed_tp4_profiles as source

        with open(source.__file__, encoding='utf-8') as handle:
            text = handle.read()
        self.assertIn('packed_block=224 * mb', text, 'the measured need of a packed block, per bank')
        with open(str(HERE.parent.parent / 'docs' / 'tp4-drafter-bf16.md'), encoding='utf-8') as handle:
            self.assertIn('557,056 B', handle.read())
        self.assertEqual(twins.MEMORY['parent_blocks'], parked.KV_BLOCKS)
        self.assertEqual(twins.MEMORY['parent_trace_region'], parked.TRACE_REGION)

    def test_the_pool_still_holds_four_full_windows_and_the_contract_agrees_with_it(self):
        plan = twins.memory_plan()
        self.assertEqual(plan['pool_tokens'], (16500 - 8) * 64)
        self.assertGreaterEqual(plan['pool_tokens'], 4 * 262144)
        for name in OCTO:
            with self.subTest(profile=name):
                profile = dict(profiles()[name], name=name)
                self.assertIsNone(contract.kv_pool_problem(profile))
                self.assertIsNone(contract.kv_reservation_problem(profile))
                self.assertEqual(contract.real_pool_blocks(profile), 16500)
                limits = contract.request_limits(profile)
                self.assertEqual(limits['kv_pool_blocks'], 16499, 'the reservation hands out the pool less the null block')
                self.assertEqual(profile['env']['QWEN36_MAX_TOKENS_ALL_USERS'], str(plan['pool_tokens']))
        for name in SOLO:
            self.assertEqual(contract.real_pool_blocks(dict(profiles()[name], name=name)), 19968)

    def test_four_of_the_largest_requests_the_contract_admits_are_resident_at_once(self):
        from serving_kv_reservation import request_blocks

        for name in OCTO:
            profile = profiles()[name]
            window = profile['engine']['max-model-len'] - profile['drafter_headroom_tokens']
            prompt = profile['max_prompt_tokens']
            self.assertEqual(prompt, 253920)
            needed = request_blocks(prompt, window - prompt)          # the answer room the window leaves that prompt (min_answer_tokens is 8,192)
            self.assertGreaterEqual(window - prompt, profile['min_answer_tokens'])
            self.assertEqual(needed, 4097)
            self.assertLessEqual(4 * needed, 16499, 'four windows reserved at once: %s' % name)
            self.assertGreater(4 * request_blocks(prompt, window - prompt), 16499 - 4 * 32, 'and the pool is tight: little more than four')

    def test_the_trace_region_is_a_whole_number_of_mebibytes_the_engine_takes(self):
        for name in OCTO:
            region = profiles()[name]['engine']['additional-config']['tt']['trace_region_size']
            self.assertEqual(region, 671088640)
            self.assertEqual(region % (1 << 20), 0)


class AdmissionOverTheImageTests(unittest.TestCase):
    """Each twin over the C2 image's own ENV: the environment is admitted; today only the device pieces keep the flag from engaging."""

    M3 = (True, 'users=8 FOUR_AS_TWO=0 PACKED_STEP=1 M3_BLOCKS=2')

    def reasons(self, name):
        log = host_tests.Lines()
        with self.assertRaises(ValueError):
            octo.octo_admission(self.M3, container_env(name), log=log)
        return [line[len(octo.REFUSED_MARKER) + 2:] for line in log.lines]

    def test_the_octo_twins_are_admitted_today_over_the_images_env_with_the_device_pieces_built(self):
        # nothing patched: serving_octo.BUILT as the tree has it, the TP4 GDN sibling's own limit
        self.assertEqual(octo.device_gaps(), [])
        for name in OCTO:
            with self.subTest(profile=name):
                log = host_tests.Lines()
                record = octo.octo_admission(self.M3, container_env(name), log=log)
                self.assertEqual((record['mode'], record['min_live'], record['rows'], record['users']), (profiles()[name]['env']['QWEN_FAST_OCTO'], 6, 8, 8))
                unqualified = [line for line in log.lines if line.startswith(octo.UNQUALIFIED_MARKER)]
                self.assertEqual(len(unqualified), len(octo.UNQUALIFIED_ITEMS), 'every card question is a line, one each')
                self.assertTrue(any(line.startswith(octo.ADMITTED_MARKER) for line in log.lines))
                self.assertFalse(any(line.startswith(octo.REFUSED_MARKER) for line in log.lines))

    def test_a_piece_that_is_not_built_refuses_every_octo_twin_by_name(self):
        for key, text in octo.DEVICE_PIECES:
            for name in OCTO[:2]:
                with self.subTest(profile=name, piece=key), patch.object(octo, 'BUILT', octo.BUILT - {key}):
                    self.assertEqual([reason.split(' is not built')[0] for reason in self.reasons(name)], ['device piece ' + key])

    def test_a_traffic_profile_with_the_flag_is_refused_whatever_else_holds(self):
        environ = container_env(P + '-traffic')
        environ['QWEN_FAST_OCTO'] = 'live'
        with host_tests.built(), self.assertRaisesRegex(ValueError, 'gate run of a gate-only'):
            octo.octo_admission(self.M3, environ, log=host_tests.Lines())

    def test_the_lone_user_twins_are_refused_today_for_the_idle_segments_alone(self):
        for name in SOLO:
            with self.subTest(profile=name):
                with self.assertRaises(ValueError) as caught:
                    octo.solo_packed_admission(self.M3, container_env(name), log=host_tests.Lines())
                reasons = str(caught.exception).split('is refused: ', 1)[1].split('; ')
                self.assertEqual(len(reasons), 1, reasons)
                self.assertIn('a lone user leaves 3 idle segments', reasons[0])
                with patch.object(octo, 'idle_capacity', Mock(return_value=3)):
                    self.assertEqual(octo.solo_packed_admission(self.M3, container_env(name), log=host_tests.Lines()), dict(min_users=1))

    def test_the_smoke_judge_reads_the_modes_these_profiles_carry(self):
        import octo_judge

        for name in OCTO:
            env = profiles()[name]['env']
            self.assertIn(octo_judge.mode(env), ('live', 'alternate'))
            self.assertEqual(octo_judge.min_live(env), 6)
        for name in SOLO:
            self.assertTrue(octo_judge.solo_packed(profiles()[name]['env']))
            self.assertEqual(octo_judge.mode(profiles()[name]['env']), 'off')


if __name__ == '__main__':
    unittest.main()
