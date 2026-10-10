"""Engine reuse, E6': the policy, the contract and the profiles of QWEN_FAST_PARKED_ENGINES at four chips.

The policy matrix (serving_fast_policy.parked_engine_problems and validate_fast_config's use of it), the contract's side
(serving_c2_contract.parked_problems, apply_environment's ownership of the switches, the boot's refusals), the generated twins held to their parents
(make_parked_profiles: regenerate-and-compare, so a parent that moves forces a regeneration), the production profile and every non-gate profile free
of any QWEN_FAST_PARKED_ name, the KV pool and trace region the memory margins were derived under pinned, and flag-off parity: with no
QWEN_FAST_PARKED_ name in the environment, validate_fast_config is what it was.
"""

import copy
import json
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_kvread_profiles as kvread_twins  # noqa: E402
import make_octo_profiles as octo_twins  # noqa: E402
import make_parked_profiles as twins  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import serving_fast_policy as policy  # noqa: E402
import serving_parked_engines as engines  # noqa: E402

PROFILES = str(HERE / 'qwen_c2_profiles.json')
FLAG = 'QWEN_FAST_PARKED_ENGINES'
PRODUCTION = twins.PRODUCTION

# What the eight-seat 262k image gives a process: the S2 shape the parked engines need.
GOOD = {'QWEN_FAST_ANY_REQUEST': '1', 'QWEN_FAST_EXTENT_REPLAY': '1', 'QWEN_FAST_PACKED_STEP': '1', 'QWEN_FAST_SHARED_CCL': '1',
        'QWEN_FAST_VERIFY_T1': '1', 'QWEN_FAST_TP': '4', 'QWEN_FAST_M3_REQUEST_WARM': '1', 'QWEN_FAST_M3_BLOCKS': '2', FLAG: '1'}


def load(name):
    return contract.load_profile(PROFILES, name)


def raw():
    return json.loads(Path(PROFILES).read_text(encoding='utf-8'))


def config(seqs=8):
    return SimpleNamespace(additional_config={'qwen_fast_t16': True},
        scheduler_config=SimpleNamespace(max_num_seqs=seqs, async_scheduling=False, scheduler_cls='x'),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        cache_config=SimpleNamespace(block_size=64, enable_prefix_caching=False),
        lora_config=None, model_config=SimpleNamespace(max_model_len=262144),
        speculative_config=SimpleNamespace(method='dflash', num_speculative_tokens=15,
                                           draft_sample_method='greedy', rejection_sample_method='standard'))


class PolicyMatrixTests(unittest.TestCase):
    def problems(self, users=8, **changes):
        environ = dict(GOOD)
        for name, value in changes.items():
            if value is None:
                environ.pop(name, None)
            else:
                environ[name] = value
        return policy.parked_engine_problems(environ, users)

    def test_the_four_card_s2_shape_passes_at_eight_seats_and_at_four(self):
        self.assertEqual(self.problems(), [])
        self.assertEqual(self.problems(users=4, QWEN_FAST_M3_BLOCKS=None), [])
        self.assertEqual(self.problems(users=4, QWEN_FAST_M3_BLOCKS='1'), [])

    def test_each_requirement_is_refused_by_name(self):
        for name, bad in (('QWEN_FAST_ANY_REQUEST', '0'), ('QWEN_FAST_EXTENT_REPLAY', None), ('QWEN_FAST_PACKED_STEP', '0'),
                          ('QWEN_FAST_SHARED_CCL', '0'), ('QWEN_FAST_VERIFY_T1', None), ('QWEN_FAST_TP', None), ('QWEN_FAST_TP', '2'),
                          ('QWEN_FAST_M3_REQUEST_WARM', None), ('QWEN_FAST_M3_REQUEST_WARM', '0')):
            with self.subTest(name=name, bad=bad):
                found = self.problems(**{name: bad})
                self.assertEqual(len(found), 1, found)
                self.assertIn(name, found[0])

    def test_the_seat_count_and_the_block_count_agree(self):
        self.assertIn('M3_BLOCKS=2', self.problems(QWEN_FAST_M3_BLOCKS='1')[0])
        self.assertIn('M3_BLOCKS=2', self.problems(QWEN_FAST_M3_BLOCKS=None)[0])
        self.assertIn('one block', self.problems(users=4, QWEN_FAST_M3_BLOCKS='2')[0])
        for users in (1, 2, 3, 5, 16):
            with self.subTest(users=users):
                self.assertTrue(any('four or eight' in text for text in self.problems(users=users)))

    def test_lanes_lookup_eager_proposals_and_gather_experiments_are_refused(self):
        self.assertIn('QWEN_FAST_EAGER_PROPOSAL', self.problems(QWEN_FAST_EAGER_PROPOSAL='1')[0])
        self.assertEqual(self.problems(QWEN_FAST_EAGER_PROPOSAL='0'), [])
        for name, value in (('QWEN_FAST_LANE', '1'), ('QWEN_FAST_SOLO_LANE', '1'), ('QWEN_FAST_LOOKUP_DRAFT', 'rolling'),
                            ('QWEN_GDN_GROUPED_GATHER_ABBA', '1'), ('QWEN_GDN_GATE_EXP_ABBA', '1')):
            with self.subTest(name=name):
                self.assertIn(name, self.problems(**{name: value})[0])
        self.assertEqual(self.problems(QWEN_FAST_LANE='0', QWEN_FAST_LOOKUP_DRAFT='0', QWEN_GDN_GATE_EXP_ABBA='0'), [])

    def test_the_flag_is_strictly_zero_or_one(self):
        for value in ('2', 'true', 'yes', '', ' 1'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                    policy.parked_engines_enabled({FLAG: value})
                self.assertIn('must be 0 or 1', policy.parked_engine_problems({FLAG: value}, 8)[0])
        self.assertFalse(policy.parked_engines_enabled({}))
        self.assertTrue(policy.parked_engines_enabled({FLAG: '1'}))

    def test_unknown_parked_names_are_refused_flag_on_or_off_and_the_drafts_switch_needs_the_engines(self):
        for environ in ({'QWEN_FAST_PARKED_ENGINE': '1'}, {FLAG: '0', 'QWEN_FAST_PARKED_X': '0'}, dict(GOOD, QWEN_FAST_PARKED_TYPO='1')):
            with self.subTest(environ=environ):
                self.assertTrue(any('is not a parked-engines setting' in text for text in policy.parked_engine_problems(environ, 8)))
        self.assertEqual(policy.parked_engine_problems({'QWEN_FAST_PARKED_DRAFTS': '0'}, 8), [])
        self.assertEqual(policy.parked_engine_problems({'QWEN_FAST_PARKED_DRAFTS': '1'}, 8),
                         ['QWEN_FAST_PARKED_DRAFTS=1 needs QWEN_FAST_PARKED_ENGINES=1'])
        self.assertIn('QWEN_FAST_PARKED_DRAFTS', policy.parked_engine_problems({'QWEN_FAST_PARKED_DRAFTS': 'yes'}, 8)[0])

    def test_the_knob_values_are_checked_with_the_flag_on(self):
        for rows in ('0', '32', '256', '2048'):
            self.assertEqual(self.problems(QWEN_FAST_PARKED_PROJECT_ROWS=rows), [], rows)
        for rows in ('-1', '16', '33', '2080', 'x', '', '0256', '2.5'):
            with self.subTest(rows=rows):
                found = self.problems(QWEN_FAST_PARKED_PROJECT_ROWS=rows)
                self.assertEqual(len(found), 1, found)
                self.assertIn('PROJECT_ROWS', found[0])
        self.assertEqual(self.problems(QWEN_FAST_PARKED_AUDIT='1'), [])
        self.assertIn('QWEN_FAST_PARKED_AUDIT', self.problems(QWEN_FAST_PARKED_AUDIT='2')[0])
        for name in engines.NEGATIVES:
            self.assertEqual(self.problems(QWEN_FAST_PARKED_NEGATIVE=name), [], name)
        self.assertIn('NEGATIVE', self.problems(QWEN_FAST_PARKED_NEGATIVE='both')[0])
        for name in engines.FAULTS:
            self.assertEqual(self.problems(QWEN_FAST_PARKED_FAULT=name), [], name)
        self.assertIn('FAULT', self.problems(QWEN_FAST_PARKED_FAULT='unpark')[0])

    def test_the_names_match_the_module_that_reads_them(self):
        self.assertEqual(policy.PARKED_ENGINES_FLAG, engines.FLAG)
        self.assertEqual(policy.PARKED_DRAFTS_FLAG, engines.DRAFTS_FLAG)
        self.assertEqual(policy.PARKED_NEGATIVE_FLAG, engines.NEGATIVE_FLAG)
        self.assertEqual(policy.PARKED_NEGATIVES, engines.NEGATIVES)
        self.assertEqual(policy.PARKED_FAULT_FLAG, engines.FAULT_FLAG)
        self.assertEqual(policy.PARKED_FAULTS, engines.FAULTS)
        self.assertEqual(policy.GATE_DRAM_BALLAST_FLAG, engines.BALLAST_FLAG)
        self.assertEqual(tuple(policy.PARKED_NAMES), tuple(contract.PARKED_NAMES))
        self.assertEqual(set(policy.PARKED_NAMES), {engines.FLAG, engines.DRAFTS_FLAG, engines.PROJECT_ROWS_FLAG, engines.AUDIT_FLAG,
                                                    engines.NEGATIVE_FLAG, engines.FAULT_FLAG, engines.OFF_AFTER_FLAG, engines.OFF_PATH_ENV})
        self.assertEqual(set(contract.PARKED_GATE_ONLY), {engines.AUDIT_FLAG, engines.NEGATIVE_FLAG, engines.FAULT_FLAG, engines.BALLAST_FLAG,
                                                          engines.OFF_AFTER_FLAG, engines.OFF_PATH_ENV})
        self.assertEqual((contract.PARKED_SWITCH, contract.PARKED_DRAFTS), (engines.FLAG, engines.DRAFTS_FLAG))
        import serving_runtime
        import serving_worker_hook
        import dflash_packed_proposal_coordinator as coordinator

        self.assertEqual((serving_runtime.PARKED_ENGINES_FLAG, serving_runtime.PARKED_AUDIT_FLAG, serving_runtime.PARKED_DRAFTS_FLAG),
                         (engines.FLAG, engines.AUDIT_FLAG, engines.DRAFTS_FLAG))
        self.assertEqual((serving_worker_hook.PARKED_ENGINES_FLAG, coordinator.PARKED_ENGINES_FLAG), (engines.FLAG, engines.FLAG))


class ValidateFastConfigTests(unittest.TestCase):
    def environ(self, **values):
        clean = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_PARKED_')}
        clean.update(values)
        return mock.patch.dict(os.environ, clean, clear=True)

    def test_flag_off_parity(self):
        for seqs in (1, 4, 8):
            with self.environ():
                got = policy.validate_fast_config(config(seqs))
            self.assertEqual(got['scheduler_requests'], seqs)
        with self.environ(**{FLAG: '0'}):
            self.assertEqual(policy.validate_fast_config(config(8))['scheduler_requests'], 8)

    def test_flag_on_takes_the_shape_and_refuses_anything_else(self):
        with self.environ(**GOOD):
            self.assertEqual(policy.validate_fast_config(config(8))['scheduler_requests'], 8)
            with self.assertRaisesRegex(ValueError, 'Parked engines .*four or eight'):
                policy.validate_fast_config(config(2))
        with self.environ(**dict(GOOD, QWEN_FAST_VERIFY_T1='0')):
            with self.assertRaisesRegex(ValueError, 'QWEN_FAST_VERIFY_T1'):
                policy.validate_fast_config(config(8))

    def test_a_typo_is_refused_even_with_the_flag_off(self):
        with self.environ(QWEN_FAST_PARKED_TYPO='1'):
            with self.assertRaisesRegex(ValueError, 'QWEN_FAST_PARKED_TYPO is not a parked-engines setting'):
                policy.validate_fast_config(config(8))


class ProfileTests(unittest.TestCase):
    def test_the_checked_in_twins_are_what_their_parents_generate(self):
        text = Path(PROFILES).read_text(encoding='utf-8')
        self.assertEqual(twins.render(twins.generate(json.loads(text))), text,
                         'run scripts/ci/make_parked_profiles.py --write (never hand-merge the profiles file)')
        self.assertEqual(twins.main(['--check']), 0)

    def test_each_twin_is_its_parent_plus_exactly_its_flags(self):
        profiles = raw()['profiles']
        for name, parent_name, env, why in twins.specs():
            with self.subTest(profile=name):
                parent, child = profiles[parent_name], profiles[name]
                self.assertEqual(child['env'], dict(parent['env'], **env))
                for key in set(parent) | set(child):
                    if key not in ('env', 'description'):
                        self.assertEqual(child.get(key), parent.get(key), key)
                self.assertIs(child['gate_only'], True)
                self.assertIn('GATE ONLY', child['description'])
                self.assertIn('UNQUALIFIED', child['description'])
                self.assertIn(parent_name, child['description'])
                self.assertEqual(contract.parked_problems(dict(child, name=name)), [])
                self.assertEqual(contract.prefix_reuse_problems(child), [])
                self.assertEqual(contract.levern_problems(child), [])

    def test_a_twin_argv_and_limits_are_its_parents(self):
        profiles = raw()['profiles']
        for name, parent_name, env, why in twins.specs():
            with self.subTest(profile=name):
                self.assertEqual(contract.engine_arguments(load(name), '/snap'), contract.engine_arguments(load(parent_name), '/snap'))
                self.assertEqual(contract.request_limits(load(name)), contract.request_limits(load(parent_name)))
                self.assertEqual(contract.parser_rechunk(load(name)), contract.parser_rechunk(load(parent_name)))

    def test_the_candidate_the_e1_arm_and_the_controls(self):
        profiles = raw()['profiles']
        candidate = profiles[twins.CANDIDATE_PARENT + '-parked']['env']
        self.assertEqual((candidate[FLAG], candidate['QWEN_FAST_PARKED_DRAFTS'], candidate['QWEN_FAST_LEVERN_BUILD_MS']), ('1', '1', 'learned'))
        e1 = profiles[twins.CANDIDATE_PARENT + '-parked-e1']['env']
        self.assertEqual(e1[FLAG], '1')
        self.assertNotIn('QWEN_FAST_PARKED_DRAFTS', e1)
        self.assertNotIn('QWEN_FAST_LEVERN_BUILD_MS', e1)
        audit = profiles[twins.CANDIDATE_PARENT + '-parked-audit']['env']
        self.assertEqual((audit['QWEN_FAST_PARKED_AUDIT'], audit['QWEN_FAST_CCL_HANDLE_GUARD'], audit['QWEN_FAST_DRAFT_SINGLES_AUDIT']),
                         ('1', 'log', 'all'))
        control = profiles[twins.AUDIT_PARENT + '-r2']['env']
        self.assertEqual((control['QWEN_FAST_PARKED_AUDIT'], control['QWEN_FAST_CCL_HANDLE_GUARD']), ('1', 'log'))
        self.assertNotIn(FLAG, control, 'the flag-off control carries the ledger and no parked engines')
        negatives = {name: profiles[profile]['env']['QWEN_FAST_PARKED_NEGATIVE'] for name, profile in (
            (name, twins.CANDIDATE_PARENT + ('-parked-audit-neg-drafter' if name == 'drafter' else '-parked-neg-' + name))
            for name in twins.NEGATIVES)}
        self.assertEqual(negatives, {name: name for name in twins.NEGATIVES})
        self.assertEqual(sorted(twins.NEGATIVES), sorted(engines.NEGATIVES))
        self.assertEqual(sorted(twins.FAULTS), sorted(engines.FAULTS))
        ballast = sorted(int(profiles[twins.CANDIDATE_PARENT + '-parked-ballast-%d' % megabytes]['env']['QWEN_FAST_GATE_DRAM_BALLAST'])
                         for megabytes in twins.BALLAST_MB)
        self.assertEqual(ballast, [megabytes * 10 ** 6 for megabytes in twins.BALLAST_MB])

    def test_only_the_generated_twins_name_any_parked_flag(self):
        profiles = raw()['profiles']
        generated = {name for name, parent, env, why in twins.specs()}
        # the octo-T8 twins of the engine-reuse parent carry its flags by inheritance (test_octo_profiles holds each as that parent plus its own flags)
        generated |= {name for name, parent, env, third, why in octo_twins.specs() if parent == octo_twins.PARKED_PARENT}
        # the W2 kill drill twin of the ship profile carries the parked flags by inheritance (test_w2_switch holds it as that parent plus its two names)
        import make_w2_kill_profiles

        generated |= set(make_w2_kill_profiles.twin_names())
        # the op-fusion programme's twins (make_fusion_profiles; test_fusion_wp holds each as the production profile plus exactly its flags) carry its parked flags by inheritance
        import make_fusion_profiles

        generated |= set(make_fusion_profiles.twin_names())
        # the region-read audit twins (make_kvread_profiles; test_kvread_profiles holds each as its parent plus one flag) carry their parent's flags by inheritance
        region_read = {twin: parent for parent, twin, mode in kvread_twins.TWINS}
        for name, profile in profiles.items():
            env = profile['env']
            named = [flag for flag in env if flag.startswith('QWEN_FAST_PARKED_') or flag in ('QWEN_FAST_GATE_DRAM_BALLAST', 'QWEN_FAST_LEVERN_BUILD_MS')]
            with self.subTest(profile=name):
                if name in region_read:
                    parent_env = profiles[region_read[name]]['env']
                    self.assertEqual(sorted(named), sorted(flag for flag in parent_env if flag.startswith('QWEN_FAST_PARKED_')
                                                           or flag in ('QWEN_FAST_GATE_DRAM_BALLAST', 'QWEN_FAST_LEVERN_BUILD_MS')),
                                     'a region-read twin names exactly its parent\'s engine-reuse flags')
                elif name in generated:
                    self.assertTrue(named or name.endswith('-r2'))
                elif 'owner_traffic_waiver' in profile:
                    # the ship candidate (test_ship_ln_w2_er): the -parked twin's three flags on TRAFFIC, the two parked switches named by its owner waiver
                    self.assertEqual(sorted(named), sorted(['QWEN_FAST_PARKED_ENGINES', 'QWEN_FAST_PARKED_DRAFTS', 'QWEN_FAST_LEVERN_BUILD_MS']))
                    self.assertEqual({flag: env[flag] for flag in named if flag.startswith('QWEN_FAST_PARKED_')},
                                     {flag: value for flag, value in profile['owner_traffic_waiver']['levers'].items() if flag.startswith('QWEN_FAST_PARKED_')})
                else:
                    self.assertEqual(named, [], 'a profile that is not a generated twin names no engine-reuse flag')
        for name in (PRODUCTION, PRODUCTION + '-audit', PRODUCTION + '-levern', PRODUCTION + '-levern-audit'):
            self.assertFalse([flag for flag in profiles[name]['env'] if flag.startswith('QWEN_FAST_PARKED_')])

    def test_the_production_profile_is_untouched_and_is_not_a_gate(self):
        profile = raw()['profiles'][PRODUCTION]
        self.assertFalse(profile.get('gate_only'))
        self.assertFalse([flag for flag in profile['env'] if 'PARKED' in flag or 'BALLAST' in flag or 'BUILD_MS' in flag])

    def test_the_kv_pool_and_the_trace_region_the_margins_were_derived_under_are_pinned(self):
        profiles = raw()['profiles']
        for name, parent_name, env, why in twins.specs():
            engine = profiles[name]['engine']
            with self.subTest(profile=name):
                self.assertEqual(engine['num-gpu-blocks-override'], twins.KV_BLOCKS)
                self.assertEqual(engine['additional-config']['tt']['trace_region_size'], twins.TRACE_REGION)
                self.assertEqual(engine['max-num-seqs'], twins.SEATS)

    def test_a_parent_whose_kv_pool_or_trace_region_moved_is_refused_by_the_generator(self):
        data = raw()
        for mutate, text in ((lambda engine: engine.update({'num-gpu-blocks-override': 24000}), 'num-gpu-blocks-override'),
                             (lambda engine: engine['additional-config']['tt'].update({'trace_region_size': 600000000}), 'trace_region_size'),
                             (lambda engine: engine.update({'max-num-seqs': 4}), 'max-num-seqs')):
            moved = copy.deepcopy(data)
            mutate(moved['profiles'][twins.CANDIDATE_PARENT]['engine'])
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, text):
                twins.generate(moved)
        moved = copy.deepcopy(data)
        moved['profiles'][twins.CANDIDATE_PARENT]['env']['QWEN_FAST_PARKED_ENGINES'] = '1'
        with self.assertRaisesRegex(ValueError, 'already names'):
            twins.generate(moved)

    def test_the_profile_env_and_the_image_env_make_the_shape_the_policy_asks_for(self):
        path = HERE.parent.parent / 'docker' / 'qwen-c2-serving.Dockerfile'
        if not path.is_file():
            self.skipTest('repository checkout only')
        joined = path.read_text(encoding='utf-8').replace(chr(92) + chr(10), ' ')
        image = dict(token.partition('=')[::2] for line in joined.split(chr(10)) if line.startswith('ENV ')
                     for token in line[4:].split() if '=' in token)
        for name, parent, env, why in twins.specs():
            if FLAG not in env:
                continue
            with self.subTest(profile=name):
                environ = contract.apply_environment(load(name), dict(image))
                self.assertEqual(policy.parked_engine_problems(environ, load(name)['engine']['max-num-seqs']), [])
        self.assertNotIn(FLAG, contract.apply_environment(load(PRODUCTION), dict(image, **{FLAG: '1'})))
        self.assertNotIn('QWEN_FAST_PARKED_DRAFTS', contract.apply_environment(load(PRODUCTION), dict(image, QWEN_FAST_PARKED_DRAFTS='1')))
        for name in contract.PARKED_NAMES + contract.PARKED_GATE_ONLY:
            self.assertNotIn(name, path.read_text(encoding='utf-8'), 'the image bakes no parked or gate knob')


class ContractTests(unittest.TestCase):
    def test_the_switch_needs_the_gate_only_four_card_s2_shape_in_the_profile(self):
        name = twins.CANDIDATE_PARENT + '-parked'
        base = dict(load(name), name=name)
        for key in ('QWEN_FAST_ANY_REQUEST', 'QWEN_FAST_EXTENT_REPLAY', 'QWEN_FAST_TP'):
            profile = copy.deepcopy(base)
            del profile['env'][key]
            with self.subTest(dropped=key):
                found = contract.parked_problems(profile)
                self.assertEqual(len(found), 1, found)
                self.assertIn(key, found[0])
        profile = copy.deepcopy(base)
        profile['engine']['max-num-seqs'] = 2
        self.assertIn('max-num-seqs 4 or 8', contract.parked_problems(profile)[0])
        profile = copy.deepcopy(base)
        profile['engine']['additional-config'] = {}
        self.assertIn('qwen_fast_t16', contract.parked_problems(profile)[0])
        profile = copy.deepcopy(base)
        profile['env'][FLAG] = '2'
        self.assertIn('neither 1 nor 0', contract.parked_problems(profile)[0])
        profile = copy.deepcopy(base)
        del profile['gate_only']
        self.assertIn('gate-only profile', contract.parked_problems(profile)[0])
        profile = copy.deepcopy(base)
        del profile['env'][FLAG]
        self.assertEqual(contract.parked_problems(profile), ['QWEN_FAST_PARKED_DRAFTS=1 needs QWEN_FAST_PARKED_ENGINES=1'])

    def test_gate_instruments_are_refused_in_a_traffic_profile_and_its_process_environment(self):
        traffic = dict(load(PRODUCTION), name=PRODUCTION)
        for knob in contract.PARKED_GATE_ONLY:
            broken = copy.deepcopy(traffic)
            broken['env'][knob] = '1'
            with self.subTest(knob=knob, where='traffic profile env'):
                self.assertTrue(any('gate instrument' in text for text in contract.parked_problems(broken)))
            with self.subTest(knob=knob, where='traffic process environment'):
                self.assertTrue(any('outside a gate profile' in text for text in contract.parked_problems(traffic, {knob: '1'})))
            with self.subTest(knob=knob, where='gate profile environment'):
                gate = dict(load(twins.CANDIDATE_PARENT + '-parked'), name=twins.CANDIDATE_PARENT + '-parked')
                self.assertEqual(contract.parked_problems(gate, {knob: '1'}), [])

    def test_an_unknown_parked_name_in_a_profile_is_refused(self):
        profile = copy.deepcopy(dict(load(PRODUCTION), name=PRODUCTION))
        profile['env']['QWEN_FAST_PARKED_TYPO'] = '1'
        self.assertIn('QWEN_FAST_PARKED_TYPO', contract.parked_problems(profile)[0])

    def test_apply_environment_leaves_the_switches_to_the_profile(self):
        for name, expected in ((PRODUCTION, None), (twins.CANDIDATE_PARENT + '-parked', '1'), (twins.CANDIDATE_PARENT + '-parked-e1', '1')):
            with self.subTest(profile=name):
                environ = contract.apply_environment(load(name), {FLAG: '1' if expected is None else '0', 'QWEN_FAST_PARKED_DRAFTS': '1'})
                self.assertEqual(environ.get(FLAG), expected)
        e1 = contract.apply_environment(load(twins.CANDIDATE_PARENT + '-parked-e1'), {'QWEN_FAST_PARKED_DRAFTS': '1'})
        self.assertNotIn('QWEN_FAST_PARKED_DRAFTS', e1, 'an inherited drafts switch never reaches an engines-only arm')
        # every other name of the family, and the ballast, is the profile's too: a stray process value never turns an arm into a negative or a fault
        stray = {name: '1' for name in set(contract.PARKED_NAMES) | set(contract.PARKED_GATE_ONLY)}
        for name in (twins.CANDIDATE_PARENT + '-parked', PRODUCTION):
            with self.subTest(profile=name, what='stray gate knobs'):
                environ = contract.apply_environment(load(name), dict(stray))
                profile_env = load(name)['env']
                self.assertEqual(sorted(key for key in stray if key in environ and key not in profile_env), [])

    def test_the_kill_switch_trigger_is_a_gate_instrument_with_a_strict_value(self):
        import serving_parked_engines as engines

        self.assertIsNone(engines.off_after({}))
        self.assertIsNone(engines.off_after({engines.OFF_AFTER_FLAG: ''}))
        self.assertEqual(engines.off_after({engines.OFF_AFTER_FLAG: '3'}), 3)
        for bad in ('0', '-1', '03', 'soon', '1.5'):
            with self.assertRaises(ValueError):
                engines.off_after({engines.OFF_AFTER_FLAG: bad})

    def test_the_governor_switch_is_a_lever_n_flag_the_contract_owns(self):
        import levern_policy

        self.assertIn('QWEN_FAST_LEVERN_BUILD_MS', contract.LEVERN_ENV_FLAGS)
        self.assertEqual(sorted(contract.LEVERN_ENV_FLAGS), sorted(levern_policy.ALL_FLAGS))
        broken = copy.deepcopy(dict(load(PRODUCTION + '-levern'), name='x'))
        broken['env']['QWEN_FAST_LEVERN_BUILD_MS'] = 'whenever'
        self.assertTrue(any('QWEN_FAST_LEVERN_BUILD_MS' in text for text in contract.levern_problems(broken)))
        without = copy.deepcopy(dict(load(PRODUCTION), name='x'))
        without['env']['QWEN_FAST_LEVERN_BUILD_MS'] = 'learned'
        self.assertTrue(any('QWEN_FAST_LEVERN_BUILD_MS set without' in text for text in contract.levern_problems(without)))


def boot(name, extra=None):
    """contract.boot under profile `name` as the .pth hook runs it, with the process state it touches restored: the profile, or the ValueError."""
    environ = {'QWEN_C2_SERVING': '1', 'QWEN_C2_PROFILES': PROFILES, 'QWEN_C2_GATE': '1'}
    environ.update(extra or {})
    saved = list(sys.argv), list(sys.meta_path), list(sys.path)
    try:
        with mock.patch.object(contract, 'install_prefix_metrics', lambda api_server: None), \
                mock.patch.object(contract, 'install_teardown_skip', lambda: None), \
                mock.patch.object(contract, 'install_levern_platform', lambda: None), \
                mock.patch.object(contract, 'resolve_snapshot', lambda profile: profile['snapshots'][0]), \
                mock.patch.object(contract, 'log', lambda *a: None), \
                mock.patch.dict(os.environ, {'QWEN_C2_PROFILE': name}):
            sys.argv[:] = ['-m', '--port', '8001']
            try:
                return contract.boot(environ=environ, orig_argv=['python3', '-m', contract.API_SERVER])
            except ValueError as error:
                return error
    finally:
        sys.argv[:], sys.meta_path[:], sys.path[:] = saved


class BootTests(unittest.TestCase):
    def test_the_parked_profiles_boot_only_as_a_gate(self):
        name = twins.CANDIDATE_PARENT + '-parked'
        result = boot(name)
        self.assertIsInstance(result, dict, result)
        self.assertEqual(result['name'], name)
        refused = boot(name, {'QWEN_C2_GATE': '0'})
        self.assertIsInstance(refused, ValueError)

    def test_a_gate_instrument_reaches_only_a_gate_profile(self):
        for knob in contract.PARKED_GATE_ONLY:
            with self.subTest(knob=knob, profile='production'):
                result = boot(PRODUCTION, {knob: '1'})
                self.assertIsInstance(result, ValueError)
                self.assertIn('outside a gate profile', str(result))
            with self.subTest(knob=knob, profile='gate'):
                self.assertIsInstance(boot(twins.CANDIDATE_PARENT + '-parked', {knob: '1'}), dict)

    def test_the_production_profile_boots_as_before(self):
        self.assertIsInstance(boot(PRODUCTION, {'QWEN_C2_GATE': '0'}), dict)


if __name__ == '__main__':
    unittest.main()
