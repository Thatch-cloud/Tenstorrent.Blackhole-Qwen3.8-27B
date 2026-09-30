"""Stage E, E6: the policy, the contract and the profiles of QWEN_FAST_PARKED_ENGINES.

The policy matrix (serving_fast_policy.parked_engine_problems, and validate_fast_config's use of it), the contract's
side (serving_c2_contract.parked_problems, apply_environment's ownership of the switch, the boot's refusals),
the c2-packed-prefix-parked profiles held to their sticky twins, and flag-off parity: with no QWEN_FAST_PARKED_
name in the environment, validate_fast_config is what it was.
"""
import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import serving_c2_contract as contract
import serving_fast_policy as policy

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES = os.path.join(HERE, 'qwen_c2_profiles.json')
FLAG = 'QWEN_FAST_PARKED_ENGINES'

# The environment the S2 image and a c2-packed-prefix profile give a process: what the parked engines need.
GOOD = {'QWEN_FAST_ANY_REQUEST': '1', 'QWEN_FAST_EXTENT_REPLAY': '1', 'QWEN_FAST_PACKED_STEP': '1',
        'QWEN_FAST_SHARED_CCL': '1', 'QWEN_FAST_VERIFY_T1': '1', FLAG: '1'}


def load(name):
    return contract.load_profile(PROFILES, name)


def config(seqs=4):
    return SimpleNamespace(additional_config={'qwen_fast_t16': True},
        scheduler_config=SimpleNamespace(max_num_seqs=seqs, async_scheduling=False, scheduler_cls='x'),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1),
        cache_config=SimpleNamespace(block_size=64, enable_prefix_caching=False),
        lora_config=None, model_config=SimpleNamespace(max_model_len=4352),
        speculative_config=SimpleNamespace(method='dflash', num_speculative_tokens=15,
                                           draft_sample_method='greedy', rejection_sample_method='standard'))


class PolicyMatrixTests(unittest.TestCase):
    def problems(self, **changes):
        environ = dict(GOOD)
        for name, value in changes.items():
            if value is None:
                environ.pop(name, None)
            else:
                environ[name] = value
        return policy.parked_engine_problems(environ, 4)

    def test_the_s2_shape_passes(self):
        self.assertEqual(policy.parked_engine_problems(dict(GOOD), 4), [])

    def test_each_requirement_is_refused_by_name(self):
        for name, bad in (('QWEN_FAST_ANY_REQUEST', '0'), ('QWEN_FAST_EXTENT_REPLAY', None),
                          ('QWEN_FAST_PACKED_STEP', '0'), ('QWEN_FAST_PACKED_STEP', None),
                          ('QWEN_FAST_SHARED_CCL', '0'), ('QWEN_FAST_SHARED_CCL', None),
                          ('QWEN_FAST_VERIFY_T1', None), ('QWEN_FAST_VERIFY_T1', '0')):
            with self.subTest(name=name, bad=bad):
                found = self.problems(**{name: bad})
                self.assertEqual(len(found), 1, found)
                self.assertIn(name, found[0])

    def test_users_other_than_four_are_refused(self):
        for users in (1, 2, 8):
            with self.subTest(users=users):
                found = policy.parked_engine_problems(dict(GOOD), users)
                self.assertEqual(len(found), 1)
                self.assertIn('max_num_seqs 4', found[0])

    def test_eager_proposals_and_gather_experiments_are_refused(self):
        self.assertIn('QWEN_FAST_EAGER_PROPOSAL', self.problems(QWEN_FAST_EAGER_PROPOSAL='1')[0])
        self.assertEqual(self.problems(QWEN_FAST_EAGER_PROPOSAL='0'), [])
        for name in ('QWEN_GDN_GROUPED_GATHER_ABBA', 'QWEN_GDN_GATE_EXP_ABBA'):
            with self.subTest(name=name):
                self.assertIn(name, self.problems(**{name: '1'})[0])
                self.assertEqual(self.problems(**{name: '0'}), [])

    def test_the_flag_is_strictly_zero_or_one(self):
        for value in ('2', 'true', 'yes', '', ' 1'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                    policy.parked_engines_enabled({FLAG: value})
                self.assertIn('must be 0 or 1', policy.parked_engine_problems({FLAG: value}, 4)[0])
        self.assertFalse(policy.parked_engines_enabled({}))
        self.assertFalse(policy.parked_engines_enabled({FLAG: '0'}))
        self.assertTrue(policy.parked_engines_enabled({FLAG: '1'}))

    def test_unknown_parked_names_are_refused_flag_on_or_off(self):
        for environ in ({'QWEN_FAST_PARKED_DRAFTS': '1'}, {'QWEN_FAST_PARKED_ENGINE': '1'},
                        dict(GOOD, QWEN_FAST_PARKED_DRAFTS='1'), {FLAG: '0', 'QWEN_FAST_PARKED_X': '0'}):
            with self.subTest(environ=environ):
                found = policy.parked_engine_problems(environ, 4)
                self.assertTrue(any('is not a parked-engines setting' in text for text in found), found)

    def test_the_knob_values_are_checked_with_the_flag_on(self):
        good_rows = ('0', '32', '256', '2048')
        for rows in good_rows:
            self.assertEqual(policy.parked_engine_problems(dict(GOOD, QWEN_FAST_PARKED_PROJECT_ROWS=rows), 4), [], rows)
        for rows in ('-1', '16', '33', '2080', 'x', '', '0256', '2.5'):
            with self.subTest(rows=rows):
                found = policy.parked_engine_problems(dict(GOOD, QWEN_FAST_PARKED_PROJECT_ROWS=rows), 4)
                self.assertEqual(len(found), 1, found)
                self.assertIn('PROJECT_ROWS', found[0])
        self.assertEqual(policy.parked_engine_problems(dict(GOOD, QWEN_FAST_PARKED_AUDIT='1'), 4), [])
        self.assertIn('QWEN_FAST_PARKED_AUDIT', policy.parked_engine_problems(dict(GOOD, QWEN_FAST_PARKED_AUDIT='2'), 4)[0])
        for value in ('carry', 'drafter', '', None):
            environ = dict(GOOD)
            if value is not None:
                environ['QWEN_FAST_PARKED_NEGATIVE'] = value
            self.assertEqual(policy.parked_engine_problems(environ, 4), [], value)
        found = policy.parked_engine_problems(dict(GOOD, QWEN_FAST_PARKED_NEGATIVE='both'), 4)
        self.assertEqual(len(found), 1)
        self.assertIn('NEGATIVE', found[0])
        for value in ('park', '', None):
            environ = dict(GOOD)
            if value is not None:
                environ['QWEN_FAST_PARKED_FAULT'] = value
            self.assertEqual(policy.parked_engine_problems(environ, 4), [], value)
        found = policy.parked_engine_problems(dict(GOOD, QWEN_FAST_PARKED_FAULT='unpark'), 4)
        self.assertEqual(len(found), 1)
        self.assertIn('FAULT', found[0])

    def test_with_the_flag_off_only_the_names_are_checked(self):
        """Off, nothing about the shape is asked: a c2 profile with no packed block sets none of it."""
        self.assertEqual(policy.parked_engine_problems({}, 1), [])
        self.assertEqual(policy.parked_engine_problems({FLAG: '0', 'QWEN_FAST_PARKED_PROJECT_ROWS': 'junk'}, 1), [])

    def test_the_names_match_the_module_that_reads_them(self):
        import serving_parked_engines as engines

        self.assertEqual(policy.PARKED_ENGINES_FLAG, engines.FLAG)
        self.assertIn(engines.PROJECT_ROWS_FLAG, policy.PARKED_NAMES)
        self.assertIn(engines.AUDIT_FLAG, policy.PARKED_NAMES)
        self.assertNotIn(engines.DEFERRED_DRAFTS_FLAG, policy.PARKED_NAMES)
        self.assertEqual(policy.PARKED_NEGATIVE_FLAG, engines.NEGATIVE_FLAG)
        self.assertEqual(policy.PARKED_NEGATIVES, engines.NEGATIVES)
        self.assertEqual(policy.GATE_DRAM_BALLAST_FLAG, engines.BALLAST_FLAG)
        self.assertEqual(policy.PARKED_FAULT_FLAG, engines.FAULT_FLAG)
        self.assertEqual(policy.PARKED_FAULTS, engines.FAULTS)
        self.assertIn(engines.NEGATIVE_FLAG, policy.PARKED_NAMES)
        self.assertIn(engines.FAULT_FLAG, policy.PARKED_NAMES)
        self.assertEqual(set(policy.PARKED_NAMES) | {engines.BALLAST_FLAG}, set(contract.PARKED_NAMES) | {engines.BALLAST_FLAG})
        self.assertEqual(set(contract.PARKED_GATE_ONLY),
                         {engines.AUDIT_FLAG, engines.NEGATIVE_FLAG, engines.FAULT_FLAG, engines.BALLAST_FLAG})
        self.assertEqual(tuple(policy.PARKED_NAMES), tuple(contract.PARKED_NAMES))
        self.assertEqual(policy.PARKED_ENGINES_FLAG, contract.PARKED_SWITCH)


class ValidateFastConfigTests(unittest.TestCase):
    def environ(self, **values):
        clean = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_PARKED_')}
        clean.update(values)
        return mock.patch.dict(os.environ, clean, clear=True)

    def test_flag_off_parity(self):
        """No QWEN_FAST_PARKED_ name: the policy's answer is exactly what it was, at any user count."""
        for seqs in (1, 4, 8):
            with self.environ():
                got = policy.validate_fast_config(config(seqs))
            self.assertEqual(got['scheduler_requests'], seqs)
            self.assertFalse(got['serving_qualified'])
        with self.environ(**{FLAG: '0'}):
            self.assertEqual(policy.validate_fast_config(config(4))['scheduler_requests'], 4)

    def test_flag_on_takes_the_s2_shape_and_refuses_anything_else(self):
        with self.environ(**GOOD):
            self.assertEqual(policy.validate_fast_config(config(4))['scheduler_requests'], 4)
            with self.assertRaisesRegex(ValueError, 'Parked engines .*max_num_seqs 4'):
                policy.validate_fast_config(config(2))
        with self.environ(**dict(GOOD, QWEN_FAST_VERIFY_T1='0')):
            with self.assertRaisesRegex(ValueError, 'QWEN_FAST_VERIFY_T1'):
                policy.validate_fast_config(config(4))
        with self.environ(**{FLAG: '1'}):
            with self.assertRaisesRegex(ValueError, 'QWEN_FAST_ANY_REQUEST'):
                policy.validate_fast_config(config(4))

    def test_a_typo_is_refused_even_with_the_flag_off(self):
        with self.environ(QWEN_FAST_PARKED_DRAFTS='1'):
            with self.assertRaisesRegex(ValueError, 'QWEN_FAST_PARKED_DRAFTS is not a parked-engines setting'):
                policy.validate_fast_config(config(4))


class ProfileTests(unittest.TestCase):
    PAIRS = (('c2-packed-prefix', 'c2-packed-prefix-parked'), ('c2-packed-prefix-gate', 'c2-packed-prefix-parked-gate'))

    def test_each_parked_profile_is_its_sticky_twin_plus_the_flag(self):
        for twin, parked in self.PAIRS:
            with self.subTest(profile=parked):
                a, b = load(twin), load(parked)
                self.assertEqual(b['env'], dict(a['env'], **{FLAG: '1'}))
                self.assertNotIn(FLAG, a['env'])
                for key in set(a) | set(b):
                    if key not in ('env', 'description', 'name'):
                        self.assertEqual(b.get(key), a.get(key), key)
                self.assertIn('Stage E', b['description'])
                self.assertIn('NOT QUALIFIED', b['description'])

    def test_the_parked_profiles_serve_the_twins_argv_and_limits(self):
        for twin, parked in self.PAIRS:
            with self.subTest(profile=parked):
                self.assertEqual(contract.engine_arguments(load(parked), '/snap'),
                                 contract.engine_arguments(load(twin), '/snap'))
                self.assertEqual(contract.request_limits(load(parked)), contract.request_limits(load(twin)))
                self.assertEqual(contract.parser_rechunk(load(parked)), contract.parser_rechunk(load(twin)))
                self.assertTrue(contract.sticky_sessions(load(parked)))
                self.assertTrue(contract.prefix_reuse(load(parked)))
                self.assertEqual(contract.prefix_reuse_problems(load(parked)), [])
                self.assertEqual(contract.parked_problems(load(parked)), [])
                self.assertTrue(contract.parked_engines(load(parked)))
                self.assertFalse(contract.parked_engines(load(twin)))

    def test_only_the_two_parked_profiles_set_the_flag(self):
        with open(PROFILES, encoding='utf-8') as handle:
            names = sorted(json.load(handle)['profiles'])
        self.assertEqual([name for name in names if FLAG in load(name)['env']],
                         ['c2-packed-prefix-parked', 'c2-packed-prefix-parked-gate'])
        for name in names:
            env = load(name)['env']
            with self.subTest(profile=name):
                for knob in contract.PARKED_GATE_ONLY:
                    self.assertNotIn(knob, env)
                self.assertEqual(contract.parked_problems(load(name)), [])

    def test_the_profile_env_and_the_image_env_make_the_shape_the_policy_asks_for(self):
        """The image's ENV (the frozen v235 environment) then the parked profile's, through the contract's own
        apply_environment: what parked_engine_problems reads at the policy."""
        path = os.path.join(HERE, '..', '..', 'docker', 'qwen-c2-serving.Dockerfile')
        if not os.path.isfile(path):
            self.skipTest('repository checkout only')
        with open(path, encoding='utf-8') as handle:
            joined = handle.read().replace(chr(92) + chr(10), ' ')
        image = dict(token.partition('=')[::2] for line in joined.split(chr(10)) if line.startswith('ENV ')
                     for token in line[4:].split() if '=' in token)
        for name in ('c2-packed-prefix-parked', 'c2-packed-prefix-parked-gate'):
            with self.subTest(profile=name):
                environ = contract.apply_environment(load(name), dict(image))
                self.assertEqual(policy.parked_engine_problems(environ, load(name)['engine']['max-num-seqs']), [])
        for name in ('c2-packed-prefix', 'c2-packed-prefix-gate', 'c2-packed', 'c2'):
            with self.subTest(profile=name):
                self.assertNotIn(FLAG, contract.apply_environment(load(name), dict(image, **{FLAG: '1'})))

    def test_the_image_bakes_no_parked_or_gate_knob(self):
        path = os.path.join(HERE, '..', '..', 'docker', 'qwen-c2-serving.Dockerfile')
        if not os.path.isfile(path):
            self.skipTest('repository checkout only')
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
        for name in contract.PARKED_NAMES + contract.PARKED_GATE_ONLY:
            self.assertNotIn(name, text)


class ContractTests(unittest.TestCase):
    def test_the_switch_needs_the_s2_shape_in_the_profile(self):
        base = load('c2-packed-prefix-parked')
        cases = (('env', 'QWEN_FAST_ANY_REQUEST', 'QWEN_FAST_ANY_REQUEST'),
                 ('env', 'QWEN_FAST_EXTENT_REPLAY', 'QWEN_FAST_EXTENT_REPLAY'))
        for section, key, wanted in cases:
            profile = json.loads(json.dumps(base))
            del profile[section][key]
            with self.subTest(dropped=key):
                found = contract.parked_problems(profile)
                self.assertEqual(len(found), 1, found)
                self.assertIn(wanted, found[0])
        profile = json.loads(json.dumps(base))
        profile['engine']['max-num-seqs'] = 2
        self.assertIn('max-num-seqs 4', contract.parked_problems(profile)[0])
        profile = json.loads(json.dumps(base))
        profile['engine']['additional-config'] = {}
        self.assertIn('qwen_fast_t16', contract.parked_problems(profile)[0])
        profile = json.loads(json.dumps(base))
        profile['env'][FLAG] = '2'
        self.assertIn('neither 1 nor 0', contract.parked_problems(profile)[0])

    def test_gate_knobs_are_refused_in_a_profile_and_outside_a_gate_profile(self):
        profile = json.loads(json.dumps(load('c2-packed-prefix-parked')))
        for knob in contract.PARKED_GATE_ONLY:
            broken = json.loads(json.dumps(profile))
            broken['env'][knob] = '1'
            with self.subTest(knob=knob, where='profile'):
                self.assertTrue(any('gate knob' in text for text in contract.parked_problems(broken)))
            with self.subTest(knob=knob, where='environment, traffic profile'):
                self.assertTrue(any('outside a gate profile' in text
                                    for text in contract.parked_problems(profile, {knob: '1'})))
            with self.subTest(knob=knob, where='environment, gate profile'):
                self.assertEqual(contract.parked_problems(load('c2-packed-prefix-parked-gate'), {knob: '1'}), [])

    def test_an_unknown_parked_name_in_a_profile_is_refused(self):
        profile = json.loads(json.dumps(load('c2-packed-prefix')))
        profile['env']['QWEN_FAST_PARKED_DRAFTS'] = '1'
        self.assertIn('QWEN_FAST_PARKED_DRAFTS', contract.parked_problems(profile)[0])

    def test_apply_environment_leaves_an_inherited_value_to_the_profile(self):
        for name, expected in (('c2-packed-prefix', None), ('c2-packed-prefix-parked', '1'), ('exact', None)):
            with self.subTest(profile=name):
                environ = contract.apply_environment(load(name), {FLAG: '1' if expected is None else '0'})
                self.assertEqual(environ.get(FLAG), expected)


def boot(name, extra=None):
    """contract.boot under profile `name` as the .pth hook runs it, with the process state it touches restored;
    the profile, or the ValueError it refused with."""
    environ = {'QWEN_C2_SERVING': '1', 'QWEN_C2_PROFILES': PROFILES}
    environ.update(extra or {})
    saved = list(sys.argv), list(sys.meta_path), list(sys.path)
    try:
        with mock.patch.object(contract, 'install_prefix_metrics', lambda api_server: None), \
                mock.patch.object(contract, 'install_teardown_skip', lambda: None), \
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
    def test_the_parked_profiles_boot(self):
        for name in ('c2-packed-prefix-parked', 'c2-packed-prefix-parked-gate'):
            with self.subTest(profile=name):
                extra = {'QWEN_C2_GATE': '1'} if name.endswith('-gate') else {}
                result = boot(name, extra)
                self.assertIsInstance(result, dict, result)
                self.assertEqual(result['name'], name)

    def test_a_gate_knob_reaches_only_a_gate_profile(self):
        for knob in contract.PARKED_GATE_ONLY:
            with self.subTest(knob=knob, profile='c2-packed-prefix-parked'):
                result = boot('c2-packed-prefix-parked', {knob: '1'})
                self.assertIsInstance(result, ValueError)
                self.assertIn('cannot serve parked engines', str(result))
                self.assertIn(knob, str(result))
            with self.subTest(knob=knob, profile='c2-packed-prefix-parked-gate'):
                self.assertIsInstance(boot('c2-packed-prefix-parked-gate', {knob: '1', 'QWEN_C2_GATE': '1'}), dict)

    def test_the_other_profiles_boot_as_before(self):
        for name in ('c2-packed-prefix', 'c2-packed', 'general-prefix', 'exact'):
            with self.subTest(profile=name):
                self.assertIsInstance(boot(name), dict)


if __name__ == '__main__':
    unittest.main()
