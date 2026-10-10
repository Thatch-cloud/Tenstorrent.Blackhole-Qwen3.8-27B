"""Lever N at TP4: the profiles, the contract's refusals and the boot hook (serving_c2_contract.levern_problems and friends).

Six gate-only eight-seat 262k profiles (docs/lever-n-tp4-design-2026-10-04.md section 4), each a twin of an existing one plus exactly the flags it names:
  c2-packed-tp4-8x262k-best-levern-time-gate      = -best-time-gate, chunked engine argv, QWEN_FAST_LEVER_N=1, the step variables, share 0.5   (the timed arm)
  c2-packed-tp4-8x262k-best-levern-r1-time-gate   = the same with QWEN_FAST_LEVERN_ROUNDS=1 in place of the share                                (the static arm)
  c2-packed-tp4-8x262k-best-levern-audit          = -best-audit, chunked engine argv, QWEN_FAST_LEVER_N=1, share 0.5, QWEN_FAST_LEVERN_AUDIT=1  (the exactness arm)
  c2-packed-tp4-8x262k-best-levern-control-audit  = -best-audit plus QWEN_FAST_LEVERN_AUDIT=1 alone, whole-prompt prefill                          (its control)
  c2-packed-tp4-8x262k-best-levern-final-hold-time-gate / -foreign-time-gate = the timed arm plus QWEN_FAST_LEVERN_FAULT=final-hold / foreign   (negative controls)
Every other profile is untouched (test_c2_packed_tp4_262k_profiles holds the 262k twins by digest) and none of them names a Lever N flag."""

import copy
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import levern_platform  # noqa: E402
import levern_policy  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
from test_qwen_prefix_image import boot_api_server  # noqa: E402

PROFILES = HERE / 'qwen_c2_profiles.json'
TIMED_BASE, AUDIT_BASE = 'c2-packed-tp4-8x262k-best-time-gate', 'c2-packed-tp4-8x262k-best-audit'
TIMED, R1, AUDIT, CONTROL = ('c2-packed-tp4-8x262k-best-levern-time-gate', 'c2-packed-tp4-8x262k-best-levern-r1-time-gate',
                             'c2-packed-tp4-8x262k-best-levern-audit', 'c2-packed-tp4-8x262k-best-levern-control-audit')
HOLD, FOREIGN = 'c2-packed-tp4-8x262k-best-levern-final-hold-time-gate', 'c2-packed-tp4-8x262k-best-levern-foreign-time-gate'
HANG = 'c2-packed-tp4-8x262k-best-levern-hang-gate'
NEW = (AUDIT, CONTROL, HOLD, FOREIGN, HANG, R1, TIMED)
BASE = {TIMED: TIMED_BASE, R1: TIMED_BASE, AUDIT: AUDIT_BASE, CONTROL: AUDIT_BASE, HOLD: TIMED_BASE, FOREIGN: TIMED_BASE, HANG: TIMED_BASE}
STEPS = {'QWEN_FAST_LEVER_N': '1', 'QWEN_FAST_LEVERN_STEP_TOKENS': '2048', 'QWEN_FAST_LEVERN_SOLO_STEP_TOKENS': '16384'}
EXTRA_ENV = {TIMED: dict(STEPS, QWEN_FAST_LEVERN_PREFILL_SHARE='0.5'),
             R1: dict(STEPS, QWEN_FAST_LEVERN_ROUNDS='1'),
             AUDIT: dict(STEPS, QWEN_FAST_LEVERN_PREFILL_SHARE='0.5', QWEN_FAST_LEVERN_AUDIT='1'),
             CONTROL: {'QWEN_FAST_LEVERN_AUDIT': '1'},
             HOLD: dict(STEPS, QWEN_FAST_LEVERN_PREFILL_SHARE='0.5', QWEN_FAST_LEVERN_FAULT='final-hold'),
             FOREIGN: dict(STEPS, QWEN_FAST_LEVERN_PREFILL_SHARE='0.5', QWEN_FAST_LEVERN_FAULT='foreign'),
             HANG: dict(STEPS, QWEN_FAST_LEVERN_PREFILL_SHARE='0.5', QWEN_FAST_STALL_DEADLINE_S='120', QWEN_FAST_CCL_HANDLE_GUARD='log')}


def profiles():
    return json.loads(PROFILES.read_text(encoding='utf-8'))['profiles']


def flat(profile):
    out = copy.deepcopy(profile)
    out.pop('description')
    return out


def expected(name):
    out = flat(profiles()[BASE[name]])
    out['env'].update(EXTRA_ENV[name])
    if name != CONTROL:
        engine = {}
        for key, value in out['engine'].items():
            if key == 'no-enable-chunked-prefill':
                engine['enable-chunked-prefill'] = True
            else:
                engine[key] = value
        out['engine'] = engine
    return out


class ProfileTests(unittest.TestCase):
    def test_each_profile_is_its_twin_plus_exactly_its_flags(self):
        found = profiles()
        for name in NEW:
            with self.subTest(name=name):
                self.assertEqual(flat(found[name]), expected(name))

    def test_the_six_profiles_are_gate_only_unqualified_and_name_their_twin(self):
        found = profiles()
        for name in NEW:
            with self.subTest(name=name):
                profile = found[name]
                self.assertIs(profile['gate_only'], True)
                self.assertEqual(profile['env']['QWEN_FAST_262K_EVIDENCE_WAIVER'], '1')
                self.assertIn('UNQUALIFIED', profile['description'])
                self.assertIn(BASE[name] if name not in (HOLD, FOREIGN, HANG) else TIMED, profile['description'])
        self.assertEqual(json.loads(PROFILES.read_text(encoding='utf-8'))['default'], 'c2-packed-tp4')

    def test_the_chunked_arms_keep_the_budget_equal_to_the_window_so_only_the_cap_splits(self):
        found = profiles()
        for name in (TIMED, R1, AUDIT, HOLD, FOREIGN, HANG):
            engine = found[name]['engine']
            self.assertIs(engine['enable-chunked-prefill'], True)
            self.assertNotIn('no-enable-chunked-prefill', engine)
            self.assertEqual(engine['max-num-batched-tokens'], engine['max-model-len'])
            self.assertIs(engine['no-async-scheduling'], True)
            self.assertEqual(engine['num-gpu-blocks-override'], 19968)
        control = found[CONTROL]['engine']
        self.assertIs(control['no-enable-chunked-prefill'], True)
        self.assertNotIn('enable-chunked-prefill', control)

    def test_the_control_is_not_armed_and_the_interleaved_arms_are(self):
        found = profiles()
        self.assertNotIn('QWEN_FAST_LEVER_N', found[CONTROL]['env'])
        self.assertEqual(found[CONTROL]['env']['QWEN_FAST_LEVERN_AUDIT'], '1')
        for name in (TIMED, R1, AUDIT, HOLD, FOREIGN, HANG):
            self.assertEqual(found[name]['env']['QWEN_FAST_LEVER_N'], '1')
            self.assertEqual(found[name]['env']['QWEN_FAST_KV_RESERVATION'], '1')
        self.assertNotIn('QWEN_FAST_LEVERN_AUDIT', found[TIMED]['env'], 'a timed arm carries no audit')
        self.assertNotIn('QWEN_FAST_LEVERN_AUDIT', found[R1]['env'])
        for name in (TIMED, R1, AUDIT, CONTROL):
            self.assertNotIn('QWEN_FAST_LEVERN_FAULT', found[name]['env'], 'only the two negative controls carry a fault')
        self.assertEqual(found[HOLD]['env']['QWEN_FAST_LEVERN_FAULT'], 'final-hold')
        self.assertEqual(found[FOREIGN]['env']['QWEN_FAST_LEVERN_FAULT'], 'foreign')
        self.assertEqual(found[AUDIT]['env']['QWEN_FAST_LEVERN_AUDIT'], '1')

    def test_no_other_profile_names_a_lever_n_flag(self):
        for name, profile in profiles().items():
            # the two merged profiles (Lever N beside prefix reuse) are held by test_levern_prefix_contract
            if name in NEW or name.startswith('c2-packed-tp4-8x262k-ship-prefix-levern'):
                continue
            with self.subTest(name=name):
                self.assertEqual([key for key in profile['env'] if key in levern_policy.ALL_FLAGS], [])
                self.assertFalse(contract.levern_on(profile))
                self.assertEqual(contract.levern_problems(profile), [])

    def test_every_new_profile_passes_the_contract(self):
        for name in NEW:
            with self.subTest(name=name):
                self.assertEqual(contract.levern_problems(profiles()[name]), [])

    def test_the_flags_parse_to_what_the_description_says(self):
        found = profiles()
        for name in (TIMED, R1, AUDIT):
            cfg = levern_policy.config(found[name]['env'])
            self.assertEqual((cfg.step, cfg.solo), (2048, 16384))
        self.assertEqual(levern_policy.config(found[TIMED]['env']).share, 0.5)
        self.assertEqual(levern_policy.config(found[R1]['env']).rounds, 1)

    def test_the_argv_of_the_chunked_arms_enables_chunked_prefill(self):
        for name in (TIMED, AUDIT, HOLD, FOREIGN, HANG):
            args = contract.engine_arguments(dict(profiles()[name], name=name), '/snapshot')
            self.assertIn('--enable-chunked-prefill', args)
            self.assertNotIn('--no-enable-chunked-prefill', args)
        args = contract.engine_arguments(dict(profiles()[CONTROL], name=CONTROL), '/snapshot')
        self.assertIn('--no-enable-chunked-prefill', args)
        self.assertNotIn('--enable-chunked-prefill', args)


class PinTests(unittest.TestCase):
    def test_the_contract_names_the_policys_flags(self):
        self.assertEqual(contract.LEVERN_SWITCH, levern_policy.FLAG)
        self.assertEqual(contract.LEVERN_AUDIT, levern_policy.AUDIT_FLAG)
        self.assertEqual(sorted(contract.LEVERN_ENV_FLAGS), sorted(levern_policy.ALL_FLAGS))

    def test_the_scheduler_side_reads_the_same_switch(self):
        import serving_prefill_admission as admission

        self.assertEqual(admission.LEVERN_FLAG, levern_policy.FLAG)

    def test_the_platform_hook_names_the_plugins_platform_module(self):
        self.assertEqual(contract.LEVERN_PLATFORM_MODULE, levern_platform.MODULE)


class RefusalTests(unittest.TestCase):
    def mutated(self, name=TIMED, env=None, engine=None, drop_env=(), drop_engine=(), **top):
        profile = copy.deepcopy(profiles()[name])
        profile['env'].update(env or {})
        for key in drop_env:
            profile['env'].pop(key, None)
        profile['engine'].update(engine or {})
        for key in drop_engine:
            profile['engine'].pop(key, None)
        profile.update(top)
        return profile

    def refused(self, profile, needle):
        problems = contract.levern_problems(profile)
        self.assertTrue(any(needle in problem for problem in problems), (needle, problems))

    def test_a_traffic_profile_is_refused(self):
        self.refused(self.mutated(gate_only=False), 'gate-only')
        profile = self.mutated()
        profile.pop('gate_only')
        self.refused(profile, 'gate-only')

    def test_the_audit_alone_needs_a_gate_only_profile(self):
        self.refused(self.mutated(CONTROL, gate_only=False), 'gate instrument')

    def test_the_kv_read_knobs_are_gate_instruments_beside_the_audit(self):
        self.refused(self.mutated(CONTROL, env={'QWEN_FAST_LEVERN_KV_READ': 'region'}, gate_only=False), 'QWEN_FAST_LEVERN_KV_READ')
        self.assertEqual(contract.levern_problems(self.mutated(CONTROL, env={'QWEN_FAST_LEVERN_KV_READ': 'region'})), [])
        self.assertEqual(contract.levern_problems(self.mutated(AUDIT, env={'QWEN_FAST_LEVERN_KV_READ': 'cross', 'QWEN_FAST_LEVERN_KV_CROSS_STEPS': '2'})), [])
        # nothing to read without the audit; a bad value; a traffic profile with the lever on and the knob
        self.refused(self.mutated(TIMED, env={'QWEN_FAST_LEVERN_KV_READ': 'region'}), 'without QWEN_FAST_LEVERN_AUDIT=1')
        self.refused(self.mutated(AUDIT, env={'QWEN_FAST_LEVERN_KV_READ': 'auto'}), 'QWEN_FAST_LEVERN_KV_READ must be one of')
        self.refused(self.mutated(AUDIT, env={'QWEN_FAST_LEVERN_KV_READ': 'region'}, gate_only=False), 'gate instrument')

    def test_each_requirement_is_checked(self):
        cases = (
            ('QWEN_FAST_ANY_REQUEST', dict(env={'QWEN_FAST_ANY_REQUEST': '0'}), 'QWEN_FAST_ANY_REQUEST=1'),
            ('four-card', dict(env={'QWEN_FAST_TP': '2'}), 'four-card'),
            ('chunked off', dict(engine={'no-enable-chunked-prefill': True}), 'enable-chunked-prefill'),
            ('chunked missing', dict(drop_engine=('enable-chunked-prefill',)), 'enable-chunked-prefill'),
            ('async', dict(engine={'no-async-scheduling': False}), 'no-async-scheduling'),
            ('budget below window', dict(engine={'max-num-batched-tokens': 131072}), 'must equal max-model-len'),
            ('budget above window', dict(engine={'max-num-batched-tokens': 300000}), 'must equal max-model-len'),
            ('block size', dict(engine={'block-size': 128}), 'block-size must be 64'),
            ('multimodal', dict(engine={'limit-mm-per-prompt': {'image': 1, 'video': 0}}), 'text only'),
            ('no reservation', dict(env={'QWEN_FAST_KV_RESERVATION': '0'}), 'KV_RESERVATION'),
            ('credit', dict(env={'QWEN_FAST_DECODE_STEPS_PER_ADMISSION': '2'}), 'DECODE_STEPS_PER_ADMISSION'),
            ('lane', dict(env={'QWEN_FAST_LANE': '1'}), 'QWEN_FAST_LANE'),
            ('prefix reuse', dict(env={'QWEN_PREFIX_REUSE': '1'}), 'QWEN_PREFIX_REUSE'),
            ('sticky', dict(env={'QWEN_FAST_STICKY_SESSIONS': '1'}), 'QWEN_FAST_STICKY_SESSIONS'),
            ('step off the chunk', dict(env={'QWEN_FAST_LEVERN_STEP_TOKENS': '3000'}), 'multiple of 2048'),
            ('share', dict(env={'QWEN_FAST_LEVERN_PREFILL_SHARE': '1.5'}), 'PREFILL_SHARE'),
            ('fault', dict(env={'QWEN_FAST_LEVERN_FAULT': 'nope'}), 'FAULT'),
        )
        for label, kwargs, needle in cases:
            with self.subTest(label):
                self.refused(self.mutated(**kwargs), needle)

    def test_the_fast_path_is_required(self):
        profile = self.mutated()
        profile['engine']['additional-config'] = dict(profile['engine']['additional-config'], qwen_fast_t16=False)
        self.refused(profile, 'qwen_fast_t16')

    def test_a_sibling_without_the_master_switch_is_a_typo(self):
        for name in (CONTROL, 'c2-packed-tp4-8x262k-best-time-gate'):
            with self.subTest(name=name):
                self.refused(self.mutated(name, env={'QWEN_FAST_LEVERN_PREFILL_SHARE': '0.5'}), 'QWEN_FAST_LEVER_N=1')

    def test_the_credit_set_to_zero_is_not_a_conflict(self):
        profile = self.mutated(env={'QWEN_FAST_DECODE_STEPS_PER_ADMISSION': '0', 'QWEN_FAST_LANE': '0'})
        self.assertEqual(contract.levern_problems(profile), [])


class ApplyEnvironmentTests(unittest.TestCase):
    def test_an_inherited_flag_is_dropped_under_a_profile_that_does_not_name_it(self):
        inherited = {name: '1' for name in contract.LEVERN_ENV_FLAGS}
        environ = contract.apply_environment(dict(profiles()[TIMED_BASE], name=TIMED_BASE), dict(inherited))
        for name in contract.LEVERN_ENV_FLAGS:
            self.assertNotIn(name, environ)

    def test_a_flag_the_profile_names_is_set_and_the_others_dropped(self):
        inherited = {name: '1' for name in contract.LEVERN_ENV_FLAGS}
        environ = contract.apply_environment(dict(profiles()[R1], name=R1), dict(inherited))
        self.assertEqual(environ['QWEN_FAST_LEVER_N'], '1')
        self.assertEqual(environ['QWEN_FAST_LEVERN_ROUNDS'], '1')
        self.assertNotIn('QWEN_FAST_LEVERN_PREFILL_SHARE', environ)
        self.assertNotIn('QWEN_FAST_LEVERN_AUDIT', environ)
        self.assertNotIn('QWEN_FAST_LEVERN_FAULT', environ)

    def test_a_profile_without_the_flags_sets_none_and_leaves_the_rest_of_the_environment(self):
        environ = contract.apply_environment(dict(profiles()[TIMED_BASE], name=TIMED_BASE), {'KEEP': 'me'})
        self.assertEqual(environ['KEEP'], 'me')
        self.assertFalse([name for name in environ if name in levern_policy.ALL_FLAGS])


class BootTests(unittest.TestCase):
    GATE = {contract.GATE_SWITCH: '1'}

    def hooks(self, name):
        profile, environ, logged, hooks, launched = boot_api_server(name, self.GATE)
        self.assertEqual(profile['name'], name, profile)
        return sorted(hook.name for hook in hooks), environ, logged, launched

    def test_a_lever_n_profile_arms_the_platform_hook_in_the_server_process(self):
        for name in (TIMED, R1, AUDIT):
            with self.subTest(name=name):
                hooks, environ, logged, launched = self.hooks(name)
                self.assertIn(contract.LEVERN_PLATFORM_MODULE, hooks)
                self.assertTrue(any('Lever N armed' in line for line in logged), logged)
                self.assertIn('--enable-chunked-prefill', launched)
                self.assertEqual(environ['QWEN_FAST_LEVER_N'], '1')

    def test_the_control_and_the_twins_arm_nothing(self):
        for name in (CONTROL, TIMED_BASE, AUDIT_BASE):
            with self.subTest(name=name):
                hooks, environ, logged, launched = self.hooks(name)
                self.assertNotIn(contract.LEVERN_PLATFORM_MODULE, hooks)
                self.assertFalse(any('Lever N' in line for line in logged))
                self.assertNotIn('--enable-chunked-prefill', launched)
                self.assertNotIn('QWEN_FAST_LEVER_N', environ)

    def test_the_audit_switch_reaches_the_environment_of_both_audit_arms(self):
        for name in (AUDIT, CONTROL):
            self.assertEqual(self.hooks(name)[1]['QWEN_FAST_LEVERN_AUDIT'], '1')

    def test_a_profile_that_breaks_a_lever_n_rule_does_not_boot(self):
        import tempfile

        broken = json.loads(PROFILES.read_text(encoding='utf-8'))
        broken['profiles'][TIMED]['engine']['max-num-batched-tokens'] = 131072
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profiles.json'
            path.write_text(json.dumps(broken), encoding='utf-8')
            with mock.patch.object(sys.modules['test_qwen_prefix_image'], 'PROFILES', path):
                refused, environ, _, hooks, _ = boot_api_server(TIMED, self.GATE)
        self.assertIsInstance(refused, ValueError)
        self.assertIn('cannot serve Lever N exactly', str(refused))
        self.assertIn('must equal max-model-len', str(refused))
        self.assertNotIn('QWEN_FAST_LEVER_N', environ, 'refused before the environment is applied')
        self.assertEqual(hooks, [])

    def test_a_gate_profile_boots_only_in_a_gate(self):
        refused, *_ = boot_api_server(TIMED)
        self.assertIsInstance(refused, ValueError)
        self.assertIn('is gate only', str(refused))


class HookTests(unittest.TestCase):
    def test_the_platform_hook_installs_the_wrap_on_the_platform_module(self):
        calls = []
        contract.install_levern_platform(on_import=lambda name, callback: calls.append((name, callback)))
        self.assertEqual(calls, [('vllm_tt_plugin.platform', levern_platform.install)])

    def test_a_platform_module_that_is_already_imported_is_wrapped_at_once(self):
        import types

        module = types.ModuleType(contract.LEVERN_PLATFORM_MODULE)
        module._apply_chunked_prefill_policy = lambda config: None
        with mock.patch.dict(sys.modules, {contract.LEVERN_PLATFORM_MODULE: module}), mock.patch.object(sys, 'meta_path', list(sys.meta_path)):
            hooks = len(sys.meta_path)
            self.assertTrue(contract.install_levern_platform())
            self.assertEqual(len(sys.meta_path), hooks, 'no post-import hook is left waiting for an import that already happened')
        self.assertTrue(getattr(module._apply_chunked_prefill_policy, levern_platform.WRAPPED, False))

    def test_a_platform_module_not_yet_imported_gets_the_post_import_hook(self):
        with mock.patch.dict(sys.modules), mock.patch.object(sys, 'meta_path', list(sys.meta_path)):
            sys.modules.pop(contract.LEVERN_PLATFORM_MODULE, None)
            hooks = len(sys.meta_path)
            contract.install_levern_platform()
            self.assertEqual(len(sys.meta_path), hooks + 1)
            self.assertEqual(sys.meta_path[0].name, contract.LEVERN_PLATFORM_MODULE)


if __name__ == '__main__':
    unittest.main()
