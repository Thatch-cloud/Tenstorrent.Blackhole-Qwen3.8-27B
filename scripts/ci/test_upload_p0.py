"""The engine-start upload levers' wiring (tp4/upload-p0): the contract refuses a misuse by name and arms the post-import hooks only when a profile asks, an inherited
switch never reaches a profile that does not name it, the smoke check reads the levers' lines, the job pack parses, and the names every file repeats are the modules'."""

import glob
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_smoke_check as smoke  # noqa: E402
import qwen_device_zeros as zeros  # noqa: E402
import qwen_lazy_shard as lazy  # noqa: E402
import serving_c2_contract as contract  # noqa: E402

PACK = HERE / 'references' / 'upload-p0-jobs'
PARENT = 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic'


def profile(env=None, gate_only=True, name='p'):
    return dict(name=name, env=dict(env or {}), gate_only=gate_only)


class NamesTests(unittest.TestCase):
    def test_the_contract_and_the_smoke_check_repeat_the_modules_names(self):
        self.assertEqual(set(contract.UPLOAD_NAMES), set(zeros.NAMES) | set(lazy.NAMES))
        self.assertEqual(len(contract.UPLOAD_NAMES), len(set(contract.UPLOAD_NAMES)))
        self.assertEqual(set(contract.UPLOAD_GATE_ONLY), {zeros.AUDIT_FLAG, lazy.AUDIT_FLAG})
        self.assertEqual((smoke.DEVICE_ZEROS_FLAG, smoke.DEVICE_ZEROS_AUDIT_FLAG), (zeros.FLAG, zeros.AUDIT_FLAG))
        self.assertEqual((smoke.DEVICE_ZEROS_ENGAGED, smoke.DEVICE_ZEROS_REFUSED, smoke.DEVICE_ZEROS_AUDIT, smoke.DEVICE_ZEROS_MISMATCH),
                         (zeros.ENGAGED, zeros.REFUSED, zeros.AUDIT_LINE, zeros.AUDIT_MISMATCH))
        self.assertEqual((smoke.LAZY_SHARD_FLAG, smoke.LAZY_SHARD_AUDIT_FLAG), (lazy.FLAG, lazy.AUDIT_FLAG))
        self.assertEqual((smoke.LAZY_SHARD_ENGAGED, smoke.LAZY_SHARD_REFUSED, smoke.LAZY_SHARD_AUDIT, smoke.LAZY_SHARD_MISMATCH),
                         (lazy.ENGAGED, lazy.REFUSED, lazy.AUDIT_LINE, lazy.AUDIT_MISMATCH))

    def test_the_modules_tags_are_the_ones_the_smoke_check_requires(self):
        self.assertEqual(set(smoke.DEVICE_ZEROS_TAGS), {'kv_cache', 'buffer_pool'})


class ContractTests(unittest.TestCase):
    def test_a_profile_without_the_names_costs_nothing(self):
        self.assertEqual(contract.upload_problems(profile({'QWEN_FAST_TP': '4'}, gate_only=False)), [])
        self.assertEqual(contract.install_upload_levers({}, on_import=lambda name, callback: self.fail('hooked')), [])
        self.assertEqual(contract.install_upload_levers({'QWEN_FAST_DEVICE_ZEROS': '0', 'QWEN_FAST_LAZY_SHARD_W': '0'},
                                                         on_import=lambda name, callback: self.fail('hooked')), [])

    def test_misuse_is_named(self):
        cases = (({'QWEN_FAST_DEVICE_ZEROS': 'true'}, True, 'must be 0 or 1'),
                 ({'QWEN_FAST_DEVICE_ZEROS_AUDIT': '1'}, True, 'without QWEN_FAST_DEVICE_ZEROS=1'),
                 ({'QWEN_FAST_LAZY_SHARD_W_AUDIT': '1'}, True, 'without QWEN_FAST_LAZY_SHARD_W=1'),
                 ({'QWEN_FAST_DEVICE_ZEROS': '1', 'QWEN_FAST_DEVICE_ZEROS_AUDIT': '1'}, False, 'is a gate profile'),
                 ({'QWEN_FAST_LAZY_SHARD_W': '1', 'QWEN_FAST_LAZY_SHARD_W_AUDIT': '1'}, False, 'is a gate profile'),
                 ({'QWEN_FAST_DEVICE_ZEROS_TYPO': '1'}, True, 'is not an upload lever name'),
                 ({'QWEN_FAST_LAZY_SHARD_WW': '1'}, True, 'is not an upload lever name'),
                 ({'QWEN_FAST_DEVICE_ZEROS': '1', 'QWEN_FAST_DEVICE_ZEROS_MIN_BYTES': 'lots'}, True, 'MIN_BYTES must be an integer'))
        for env, gate_only, needle in cases:
            with self.subTest(env=env, gate_only=gate_only):
                problems = contract.upload_problems(profile(env, gate_only=gate_only))
                self.assertTrue(any(needle in problem for problem in problems), problems)

    def test_the_levers_themselves_may_be_a_traffic_profiles_and_the_audits_may_not(self):
        self.assertEqual(contract.upload_problems(profile({'QWEN_FAST_DEVICE_ZEROS': '1', 'QWEN_FAST_LAZY_SHARD_W': '1'}, gate_only=False)), [])
        self.assertEqual(contract.upload_problems(profile({'QWEN_FAST_DEVICE_ZEROS': '1', 'QWEN_FAST_DEVICE_ZEROS_AUDIT': '1'}, gate_only=True)), [])

    def test_the_hooks_are_armed_for_the_modules_a_switch_names(self):
        seen = []
        armed = contract.install_upload_levers({'QWEN_FAST_DEVICE_ZEROS': '1'}, on_import=lambda name, callback: seen.append(name))
        self.assertEqual((armed, seen), ([zeros.FLAG], [zeros.MODEL_MODULE]))
        seen = []
        armed = contract.install_upload_levers({'QWEN_FAST_LAZY_SHARD_W': '1'}, on_import=lambda name, callback: seen.append(name))
        self.assertEqual((armed, seen), ([lazy.FLAG], [lazy.TP_COMMON]))
        seen = []
        armed = contract.install_upload_levers({'QWEN_FAST_DEVICE_ZEROS': '1', 'QWEN_FAST_LAZY_SHARD_W': '1'}, on_import=lambda name, callback: seen.append(name))
        self.assertEqual((armed, sorted(seen)), ([zeros.FLAG, lazy.FLAG], sorted([zeros.MODEL_MODULE, lazy.TP_COMMON])))

    def test_an_inherited_switch_never_reaches_a_profile_that_does_not_name_it(self):
        inherited = {name: '1' for name in contract.UPLOAD_NAMES}
        inherited['KEEP'] = 'me'
        base = dict(name='p', mesh_graph_descriptor='/x', env={'QWEN_FAST_TP': '4'})
        environ = contract.apply_environment(base, dict(inherited))
        self.assertEqual(environ['KEEP'], 'me')
        self.assertFalse([name for name in contract.UPLOAD_NAMES if name in environ])
        named = dict(base, env={'QWEN_FAST_TP': '4', 'QWEN_FAST_DEVICE_ZEROS': '1'})
        environ = contract.apply_environment(named, dict(inherited))
        self.assertEqual(environ['QWEN_FAST_DEVICE_ZEROS'], '1')
        self.assertFalse([name for name in contract.UPLOAD_NAMES if name != 'QWEN_FAST_DEVICE_ZEROS' and name in environ])

    def boot(self, env, extra_environ=None):
        """contract.boot under a copy of the general-tp4 profile that also names `env`, in the API server, the process-wide effects patched out. -> (hooks, log lines)."""
        import copy
        import json
        import tempfile

        with open(str(HERE / 'qwen_c2_profiles.json'), encoding='utf-8') as handle:
            data = json.load(handle)
        entry = copy.deepcopy(data['profiles']['general-tp4'])
        entry['env'].update(env)
        data['profiles']['upload-test'] = entry
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'profiles.json')
            with open(path, 'w', encoding='utf-8') as handle:
                json.dump(data, handle)
            environ = dict(extra_environ or {}, QWEN_C2_SERVING='1', QWEN_C2_PROFILE='upload-test', QWEN_C2_PROFILES=path, MESH_DEVICE='P300')
            before, saved_argv, lines = list(sys.meta_path), list(sys.argv), []
            sys.argv[:] = ['api_server', '--model', 'Qwen/Qwen3.8-27B', '--port', '8000', '--max-model-len', '65536', '--max-num-seqs', '2',
                           '--additional-config', '{"tt": {"fabric_config": "FABRIC_1D"}}']
            try:
                with mock.patch.dict(os.environ, {'QWEN_C2_PROFILE': 'upload-test'}), mock.patch.object(contract, 'fix_sys_path'), \
                        mock.patch.object(contract, 'install_teardown_skip'), mock.patch.object(contract, 'resolve_snapshot', return_value='/snap'), \
                        mock.patch.object(contract, 'install_prefix_metrics'), mock.patch.object(contract, 'read_salt_key', return_value=(None, 'none')), \
                        mock.patch.object(contract.os.path, 'exists', return_value=False), \
                        mock.patch.object(contract, 'log', side_effect=lambda message, *values: lines.append(message % values if values else message)):
                    contract.boot(environ=environ, orig_argv=['python3', '-m', contract.API_SERVER])
                hooks = [hook for hook in sys.meta_path if hook not in before and isinstance(hook, contract.PostImportHook)]
            finally:
                sys.argv[:] = saved_argv
                sys.meta_path[:] = before
        return hooks, lines

    def test_boot_arms_the_hooks_a_profile_asks_for_and_only_those(self):
        hooks, lines = self.boot({})
        self.assertFalse([hook.name for hook in hooks if hook.name in (zeros.MODEL_MODULE, lazy.TP_COMMON)])
        self.assertFalse([line for line in lines if 'upload levers' in line])
        hooks, lines = self.boot({'QWEN_FAST_DEVICE_ZEROS': '1', 'QWEN_FAST_LAZY_SHARD_W': '1'})
        self.assertEqual(sorted(hook.name for hook in hooks if hook.name in (zeros.MODEL_MODULE, lazy.TP_COMMON)), sorted([zeros.MODEL_MODULE, lazy.TP_COMMON]))
        self.assertTrue([line for line in lines if 'engine-start upload levers armed' in line])
        hooks, _ = self.boot({'QWEN_FAST_LAZY_SHARD_W': '1'})
        self.assertEqual([hook.name for hook in hooks if hook.name in (zeros.MODEL_MODULE, lazy.TP_COMMON)], [lazy.TP_COMMON])

    def test_boot_refuses_a_misused_profile_before_anything_is_hooked(self):
        for env, needle in (({'QWEN_FAST_DEVICE_ZEROS': '3'}, 'must be 0 or 1'), ({'QWEN_FAST_DEVICE_ZEROS_AUDIT': '1'}, 'without QWEN_FAST_DEVICE_ZEROS=1')):
            with self.subTest(env=env):
                with self.assertRaises(ValueError) as caught:
                    self.boot(env)
                self.assertIn('engine-start upload levers', str(caught.exception))
                self.assertIn(needle, str(caught.exception))


class SmokeRuleTests(unittest.TestCase):
    ZEROS_ON = {'QWEN_FAST_DEVICE_ZEROS': '1'}
    ZEROS_AUDIT = {'QWEN_FAST_DEVICE_ZEROS': '1', 'QWEN_FAST_DEVICE_ZEROS_AUDIT': '1'}
    LAZY_ON = {'QWEN_FAST_LAZY_SHARD_W': '1'}
    LAZY_AUDIT = {'QWEN_FAST_LAZY_SHARD_W': '1', 'QWEN_FAST_LAZY_SHARD_W_AUDIT': '1'}

    @staticmethod
    def text(*lines):
        return '\n'.join(lines)

    ENGAGED_KV = '[PINDIAG] tp4 device zeros engaged tag=kv_cache shape=(19968, 1, 64, 256) dtype=bf8 bytes_per_card=347602944 audit=on'
    ENGAGED_POOL = '[PINDIAG] tp4 device zeros engaged tag=buffer_pool shape=(1, 1, 2048, 5120) dtype=bf16 bytes_per_card=20971520 audit=on'
    PASS_KV = '[PINDIAG] tp4 device zeros audit exact=True tag=kv_cache tensor=1 shape=(19968, 1, 64, 256) dtype=bf8 windows=8 sampled_bytes_per_chip=139264 chips=4 raw=numpy'
    PASS_POOL = '[PINDIAG] tp4 device zeros audit exact=True tag=buffer_pool tensor=1 shape=(1, 1, 2048, 5120) dtype=bf16 windows=1 sampled_bytes_per_chip=20971520 chips=4 raw=numpy'
    MISMATCH = '[PINDIAG] tp4 device zeros audit mismatch tag=kv_cache tensor=1 exact=False dtype=bf8 differing_bytes=384 nonzero_bytes=384 exponent_only=True first=(0, None, 3)'
    LAZY_ENGAGED = '[PINDIAG] tp4 lazy shard engaged audit=on audit_loads=2'
    LAZY_PASS = '[PINDIAG] tp4 lazy shard audit exact=True name=mlp.gate_proj.weight.tp chips=4 bytes_per_chip=1 raw=numpy'

    def test_a_profile_with_the_levers_off_and_no_lines_is_clean(self):
        self.assertEqual(smoke.upload_p0_problems({}, 'nothing here'), [])
        self.assertEqual(smoke.upload_p0_problems(None, ''), [])

    def test_an_engaged_line_on_a_profile_without_the_switch_is_a_problem(self):
        self.assertTrue(any('ran on a profile without it' in problem for problem in smoke.upload_p0_problems({}, self.text(self.ENGAGED_KV))))
        self.assertTrue(any('ran on a profile without it' in problem for problem in smoke.upload_p0_problems({}, self.text(self.LAZY_ENGAGED))))

    def test_the_unaudited_lever_needs_both_engaged_lines(self):
        self.assertEqual(smoke.upload_p0_problems(self.ZEROS_ON, self.text(self.ENGAGED_KV, self.ENGAGED_POOL)), [])
        problems = smoke.upload_p0_problems(self.ZEROS_ON, self.text(self.ENGAGED_KV))
        self.assertEqual(len(problems), 1)
        self.assertIn('buffer_pool', problems[0])
        self.assertEqual(len(smoke.upload_p0_problems(self.ZEROS_ON, '')), 2)

    def test_the_audited_lever_needs_a_passing_audit_line_per_tag(self):
        good = self.text(self.ENGAGED_KV, self.ENGAGED_POOL, self.PASS_KV, self.PASS_POOL)
        self.assertEqual(smoke.upload_p0_problems(self.ZEROS_AUDIT, good), [])
        problems = smoke.upload_p0_problems(self.ZEROS_AUDIT, self.text(self.ENGAGED_KV, self.ENGAGED_POOL, self.PASS_KV))
        self.assertEqual(len(problems), 1)
        self.assertIn('tag=buffer_pool', problems[0])
        self.assertTrue(smoke.upload_p0_problems(self.ZEROS_AUDIT, self.text(self.ENGAGED_KV, self.ENGAGED_POOL)))

    def test_audit_lines_on_a_profile_without_the_audit_flag_are_a_problem(self):
        problems = smoke.upload_p0_problems(self.ZEROS_ON, self.text(self.ENGAGED_KV, self.ENGAGED_POOL, self.PASS_KV))
        self.assertEqual(len(problems), 1)
        self.assertIn('the log holds audit lines', problems[0])

    def test_a_mismatch_or_a_refusal_fails_the_arm_whatever_else_passed(self):
        text = self.text(self.ENGAGED_KV, self.ENGAGED_POOL, self.PASS_KV, self.PASS_POOL, self.MISMATCH)
        problems = smoke.upload_p0_problems(self.ZEROS_AUDIT, text)
        self.assertEqual(len(problems), 1)
        self.assertIn('latched off or was refused', problems[0])
        refused = self.text(self.ENGAGED_KV, self.ENGAGED_POOL, '[PINDIAG] tp4 device zeros refused reason=the device fill raised RuntimeError')
        self.assertTrue(smoke.upload_p0_problems(self.ZEROS_ON, refused))
        self.assertTrue(smoke.upload_p0_problems({}, '[PINDIAG] tp4 lazy shard refused reason=x'))

    def test_the_lazy_lever_needs_its_engaged_line_and_its_audit(self):
        self.assertEqual(smoke.upload_p0_problems(self.LAZY_ON, self.text(self.LAZY_ENGAGED)), [])
        self.assertEqual(smoke.upload_p0_problems(self.LAZY_AUDIT, self.text(self.LAZY_ENGAGED, self.LAZY_PASS)), [])
        self.assertEqual(len(smoke.upload_p0_problems(self.LAZY_AUDIT, self.text(self.LAZY_ENGAGED))), 1)
        self.assertEqual(len(smoke.upload_p0_problems(self.LAZY_ON, '')), 1)
        mismatch = '[PINDIAG] tp4 lazy shard audit mismatch name=x exact=False differing_bytes=3 chips=4'
        self.assertTrue(smoke.upload_p0_problems(self.LAZY_AUDIT, self.text(self.LAZY_ENGAGED, self.LAZY_PASS, mismatch)))

    def test_both_levers_are_judged_independently(self):
        env = dict(self.ZEROS_AUDIT, **self.LAZY_AUDIT)
        good = self.text(self.ENGAGED_KV, self.ENGAGED_POOL, self.PASS_KV, self.PASS_POOL, self.LAZY_ENGAGED, self.LAZY_PASS)
        self.assertEqual(smoke.upload_p0_problems(env, good), [])
        self.assertEqual(len(smoke.upload_p0_problems(env, self.text(self.ENGAGED_KV, self.ENGAGED_POOL, self.PASS_KV, self.PASS_POOL))), 2)

    def test_the_rule_runs_in_both_the_smoke_check_and_the_gate_arm_reader(self):
        source = (HERE / 'c2_smoke_check.py').read_text(encoding='utf-8')
        self.assertEqual(source.count('problems.extend(upload_p0_problems(env, container_text))'), 1)
        self.assertEqual(source.count('problems += upload_p0_problems(env, container_text)'), 1)


class PackTests(unittest.TestCase):
    def templates(self):
        return sorted(glob.glob(str(PACK / '*.env')))

    def test_every_template_passes_the_job_reader_with_rc_zero(self):
        for path in self.templates():
            with self.subTest(template=os.path.basename(path)):
                result = subprocess.run([sys.executable, '-s', str(HERE / 'c2_serving_job.py'), path, str(HERE / 'qwen_c2_profiles.json')],
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr[-600:])

    def test_order_lists_exactly_the_templates_with_one_image_tag(self):
        text = (PACK / 'ORDER.txt').read_text(encoding='utf-8')
        rows = [line.split() for line in text.splitlines() if line and not line.startswith('#')]
        self.assertEqual(sorted(row[0] for row in rows), sorted(os.path.basename(path)[:-4] for path in self.templates()))
        self.assertEqual({row[2] for row in rows}, {'tp4-upload-p0-1'})
        self.assertTrue(all(row[1] in ('stop', 'soft') and row[3].isdigit() for row in rows))
        for path in self.templates():
            self.assertIn('C2_IMAGE_TAG=tp4-upload-p0-1', Path(path).read_text(encoding='utf-8'))

    def test_the_profiles_the_templates_name_are_the_parent_and_its_upload_twins(self):
        import make_upload_profiles as generator

        allowed = {PARENT, 'c2-packed-tp4'} | set(generator.twin_names())
        for path in self.templates():
            values = dict(line.split('=', 1) for line in Path(path).read_text(encoding='utf-8').splitlines() if line and not line.startswith('#'))
            self.assertIn(values.get('C2_PROFILE'), allowed | {None}, path)
        named = {dict(line.split('=', 1) for line in Path(path).read_text(encoding='utf-8').splitlines()
                      if line and not line.startswith('#') and '=' in line).get('C2_PROFILE') for path in self.templates()}
        self.assertTrue({PARENT + '-dzero-audit', PARENT + '-lazy-audit', PARENT + '-upload-audit', PARENT + '-upload', PARENT}.issubset(named))

    def test_no_template_names_a_card_host_or_registry_and_none_touches_production(self):
        for path in self.templates() + [str(PACK / 'ORDER.txt')]:
            text = Path(path).read_text(encoding='utf-8')
            self.assertIsNone(re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text), path)          # no address
            self.assertIsNone(re.search(r'[\w-]+\.(?:local|lan|internal)\b|:\d{4,5}/', text), path)        # no host or registry
            self.assertNotIn('@', text, path)
            actions = [line for line in text.splitlines() if line.startswith('C2_ACTIONS=')]
            for line in actions:
                for forbidden in ('agentstart', 'agentstop', 'push', 'rmi', 'unserve'):
                    self.assertNotIn(forbidden, line.split('=', 1)[1].split(), (path, forbidden))

    def test_the_timed_arms_are_the_audit_free_profiles(self):
        for name in ('T0-control-a', 'T1-upload-a', 'T2-control-b', 'T3-upload-b', 'T4-dzero-only', 'T5-lazy-only'):
            text = (PACK / (name + '.env')).read_text(encoding='utf-8')
            profile_line = [line for line in text.splitlines() if line.startswith('C2_PROFILE=')][0]
            self.assertNotIn('audit', profile_line, name)
        for name in ('A1-dzero-audit-attach', 'A2-lazy-audit-attach', 'A3-upload-audit-attach'):
            text = (PACK / (name + '.env')).read_text(encoding='utf-8')
            self.assertIn('-audit', [line for line in text.splitlines() if line.startswith('C2_PROFILE=')][0], name)


if __name__ == '__main__':
    unittest.main()
