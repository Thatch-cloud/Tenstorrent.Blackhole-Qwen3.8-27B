import json
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import serving_c2_contract as contract

PROFILES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qwen_c2_profiles.json')


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


def enforce(params, prompt_tokens=1000, max_model_len=131328, budget=4096, max_prompt_tokens=61440):
    return contract.enforce_request(params, prompt_tokens=prompt_tokens, max_model_len=max_model_len,
                                    budget=budget, eos_ids=frozenset((248046, 248044)),
                                    max_prompt_tokens=max_prompt_tokens)


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
        for name in ('exact', 'coding'):
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
            longest = min(capacity - budget, profile.get('max_prompt_tokens') or capacity) + budget
            self.assertGreaterEqual(engine['num-gpu-blocks-override'],
                                    engine['max-num-seqs'] * -(-longest // 64), name)

    def test_default_profile_is_general(self):
        environ = dict(os.environ)
        os.environ.pop('QWEN_C2_PROFILE', None)
        try:
            self.assertEqual(contract.load_profile(PROFILES)['name'], 'general')
        finally:
            os.environ.clear()
            os.environ.update(environ)


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


class GeneralVariantTest(unittest.TestCase):
    def test_p300_keeps_the_agents_mesh_and_additional_config(self):
        profile = contract.load_profile(PROFILES, 'general-p300')
        environ = contract.apply_environment(profile, {'TT_MESH_GRAPH_DESC_PATH': '/p300.textproto'})
        self.assertEqual(environ['TT_MESH_GRAPH_DESC_PATH'], '/p300.textproto')
        argv = contract.rewrite_argv(['-m', '--additional-config', '{"tt": {"fabric_config": "FABRIC_1D"}}',
                                      '--max-model-len', '1'], profile, '/snap')
        self.assertEqual(argv.count('--additional-config'), 1)
        self.assertIn('FABRIC_1D', ' '.join(argv))
        self.assertEqual(argv[argv.index('--max-model-len') + 1], '65536')

    def test_lean_unsets_fast_path_flags_and_keeps_the_k_stack(self):
        profile = contract.load_profile(PROFILES, 'general-lean')
        environ = contract.apply_environment(profile, {'QWEN_FAST_PACKED_STEP': '1', 'QWEN_FAST_OUTPUT_BUDGET': '256',
                                                       'QWEN_MLP_BLOCK_STREAM_EXPERIMENT': '1', 'QWEN_GDN_FUSED_DECODE': '1',
                                                       'QWEN_SDPA_BF8': '1'})
        self.assertEqual(sorted(environ), ['QWEN_GDN_FUSED_DECODE', 'QWEN_SDPA_BF8', 'TT_MESH_GRAPH_DESC_PATH'])


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


if __name__ == '__main__':
    unittest.main()
