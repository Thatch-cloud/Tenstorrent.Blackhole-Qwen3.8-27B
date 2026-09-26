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
        for name in ('exact', 'coding', 'c2', 'c2-gate'):
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
        self.assertEqual(gate['env'], {key: value for key, value in c2['env'].items() if key != 'QWEN_FAST_PACKED_STEP'},
                         'the c2 code paths, the c2 ceiling')
        self.assertNotIn('QWEN_FAST_PACKED_STEP', gate['env'])
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


if __name__ == '__main__':
    unittest.main()
