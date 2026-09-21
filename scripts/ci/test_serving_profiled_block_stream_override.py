from pathlib import Path
from types import SimpleNamespace
import unittest

import profiled_block_stream_override as override


def require_hardware(environment):
    """Verbatim copy of mlp_block_stream_runtime.require_hardware's condition."""
    if (any(environment.get(name) != '1' for name in
            ('QWEN_MLP_BLOCK_STREAM_EXPERIMENT', 'QWEN_CARDS_ALLOCATED', 'QWEN_HARDWARE_TESTS'))
            or environment.get('TT_METAL_SIMULATOR') or environment.get('QWEN_SIM_ONLY') == '1'
            or environment.get('TT_METAL_DEVICE_PROFILER')):
        raise ValueError('Explicit allocated unprofiled block-stream hardware experiment required')


def full_env(**overrides):
    base = dict(QWEN_MLP_BLOCK_STREAM_EXPERIMENT='1', QWEN_CARDS_ALLOCATED='1', QWEN_HARDWARE_TESTS='1')
    base.update(overrides)
    return base


class RequestedTests(unittest.TestCase):
    def test_only_exact_one_is_requested(self):
        self.assertFalse(override.requested({}))
        self.assertFalse(override.requested({override.ENV: '0'}))
        self.assertFalse(override.requested({override.ENV: 'true'}))
        self.assertTrue(override.requested({override.ENV: '1'}))


class InstallTests(unittest.TestCase):
    def make_runtime(self):
        return SimpleNamespace(require_hardware=require_hardware)

    def test_not_requested_returns_none_and_untouched(self):
        runtime = self.make_runtime()
        log_calls = []
        result = override.install(log=lambda *a: log_calls.append(a), environ={}, runtime=runtime)
        self.assertIsNone(result)
        self.assertIs(runtime.require_hardware, require_hardware)
        self.assertEqual(log_calls, [])

    def test_requested_without_profiler_returns_none_and_logs(self):
        runtime = self.make_runtime()
        log_calls = []
        result = override.install(log=lambda *a: log_calls.append(a),
                                   environ={override.ENV: '1'}, runtime=runtime)
        self.assertIsNone(result)
        self.assertIs(runtime.require_hardware, require_hardware)
        self.assertEqual(len(log_calls), 1)
        self.assertIn('nothing to override', log_calls[0][0])
        self.assertEqual(log_calls[0][1], override.ENV)

    def test_requested_with_profiler_wraps_and_admits(self):
        runtime = self.make_runtime()
        log_calls = []
        result = override.install(log=lambda *a: log_calls.append(a),
                                   environ={override.ENV: '1', 'TT_METAL_DEVICE_PROFILER': '1'},
                                   runtime=runtime)
        self.assertEqual(result, dict(env=override.ENV, wrapped=True))
        self.assertIsNot(runtime.require_hardware, require_hardware)
        self.assertIs(runtime._unprofiled_require_hardware, require_hardware)
        self.assertEqual(len(log_calls), 1)
        self.assertIn('accepts the device profiler', log_calls[0][0])

        admitted_env = full_env(TT_METAL_DEVICE_PROFILER='1')
        runtime.require_hardware(admitted_env)  # does not raise

        missing_flag = full_env(TT_METAL_DEVICE_PROFILER='1', QWEN_HARDWARE_TESTS='0')
        with self.assertRaises(ValueError):
            runtime.require_hardware(missing_flag)

        simulator_env = full_env(TT_METAL_DEVICE_PROFILER='1', TT_METAL_SIMULATOR='1')
        with self.assertRaises(ValueError):
            runtime.require_hardware(simulator_env)

    def test_install_twice_keeps_single_wrap(self):
        runtime = self.make_runtime()
        environ = {override.ENV: '1', 'TT_METAL_DEVICE_PROFILER': '1'}
        override.install(log=lambda *a: None, environ=environ, runtime=runtime)
        wrapped_once = runtime.require_hardware
        original = runtime._unprofiled_require_hardware
        self.assertIs(original, require_hardware)

        override.install(log=lambda *a: None, environ=environ, runtime=runtime)
        self.assertIs(runtime._unprofiled_require_hardware, original)
        self.assertIs(runtime.require_hardware, wrapped_once)

        admitted_env = full_env(TT_METAL_DEVICE_PROFILER='1')
        runtime.require_hardware(admitted_env)
        missing_flag = full_env(TT_METAL_DEVICE_PROFILER='1', QWEN_CARDS_ALLOCATED='0')
        with self.assertRaises(ValueError):
            runtime.require_hardware(missing_flag)


class WiringTests(unittest.TestCase):
    def test_serving_runtime_calls_the_override_hook(self):
        source = Path(__file__).parent.joinpath('serving_runtime.py').read_text(encoding='utf-8')
        self.assertIn('from profiled_block_stream_override import install as admit_profiled_block_stream', source)
        self.assertIn('admit_profiled_block_stream(log=pindiag)', source)


if __name__ == '__main__':
    unittest.main()
