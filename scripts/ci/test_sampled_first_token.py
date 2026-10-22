"""The first token of a sampled request, through the real FastServingLifecycle (QWEN_FAST_SAMPLED, docs/tp4-sampled-verify.md section 2.2), pure CPU.

The plugin samples a request's token 0 from the prefill logits with its own unkeyed host sampler and from_prefill takes it as the session's seed. With the switch on the lifecycle
wraps the runner's host sampler: a sampled request is armed at admission, the one host-sampler call of its final prefill chunk returns the keyed draw at position len(prompt), and
the lifecycle checks after sample_tokens that the token it commits is that draw (an armed request that was not drawn fails the engine). Held here against the lifecycle's own
fixture: the draw, a greedy request and the plugin's sampler untouched, the check that fails closed, the restore at close, and (switch off) a lifecycle that never touches the sampler.
"""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / 'speculative-decoding' / 'harness'))

import sampled_accept as sa  # noqa: E402
import sampled_test_support as support  # noqa: E402
import sampled_verify as sv  # noqa: E402
from contextlib import nullcontext  # noqa: E402
from serving_lifecycle import FastServingLifecycle  # noqa: E402
import test_serving_fast_policy as policy_tests  # noqa: E402
import test_serving_request_factory as factory_tests  # noqa: E402
import test_serving_worker_hook as hook_tests  # noqa: E402


def sampled_params(base, **values):
    merged = dict(vars(base))
    merged.update(dict(temperature=0.7, top_p=0.9, top_k=8, min_p=0.0, seed=None, extra_args={sv.SEED_KEY: 4242}))
    merged.update(values)
    return SimpleNamespace(**merged)


class Plugin(object):
    """The plugin runner's host sampler and the sample_tokens that calls it, the way model_runner._get_output_tokens does for a prefill."""

    def __init__(self, lifecycle, worker, bridge, logits, token=10, calls_sampler=True):
        self.logits, self.token, self.calls_sampler, self.seen = logits, token, calls_sampler, []
        self.runner = worker.model_runner
        self.bridge = bridge

        def inner(logits=None, sampling_metadata=None):
            self.seen.append(logits.clone())
            logits.div_(0.5)                                  # vLLM's sampler may rescale in place
            return SimpleNamespace(sampled_token_ids=torch.full((1, 1), self.token, dtype=torch.int32), logprobs_tensors=None)

        self.inner = inner
        if lifecycle.first_token is not None:
            lifecycle.first_token.inner = inner
        else:
            self.runner.host_sampler = inner

        def sample_tokens(grammar_output):
            token = self.token
            if self.calls_sampler:
                output = self.runner.host_sampler(logits=self.logits.clone(), sampling_metadata=object())
                token = int(output.sampled_token_ids[0, 0])
            bridge.state.output_token_ids[:] = [token]
            return SimpleNamespace(req_ids=['request'], sampled_token_ids=[[token]])

        self.runner.sample_tokens.side_effect = sample_tokens


class Clean(unittest.TestCase):
    def setUp(self):
        sv.reset()
        self.addCleanup(sv.reset)
        patcher = mock.patch.dict(os.environ, support.clean_environment(), clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        logger = mock.patch.object(sv, 'log_line')
        self.log = logger.start()
        self.addCleanup(logger.stop)

    def lifecycle(self, switch, runner_options=None):
        """test_serving_lifecycle's own fixture (one request, a prefill step, a bridge) with the plugin runner's host sampler and sample_on_device_mode added to the runner."""
        worker, bridge, _, decode = hook_tests.WorkerHookTests().fixture()
        worker.model_runner.execute_model.return_value = None
        worker.model_runner.sample_tokens.return_value = SimpleNamespace(req_ids=['request'], sampled_token_ids=[[10]])
        worker.model_runner.host_sampler = mock.Mock(name='host_sampler')
        worker.model_runner.sample_on_device_mode = None
        for name, value in (runner_options or {}).items():
            setattr(worker.model_runner, name, value)
        _, _, _, arguments = factory_tests.RequestFactoryTests().fixture()
        new = SimpleNamespace(req_id='request', prompt_token_ids=[1] * 4096, num_computed_tokens=0, mm_features=[], prompt_embeds=None, lora_request=None,
                              sampling_params=arguments['state'].sampling_params)
        scheduled = SimpleNamespace(finished_req_ids=set(), scheduled_new_reqs=[new], scheduled_cached_reqs=SimpleNamespace(req_ids=[]), scheduled_spec_decode_tokens={},
                                    num_scheduled_tokens={'request': 4096}, total_num_scheduled_tokens=4096)
        capture = SimpleNamespace(capture=mock.Mock(side_effect=lambda: nullcontext()), close=mock.Mock())

        def factory(state, features):
            features.close()
            return bridge

        build = mock.Mock(side_effect=factory)
        with mock.patch.dict(os.environ, {sv.FLAG: '1'} if switch else {}):
            lifecycle = FastServingLifecycle(worker, config=policy_tests.FastPolicyTests().fixture(), capture_factory=mock.Mock(return_value=capture), bridge_factory=build,
                                             eos_ids=(99,), cancelled=lambda: False)
        self.addCleanup(lambda: None if lifecycle.closed else lifecycle.close())
        return lifecycle, worker, bridge, capture, build, scheduled, decode


class FirstTokenLifecycleTests(Clean):
    def test_a_sampled_request_starts_from_the_keyed_draw_at_the_prompt_length(self):
        with mock.patch.dict(os.environ, {sv.FLAG: '1'}):
            lifecycle, worker, bridge, capture, build, scheduled, _ = self.lifecycle(True)
            sv.attach([SimpleNamespace(sampled=True)])
            request = scheduled.scheduled_new_reqs[0]
            request.sampling_params = sampled_params(request.sampling_params)
            row = np.round(np.random.default_rng(3).normal(size=support.VOCAB) * 16).astype(np.float32) / 8
            plugin = Plugin(lifecycle, worker, bridge, torch.from_numpy(row).reshape(1, -1).clone())
            self.assertIsNone(worker.execute_model(scheduled))
            self.assertIsNotNone(lifecycle.first_token.armed)
            worker.sample_tokens(None)
        expected = sa.sample_logits(row, sa.SamplingSpec(0.7, 0.9, 8), 4242, len(request.prompt_token_ids))
        self.assertEqual(bridge.state.output_token_ids[0], expected)
        self.assertNotEqual(expected, plugin.token, 'the plugin\'s own token was replaced')
        self.assertTrue(torch.equal(plugin.seen[0], torch.from_numpy(row).reshape(1, -1)), 'the plugin\'s sampler ran on the pristine logits')
        self.assertIsNone(lifecycle.first_token.armed)
        build.assert_called_once()
        self.assertEqual(lifecycle.decoding_ids, ['request'])

    def test_a_greedy_request_in_a_sampled_profile_keeps_the_plugins_sampler(self):
        with mock.patch.dict(os.environ, {sv.FLAG: '1'}):
            lifecycle, worker, bridge, capture, build, scheduled, _ = self.lifecycle(True)
            sv.attach([SimpleNamespace(sampled=True)])
            plugin = Plugin(lifecycle, worker, bridge, torch.ones(1, 10))
            worker.execute_model(scheduled)
            self.assertIsNone(lifecycle.first_token.armed)
            worker.sample_tokens(None)
        self.assertEqual(bridge.state.output_token_ids[0], plugin.token)
        self.assertEqual(len(plugin.seen), 1)
        self.assertFalse(lifecycle.failed)

    def test_an_armed_request_the_plugin_sampled_itself_fails_the_engine_closed(self):
        with mock.patch.dict(os.environ, {sv.FLAG: '1'}):
            lifecycle, worker, bridge, capture, build, scheduled, _ = self.lifecycle(True)
            sv.attach([SimpleNamespace(sampled=True)])
            request = scheduled.scheduled_new_reqs[0]
            request.sampling_params = sampled_params(request.sampling_params)
            Plugin(lifecycle, worker, bridge, torch.ones(1, 10), calls_sampler=False)
            worker.execute_model(scheduled)
            with self.assertRaisesRegex(ValueError, 'without the keyed draw'):
                worker.sample_tokens(None)
        self.assertTrue(lifecycle.failed)
        build.assert_not_called()

    def test_a_request_the_gate_refused_is_not_armed_and_draws_nothing(self):
        with mock.patch.dict(os.environ, {sv.FLAG: '1'}):
            lifecycle, worker, bridge, capture, build, scheduled, _ = self.lifecycle(True)
            request = scheduled.scheduled_new_reqs[0]
            request.sampling_params = sampled_params(request.sampling_params)
            # no attach: the gate is closed, so the fixture's lifecycle (no quarantine) raises the old refusal at admission
            with self.assertRaisesRegex(ValueError, 'Fast serving requires a greedy single-sequence request'):
                worker.execute_model(scheduled)
        self.assertIsNone(lifecycle.first_token.armed)

    def test_close_gives_the_plugins_sampler_back(self):
        with mock.patch.dict(os.environ, {sv.FLAG: '1'}):
            lifecycle, worker, bridge, capture, build, scheduled, _ = self.lifecycle(True)
            wrapper = worker.model_runner.host_sampler
            self.assertIsInstance(wrapper, sv.FirstToken)
            inner = wrapper.inner
            lifecycle.close()
        self.assertIs(worker.model_runner.host_sampler, inner)
        self.assertIsNone(lifecycle.first_token)

    def test_a_prefill_sampled_on_the_device_cannot_start_a_sampled_lifecycle(self):
        with self.assertRaisesRegex(sv.SampledRefused, 'host'):
            self.lifecycle(True, runner_options=dict(sample_on_device_mode='all'))
        lifecycle = self.lifecycle(True, runner_options=dict(sample_on_device_mode='decode_only'))[0]
        self.assertIsNotNone(lifecycle.first_token)


class SwitchOffTests(Clean):
    def test_the_lifecycle_never_touches_the_sampler_with_the_switch_off(self):
        with mock.patch.dict(sys.modules, {'sampled_verify': None}):
            lifecycle, worker, bridge, capture, build, scheduled, _ = self.lifecycle(False)
            self.assertIsNone(lifecycle.first_token)
            sampler = worker.model_runner.host_sampler
            worker.execute_model(scheduled)
            worker.sample_tokens(None)
            lifecycle.close()
        self.assertIs(worker.model_runner.host_sampler, sampler)
        self.assertFalse(lifecycle.failed)

    def test_a_sampled_request_is_refused_as_ever_with_the_switch_off(self):
        lifecycle, worker, bridge, capture, build, scheduled, _ = self.lifecycle(False)
        request = scheduled.scheduled_new_reqs[0]
        request.sampling_params = sampled_params(request.sampling_params)
        with self.assertRaisesRegex(ValueError, 'Fast serving requires a greedy single-sequence request'):
            worker.execute_model(scheduled)


if __name__ == '__main__':
    unittest.main()
