"""Host-only scheduling and lifecycle tests using staged plugin method bodies."""

import ast
import os
import sys
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import numpy as np

import interleave
from continuation import ContinuationController
from test_continuation import RecordingBackend

SOURCE = Path(os.environ.get("SIM_ROOT", "/opt/ttsim")) / "vllm-tt-plugin/src/vllm_tt_plugin"


class Mode(Enum):
    DEFAULT = 0
    PREFILL_ONLY = 1
    DECODE_ONLY = 2

    @classmethod
    def from_prefill_intent(cls, value):
        return cls.PREFILL_ONLY if value else cls.DECODE_ONLY


class Queue(list):
    def prepend_requests(self, requests):
        self[:0] = list(requests)


class BaseScheduler:
    def schedule(self):
        if getattr(self, "fail", False):
            raise RuntimeError("base scheduling failure")
        if getattr(self, "preempt", False):
            self.preempt = False
            self.waiting.extend(self.running)
            self.running.clear()
            return SimpleNamespace(total_num_scheduled_tokens=0, scheduled=[])
        scheduled = list(self.running)
        while self.skipped_waiting and len(self.running) < self.max_num_running_reqs:
            request = self.skipped_waiting.pop(0)
            self.running.append(request)
            scheduled.append(request)
        while self.waiting and len(self.running) < self.max_num_running_reqs:
            request = self.waiting.pop(0)
            self.running.append(request)
            scheduled.append(request)
        return SimpleNamespace(total_num_scheduled_tokens=len(scheduled),
                               scheduled=[request.request_id for request in scheduled])


def scheduler_class():
    tree = ast.parse((SOURCE / "scheduler.py").read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TTScheduler")
    selected = {"schedule", "_schedule_prefill_only", "_schedule_decode_only", "_has_pending_prefill",
                "_finalize_scheduler_output"}
    methods = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in selected]
    for node in ast.walk(ast.Module(body=methods, type_ignores=[])):
        if isinstance(node, ast.FunctionDef):
            node.returns = None
        if isinstance(node, ast.arg):
            node.annotation = None
    cls = ast.ClassDef(name="Scheduler", bases=[ast.Name(id="BaseScheduler", ctx=ast.Load())],
                       keywords=[], body=methods, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = dict(BaseScheduler=BaseScheduler, TTSchedulingMode=Mode, cast=lambda kind, value: value,
                     Request=object, create_request_queue=lambda policy: Queue())
    exec(compile(module, "scheduler.py", "exec"), namespace)
    return namespace["Scheduler"]


def method(filename, name):
    tree = ast.parse((SOURCE / filename).read_text())
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    node.returns = None
    for argument in ast.walk(node.args):
        if isinstance(argument, ast.arg):
            argument.annotation = None
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, "exec"), namespace)
    return namespace[name]


class InterleaveTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, QWEN_PREFILL_CONTINUATION="1",
                                      TT_PREFILL_DECODE_INTERLEAVE="1", TT_DECODE_STEPS_PER_PREFILL_CHUNK="1")
        self.environment.start()
        self.addCleanup(self.environment.stop)
        modules = {"vllm_tt_plugin.qwen_interleave": interleave,
                   "vllm.v1.core.sched.request_queue": SimpleNamespace(create_request_queue=lambda policy: Queue()),
                   "vllm_tt_plugin.config": SimpleNamespace(get_tt_data_parallel_size=lambda config: 1)}
        self.modules = patch.dict(sys.modules, modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def scheduler(self, partial=False):
        scheduler = scheduler_class()()
        scheduler.running = [SimpleNamespace(request_id="B", is_prefill_chunk=False)]
        if partial:
            scheduler.running.append(SimpleNamespace(request_id="A", is_prefill_chunk=True))
        scheduler.waiting = Queue([SimpleNamespace(request_id="C", is_prefill_chunk=True),
                                   SimpleNamespace(request_id="D", is_prefill_chunk=True)])
        scheduler.skipped_waiting = Queue()
        scheduler.max_num_running_reqs, scheduler.policy = 8, "fcfs"
        scheduler._forced_mode = Mode.DEFAULT
        return scheduler

    def test_ratios(self):
        for ratio in (1, 2, 4):
            with self.subTest(ratio=ratio), patch.dict(os.environ, TT_DECODE_STEPS_PER_PREFILL_CHUNK=str(ratio)):
                scheduler = self.scheduler(partial=True)
                outputs = [scheduler.schedule().scheduled for _ in range(2 * (ratio + 1))]
                self.assertEqual(outputs, ([["A"]] + [["B"]] * ratio) * 2)
                self.assertEqual([request.request_id for request in scheduler.waiting], ["C", "D"])

    def test_new_admission_is_single_owner(self):
        scheduler = self.scheduler()
        self.assertEqual(scheduler.schedule().scheduled, ["C"])
        self.assertEqual([request.request_id for request in scheduler.waiting], ["D"])
        self.assertEqual(scheduler.max_num_running_reqs, 8)

    def test_skipped_queue_hidden_for_continuation(self):
        scheduler = self.scheduler(partial=True)
        scheduler.skipped_waiting.append(SimpleNamespace(request_id="grammar", is_prefill_chunk=True))
        self.assertEqual(scheduler.schedule().scheduled, ["A"])
        self.assertEqual(len(scheduler.skipped_waiting), 1)

    def test_queues_and_capacity_restored_on_exception(self):
        scheduler = self.scheduler(partial=True)
        waiting, skipped = scheduler.waiting, scheduler.skipped_waiting
        scheduler.fail = True
        with self.assertRaises(RuntimeError):
            scheduler.schedule()
        self.assertIs(scheduler.waiting, waiting)
        self.assertIs(scheduler.skipped_waiting, skipped)
        self.assertEqual(scheduler.max_num_running_reqs, 8)
        self.assertEqual({request.request_id for request in scheduler.running}, {"A", "B"})

    def test_preempted_partial_returns_to_original_queue(self):
        scheduler = self.scheduler(partial=True)
        scheduler.preempt = True
        scheduler.schedule()
        self.assertEqual([request.request_id for request in scheduler.waiting], ["A", "C", "D"])
        self.assertEqual([request.request_id for request in scheduler.running], ["B"])

    def test_multiple_partial_owners_fail(self):
        scheduler = self.scheduler(partial=True)
        scheduler.running.append(SimpleNamespace(request_id="X", is_prefill_chunk=True))
        with self.assertRaises(RuntimeError):
            scheduler.schedule()

    def test_flags_off_preserve_legacy_admission(self):
        with patch.dict(os.environ, QWEN_PREFILL_CONTINUATION="0", TT_PREFILL_DECODE_INTERLEAVE="0"):
            self.assertEqual(self.scheduler().schedule().scheduled, ["C", "D"])

    def test_no_decode_has_no_artificial_pause(self):
        scheduler = self.scheduler(partial=True)
        scheduler.running = [request for request in scheduler.running if request.is_prefill_chunk]
        self.assertEqual([scheduler.schedule().scheduled for _ in range(3)], [["A"]] * 3)

    def test_full_capacity_falls_back_to_decode(self):
        scheduler = self.scheduler()
        scheduler.max_num_running_reqs = 1
        self.assertEqual(scheduler.schedule().scheduled, ["B"])
        self.assertEqual(len(scheduler.waiting), 2)

    def test_invalid_flags_and_ratio(self):
        with patch.dict(os.environ, QWEN_PREFILL_CONTINUATION="0"):
            with self.assertRaises(ValueError):
                interleave.enabled()
        with patch.dict(os.environ, TT_DECODE_STEPS_PER_PREFILL_CHUNK="0"):
            with self.assertRaises(ValueError):
                self.scheduler().schedule()

    def test_config_gate(self):
        config = SimpleNamespace(
            scheduler_config=SimpleNamespace(enable_chunked_prefill=True, max_num_batched_tokens=2048,
                                             max_num_seqs=8, long_prefill_token_threshold=0),
            parallel_config=SimpleNamespace(data_parallel_size=1),
            model_config=SimpleNamespace(model="Qwen/Qwen3.8-27B"), speculative_config=None,
            cache_config=SimpleNamespace(enable_prefix_caching=False))
        method("platform.py", "_apply_chunked_prefill_policy")(config)
        self.assertTrue(config.scheduler_config.disable_chunked_mm_input)
        with patch.object(config.model_config, "model", interleave.REVIEWED_SNAPSHOT):
            self.assertTrue(interleave.validate_config(config))
        for model in ("Qwen/Qwen3.6-27B", interleave.REVIEWED_SNAPSHOT + "-unreviewed", "/tmp/Qwen3.8-27B"):
            with self.subTest(model=model), patch.object(config.model_config, "model", model):
                with self.assertRaises(ValueError):
                    interleave.validate_config(config)
        for target, attribute, value in ((config.scheduler_config, "max_num_batched_tokens", 1024),
                                          (config.cache_config, "enable_prefix_caching", True),
                                          (config.parallel_config, "data_parallel_size", 2)):
            with self.subTest(attribute=attribute), patch.object(target, attribute, value):
                with self.assertRaises(ValueError):
                    interleave.validate_config(config)

    def runner(self):
        backend = RecordingBackend()
        controller = ContinuationController(backend)
        events = []
        wrapper = SimpleNamespace(model=[SimpleNamespace(_chunked_chunk_size=32)],
                                  _prefill_continuation_controller=controller)
        wrapper.prefill_forward = lambda **kwargs: controller.forward(
            kwargs["tokens"], kwargs["page_table"], kwargs["prompt_lens"], kwargs)
        runner = SimpleNamespace(model=wrapper, scheduler_config=SimpleNamespace(max_num_batched_tokens=32),
                                 async_decode=SimpleNamespace(wait_for_all_pending_async_steps=lambda: events.append("drain")),
                                 request_specific_rope=True, requests={"A": SimpleNamespace()},
                                 _req_state_slot={"A": 2}, kv_caches=None, trace_mode="all")
        return runner, controller, events

    def inputs(self, runner, start=0, end=32, final=False):
        return SimpleNamespace(prefill_request_identity=interleave.identities(runner, ["A"]),
                               perform_device_sampling=False, block_tables_per_layer=None,
                               intermediate_prefill_mask=torch.tensor([not final]),
                               input_tokens=torch.arange(end).reshape(1, -1), block_tables=torch.tensor([[5, 1, 6]]),
                               prompt_lens=[end], input_positions=[start], multi_modal_kwargs={}, prefill_empty_slots=[2])

    def test_runner_forwards_identity_marker_and_drains(self):
        runner, controller, events = self.runner()
        submit = method("model_runner.py", "submit_prefill")
        submit(runner, self.inputs(runner), [1])
        submit(runner, self.inputs(runner, 32, 64, True), [1])
        self.assertEqual(controller.backend.calls, [(0, 32, False, 2), (32, 64, True, 2)])
        self.assertEqual(events, ["drain", "drain"])
        self.assertEqual(runner.requests["A"].mrope_position_delta, 0)

    def test_real_plugin_text_metadata_reaches_continuation(self):
        runner, controller, _ = self.runner()
        runner.input_batch = SimpleNamespace(num_reqs=1, req_ids=["A"])
        runner.requests["A"].mm_features = []
        inputs = self.inputs(runner, end=15, final=True)
        inputs.input_positions = np.array([0], dtype=np.int32)
        inputs.prompt_lens = np.array([15], dtype=np.int32)
        inputs.multi_modal_kwargs = method("model_runner.py", "_gather_multi_modal_inputs")(runner)
        self.assertEqual(inputs.multi_modal_kwargs, {"pixel_values": [None], "image_grid_thw": [None]})
        method("model_runner.py", "submit_prefill")(runner, inputs, [1])
        self.assertEqual(controller.backend.calls, [(0, 15, True, 2)])

    def test_finished_request_cancels_and_stale_input_rejected(self):
        runner, controller, events = self.runner()
        inputs = self.inputs(runner)
        submit = method("model_runner.py", "submit_prefill")
        submit(runner, inputs, [1])
        release = method("model_runner.py", "_release_dead_state_slots")
        release(runner, SimpleNamespace(finished_req_ids={"A"}, preempted_req_ids=None))
        self.assertIsNone(controller.active)
        self.assertFalse(runner._req_state_slot)
        with self.assertRaises(ValueError):
            submit(runner, inputs, [1])
        replacement = self.inputs(runner)
        self.assertNotEqual(replacement.prefill_request_identity, inputs.prefill_request_identity)
        submit(runner, replacement, [1])

    def test_preemption_restarts_at_zero_with_new_generation(self):
        runner, controller, events = self.runner()
        submit = method("model_runner.py", "submit_prefill")
        submit(runner, self.inputs(runner), [1])
        interleave.release_requests(runner, SimpleNamespace(finished_req_ids=set(), preempted_req_ids={"A"}))
        submit(runner, self.inputs(runner), [1])
        self.assertEqual([call[0] for call in controller.backend.calls], [0, 0])

    def test_completed_input_cannot_replay(self):
        runner, controller, events = self.runner()
        inputs = self.inputs(runner, final=True)
        submit = method("model_runner.py", "submit_prefill")
        submit(runner, inputs, [1])
        with self.assertRaises(ValueError):
            submit(runner, inputs, [1])

    def test_budget_mismatch_fails_before_work(self):
        runner, controller, events = self.runner()
        runner.scheduler_config.max_num_batched_tokens = 16
        with self.assertRaises(ValueError):
            method("model_runner.py", "submit_prefill")(runner, self.inputs(runner), [1])
        self.assertFalse(controller.backend.calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
