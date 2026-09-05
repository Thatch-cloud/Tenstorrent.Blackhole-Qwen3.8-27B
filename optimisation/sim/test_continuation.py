"""Host contract tests, including methods extracted from the staged upstream source."""

import ast
import os
import sys
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from continuation import ContinuationController, TPPrefillBackend

SOURCE = Path(os.environ.get("SIM_ROOT", "/opt/ttsim")) / "tt-metal"


def source_method(filename, method, namespace):
    tree = ast.parse((SOURCE / "models/demos/blackhole/qwen36/tt" / filename).read_text())
    candidates = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == method]
    if len(candidates) != 1:
        raise AssertionError(f"Expected one {method}")
    module = ast.Module(body=[candidates[0]], type_ignores=[])
    exec(compile(module, filename, "exec"), namespace)
    return namespace[method]


class RecordingBackend:
    chunk_size, block_size, max_length, max_slots, page_capacity, physical_pages, vocab_size = 32, 32, 256, 8, 8, 16, 1024

    def __init__(self):
        self.calls = []
        self.fail = False
        self.drains = 0

    def run(self, tokens, pages, start, end, final, slot):
        self.calls.append((start, end, final, slot))
        if self.fail:
            raise RuntimeError("Injected backend failure")
        return torch.zeros(1, 1, self.vocab_size), torch.zeros(1, dtype=torch.long)

    def drain(self):
        self.drains += 1


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.backend = RecordingBackend()
        self.controller = ContinuationController(self.backend)
        self.tokens = torch.arange(96).reshape(1, -1)
        self.pages = torch.tensor([[5, 1, 6]])

    def call(self, start=0, end=32, final=False, owner="A", epoch=0, slot=2, **changes):
        metadata = dict(start_pos=[start], intermediate_prefill_mask=[not final], request_ids=[owner],
                        request_epochs=[epoch], empty_slots=[slot])
        metadata.update(changes)
        return self.controller.forward(self.tokens[:, :end], self.pages, [end], metadata)

    def test_full_chunks_then_tail(self):
        self.call()
        self.call(32, 64)
        self.call(64, 79, True, slot=4)
        self.assertEqual(self.backend.calls, [(0, 32, False, 2), (32, 64, False, 2), (64, 79, True, 4)])
        self.assertIsNone(self.controller.active)

    def test_short_and_aligned_final(self):
        self.call(0, 15, True)
        self.call(0, 32, True)
        self.assertIsNone(self.controller.active)

    def test_explicit_final_marker_not_token_width(self):
        self.call()
        self.assertEqual(self.controller.active["end"], 32)
        self.assertFalse(self.backend.calls[-1][2])

    def test_missing_metadata_rejected_without_work(self):
        with self.assertRaises(ValueError):
            self.controller.forward(self.tokens, self.pages, [32], {"start_pos": [0]})
        self.assertFalse(self.backend.calls)

    def test_other_owner_cannot_reset_scratch(self):
        self.call()
        with self.assertRaises(ValueError):
            self.call(0, 15, True, owner="B")
        self.call(32, 64, True)
        self.assertEqual(len(self.backend.calls), 2)

    def test_stale_epoch_gap_and_duplicate(self):
        self.call()
        for start, epoch in ((32, 1), (64, 0), (0, 0)):
            with self.subTest(start=start, epoch=epoch), self.assertRaises(ValueError):
                self.call(start, start + 32, epoch=epoch)
        self.assertEqual(len(self.backend.calls), 1)

    def test_prefix_mutation(self):
        self.call()
        self.tokens[0, 0] = 90
        with self.assertRaises(ValueError):
            self.call(32, 64)

    def test_page_remapping(self):
        self.call()
        self.pages[0, 0] = 7
        with self.assertRaises(ValueError):
            self.call(32, 64)

    def test_invalid_ranges_and_metadata(self):
        for options in (dict(end=15), dict(end=64), dict(start=1), dict(start=0.0), dict(slot=-1),
                        dict(slot=8), dict(intermediate_prefill_mask=[0]), dict(pixel_values=torch.ones(1))):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.call(**options)
        self.assertFalse(self.backend.calls)

    def test_bad_page_or_token_id(self):
        self.pages[0, 0] = 16
        with self.assertRaises(ValueError):
            self.call()
        self.pages[0, 0] = 5
        self.tokens[0, 0] = -1
        with self.assertRaises(ValueError):
            self.call()

    def test_cancellation_and_slot_reuse(self):
        self.call()
        with self.assertRaises(ValueError):
            self.controller.cancel("A", 1)
        self.controller.cancel("A", 0)
        with self.assertRaises(ValueError):
            self.call(32, 64)
        self.call(0, 32, True, owner="B", epoch=1)

    def test_failed_execution_poisoned_until_cancel(self):
        self.backend.fail = True
        with self.assertRaises(RuntimeError):
            self.call()
        self.assertTrue(self.controller.poisoned)
        with self.assertRaises(RuntimeError):
            self.call(owner="B")
        self.controller.cancel("A", 0)
        self.backend.fail = False
        self.call(0, 32, True, owner="B")

    def test_failed_cancel_keeps_owner_and_poison(self):
        self.backend.fail = True
        with self.assertRaises(RuntimeError):
            self.call()
        with patch.object(self.backend, "drain", side_effect=RuntimeError("drain failed")):
            with self.assertRaises(RuntimeError):
                self.controller.cancel("A", 0)
        self.assertTrue(self.controller.poisoned)
        self.assertEqual(self.controller.active["owner"], ("A", 0))

    def test_insufficient_page_capacity(self):
        self.backend.page_capacity = 0
        with self.assertRaises(ValueError):
            self.call()
        self.assertFalse(self.backend.calls)


class FakeTTNN:
    int32, uint32, bfloat16 = torch.int32, torch.int32, torch.bfloat16
    ROW_MAJOR_LAYOUT, TILE_LAYOUT = "row", "tile"

    def __init__(self, model):
        self.model = model

    def ReplicateTensorToMesh(self, device):
        return None

    def ConcatMeshToTensor(self, device, dim):
        return None

    def from_torch(self, tensor, **kwargs):
        return tensor.clone()

    def to_torch(self, tensor, **kwargs):
        return tensor

    def copy_host_to_device_tensor(self, source, destination):
        destination.copy_(source)

    def synchronize_device(self, device):
        self.model.events.append("drain")

    def deallocate(self, tensor):
        pass

    def execute_trace(self, device, trace, **kwargs):
        start = int(self.model._chunk_start_idx_tensor[0])
        self.model.starts.append(start)
        self.model.processed.extend(self.model._chunk_token_buf.flatten().tolist())
        self.model.events.append("trace")
        if self.model.fail:
            raise RuntimeError("Injected trace failure")


class TraceModel:
    def __init__(self):
        self.device = self.mesh_device = "mock-device"
        self.num_devices = 2
        self.args = SimpleNamespace(max_seq_len=256, max_batch_size=8, vocab_size=4)
        self._chunked_chunk_size = 32
        self._chunked_trace_id = 1
        self._paged_kv_caches = [(torch.zeros(16, 2, 32, 256),) * 2]
        self._chunk_full_page_table_buf = torch.zeros(1, 8, dtype=torch.int32)
        self._chunk_page_table_buf = torch.zeros(1, 1, dtype=torch.int32)
        self._chunk_token_buf = torch.zeros(1, 32, dtype=torch.int32)
        self._chunk_start_idx_tensor = torch.zeros(1, dtype=torch.int32)
        self._chunk_cos_buf = self._chunk_sin_buf = torch.zeros(1)
        self._chunked_trace_output = torch.zeros(1, 1, 4)
        self.layers = []
        self.events, self.starts, self.processed, self.writes = [], [], [], []
        self.fail = False
        self.bound = False
        self.runtime = FakeTTNN(self)
        namespace = dict(torch=torch, ttnn=self.runtime, os=os, get_block_size=lambda cache: 32,
                         logger=SimpleNamespace(info=lambda message: None))
        self._prefill_traced_chunked_tp = MethodType(source_method("model.py", "_prefill_traced_chunked_tp", namespace), self)

    def _reset_gdn_state_for_new_sequence(self):
        self.events.append("reset")
        self.processed.clear()

    def _rope_tp_cos_sin_torch(self, start, size):
        return torch.zeros(1), torch.zeros(1)

    def _set_vision_merge(self, *args):
        pass

    def _vis_row_offset_for(self, *args):
        return 0

    def _masked_bucket_logits_tp(self, *args):
        self.events.append("logits")
        return torch.full((1, 1, 4), float(sum(self.processed)))

    def prefill_masked_bucket(self, tokens, pages, actual_len, chunk_start, **kwargs):
        if chunk_start == 0:
            self._reset_gdn_state_for_new_sequence()
        self.starts.append(chunk_start)
        self.processed.extend(tokens.flatten().tolist())
        return self._masked_bucket_logits_tp()

    def _build_request_rope(self, *args):
        self.events.append("rope")

    def _bind_gdn_prefill_scratch(self):
        self.bound = True
        self.events.append("bind")
        return "saved-decode-buffers"

    def _unbind_gdn_prefill_scratch(self, previous):
        self.bound = False
        self.events.append("unbind")

    def _write_gdn_slot(self, slot, *args):
        if self.bound:
            raise AssertionError("Decode slot written while scratch still bound")
        self.writes.append(slot)


class UpstreamIntegrationTests(unittest.TestCase):
    def test_legacy_defaults_and_short_final(self):
        for length in (15, 32, 79):
            with self.subTest(length=length):
                model = TraceModel()
                output = model._prefill_traced_chunked_tp(
                    torch.arange(length).reshape(1, -1), torch.tensor([[5, 1, 6]]),
                    length, length // 32, 32, length % 32)
                self.assertEqual(model.processed, list(range(length)))
                self.assertEqual(float(output[0, 0, 0]), sum(range(length)))

    def test_final_snapshot_is_copied_before_unbinding(self):
        model = TraceModel()
        state = SimpleNamespace(rec_state=torch.tensor([7.0]), conv_states=[torch.tensor([9.0])])
        model.layers = [SimpleNamespace(is_full_attention=False, attention=state)]
        original_unbind = model._unbind_gdn_prefill_scratch
        def unbind(previous):
            state.rec_state.zero_()
            state.conv_states[0].zero_()
            original_unbind(previous)
        model._unbind_gdn_prefill_scratch = unbind
        with patch.dict(sys.modules, ttnn=model.runtime):
            backend = TPPrefillBackend(model)
        with patch.object(model, "_write_gdn_slot") as write:
            backend.run(torch.arange(32).reshape(1, -1), torch.tensor([[5]]), 0, 32, True, 2)
        slot, recurrent, convolution = write.call_args.args
        self.assertEqual(slot, 2)
        self.assertEqual(recurrent[0].item(), 7)
        self.assertEqual(convolution[0][0].item(), 9)
        self.assertFalse(model.bound)

    def test_failed_slot_publication_requires_restart(self):
        model = TraceModel()
        with patch.dict(sys.modules, ttnn=model.runtime):
            backend = TPPrefillBackend(model)
        with patch.object(model, "_write_gdn_slot", side_effect=RuntimeError("partial slot write")):
            with self.assertRaises(RuntimeError):
                backend.run(torch.arange(32).reshape(1, -1), torch.tensor([[5]]), 0, 32, True, 2)
        self.assertTrue(backend.restart_required)
        with self.assertRaisesRegex(RuntimeError, "requires restart"):
            backend.drain()

    def test_failed_trace_drain_retains_host_references(self):
        model = TraceModel()
        with patch.object(model.runtime, "synchronize_device", side_effect=RuntimeError("drain failed")):
            with self.assertRaises(RuntimeError):
                model._prefill_traced_chunked_tp(
                    torch.arange(32).reshape(1, -1), torch.tensor([[5]]), 32, 1, 32, 0,
                    start_pos=0, is_last=False)
        self.assertIn("pt_host", model._prefill_failed_dma_refs)
        self.assertEqual(len(model._prefill_failed_dma_refs["_host_refs"]), 5)

    def test_range_replay_skips_prefix_and_suppresses_logits(self):
        model = TraceModel()
        tokens = torch.arange(79).reshape(1, -1)
        pages = torch.tensor([[5, 1, 6]])
        model._prefill_traced_chunked_tp(tokens, pages, 32, 1, 32, 0, start_pos=0, is_last=False)
        model._prefill_traced_chunked_tp(tokens, pages, 64, 2, 32, 0, start_pos=32, is_last=False)
        self.assertNotIn("logits", model.events)
        model._prefill_traced_chunked_tp(tokens, pages, 79, 2, 32, 15, start_pos=64, is_last=True)
        self.assertEqual(model.starts, [0, 32, 64])
        self.assertEqual(model.processed, list(range(79)))
        self.assertEqual(model.events.count("reset"), 1)
        self.assertEqual(model.events.count("logits"), 1)

    def test_backend_publishes_only_final_and_restores_decode_binding(self):
        model = TraceModel()
        with patch.dict(sys.modules, ttnn=model.runtime):
            backend = TPPrefillBackend(model)
        tokens, pages = torch.arange(64).reshape(1, -1), torch.tensor([[5, 1]])
        intermediate, _ = backend.run(tokens[:, :32], pages, 0, 32, False, 2)
        self.assertEqual(intermediate.shape, (1, 1, 4))
        self.assertFalse(model.bound)
        self.assertEqual(model.writes, [])
        backend.run(tokens, pages, 32, 64, True, 3)
        self.assertEqual(model.writes, [3])
        self.assertEqual(model.events.count("rope"), 1)
        self.assertEqual(model.events.count("reset"), 1)

    def test_backend_restores_binding_after_trace_failure(self):
        model = TraceModel()
        model.fail = True
        with patch.dict(sys.modules, ttnn=model.runtime):
            backend = TPPrefillBackend(model)
        with self.assertRaises(RuntimeError):
            backend.run(torch.arange(32).reshape(1, -1), torch.tensor([[5]]), 0, 32, False, 2)
        self.assertFalse(model.bound)
        self.assertFalse(model.writes)
        self.assertEqual(model.events[-1], "unbind")

    def test_wrapper_routes_metadata_only_when_enabled(self):
        calls = []
        namespace = dict(os=os)
        forward = source_method("qwen36_vllm.py", "prefill_forward", namespace)
        wrapper = SimpleNamespace(model=[SimpleNamespace(num_devices=2, args=SimpleNamespace(max_batch_size=8))],
                                  _has_visual=lambda *args: False,
                                  _prefill_forward_tp_batched=lambda *args: "legacy")
        module = SimpleNamespace(dispatch_prefill=lambda *args: calls.append(args) or "continuation")
        with patch.dict(sys.modules, qwen_prefill_continuation=module):
            with patch.dict(os.environ, QWEN_PREFILL_CONTINUATION="0"):
                self.assertEqual(forward(wrapper, None, None, None, None), "legacy")
            with patch.dict(os.environ, QWEN_PREFILL_CONTINUATION="1"):
                self.assertEqual(forward(wrapper, None, None, None, [64], start_pos=[32]), "continuation")
        self.assertEqual(calls[0][-1]["start_pos"], [32])


if __name__ == "__main__":
    unittest.main(verbosity=2)
