"""Sticky sessions, phase 1 (QWEN_FAST_STICKY_SESSIONS): the CPU tests of every path.

Phase 1 retains nothing of a finished request's engine, slot or end-of-turn state (none of it is what a
cold prefill writes); it resumes a continuation exactly from the last 2048-token boundary at or below
its prompt minus 2048 that the prefix-reuse grafts kept: vLLM's KV prefix (published below floor2048(P))
and a host GDN checkpoint (planned at floor2048(P) - 2048, because vLLM drops a DFlash hit's last
block). What is held here:

  flag off   serving_lifecycle, dflash_prefill_window, serving_packed_step.kv_guard and the scheduler
             graft make exactly the calls and decisions of their pre-sticky versions (the merge commit's,
             loaded from git; skipped without history), the registry is never read, the runtime builds
             the capture it always built, the policy refuses the prefix cache, and the contract refuses the
             fast path beside prefix reuse;
  retain     turn N publishes KV up to floor2048(P) and checkpoints floor2048(P) - 2048;
  extend     turn N+1, extending turn N token for token, is granted Q = that checkpoint, admitted by the
             lifecycle at R == Q with a capture counting from R, and prefills only [Q, P');
  mismatch   a divergence below the checkpoint, a shorter fork, another salt, an unsalted request: a lower
             Q or none, i.e. the cold path, never an inexact hit;
  eviction   slot demand (vLLM's LRU takes the prefix blocks: coupled eviction drops the checkpoint), host
             store pressure (the byte LRU drops it) and the kill switch each send the next turn colder;
             the node's session TTL is upstream (design 1: no engine-side TTL), so nothing here ages out;
  abort      a granted request aborted before or during its prefill releases its grant, pin and capture;
  fault      an admission the trim should have made impossible (no grant, a stale one, a misaligned or too
             high R, a split tail, a capture that does not record the route) fails the engine;
  exact      on the G1 model graft's toy (real model.py and qwen36_vllm.py, fake ttnn), the S2 capture
             records the prefix route's slot and a hit equals a cold run: logits, KV and GDN slot;
  guard      two same-tenant users sharing their first block, both crossing a 64-token boundary in one
             round, are not a K/V conflict; a real one still is;
  writers    every device write of a resumed request lands in its own blocks at or above R.
"""

import contextlib
import os
import subprocess
import sys
import types
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock
from unittest.mock import Mock, call, patch

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dflash_prefill_window as window  # noqa: E402
import qwen_prefix_registry as prefix_registry  # noqa: E402
import qwen_prefix_scheduler_patch as graft  # noqa: E402
import serving_c2_contract as contract  # noqa: E402
import serving_fast_policy as policy  # noqa: E402
import serving_lifecycle  # noqa: E402
import serving_packed_step  # noqa: E402
import serving_page_binding  # noqa: E402
import serving_runtime  # noqa: E402
import verify_trace_t2  # noqa: E402

# The fakes other suites own, imported as modules so their TestCases are not collected here again.
import test_qwen_prefix_scheduler_patch as graft_fakes  # noqa: E402
import test_serving_fast_policy as policy_tests  # noqa: E402
import test_serving_request_factory as factory_tests  # noqa: E402
import test_serving_runtime as runtime_tests  # noqa: E402
import test_serving_worker_hook as worker_hook_tests  # noqa: E402

STICKY = 'QWEN_FAST_STICKY_SESSIONS'     # serving_fast_policy.STICKY_SESSIONS_FLAG (PolicyTests holds it)
ON = {STICKY: '1'}
CHUNK = prefix_registry.CHUNK
BLOCK = prefix_registry.BLOCK
# The pre-sticky tree: the merge of the S2 gates head (d9aa19cc) and the hosted G1 release (2676638a).
PARENT = 'ae087e07'


def parent_module(relative, commit=PARENT):
    """scripts/ci/<relative> at `commit`, loaded beside today's modules (its imports resolve to today's
    siblings); None without git history (the integration workflow checks out with fetch-depth 0)."""
    try:
        result = subprocess.run(['git', 'show', '%s:scripts/ci/%s' % (commit, relative)], capture_output=True,
                                cwd=str(HERE), timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    module = ModuleType(relative[:-3] + '_parent')
    module.__file__ = str(HERE / relative)
    exec(compile(result.stdout.decode('utf-8'), '%s@%s' % (relative, commit), 'exec'), module.__dict__)
    return module


@contextmanager
def sticky_env(value):
    """QWEN_FAST_STICKY_SESSIONS exactly as given (None: unset), whatever the shell has set."""
    with mock.patch.dict(os.environ):
        os.environ.pop(STICKY, None)
        if value is not None:
            os.environ[STICKY] = value
        yield


class Tripwire(ModuleType):
    """A registry holder that fails the test if anything reads it."""

    def __getattr__(self, name):
        raise AssertionError('the prefix registry was read (%s) with sticky sessions off' % name)


@contextmanager
def registry_holder(holder):
    with mock.patch.dict(sys.modules):
        sys.modules.pop(prefix_registry.REGISTRY_KEY, None)
        if holder is not None:
            sys.modules[prefix_registry.REGISTRY_KEY] = holder
        yield holder


def committed(registry, req_id, q, tokens):
    """A grant for req_id at q, committed as the scheduler graft's per-step commit does."""
    registry.begin_step()
    checkpoint = None
    if q:
        checkpoint = registry.put('key-%d' % q, q, tokens[0:q], rec=['rec'], carry=['carry'], nbytes=1)
    request = SimpleNamespace(all_token_ids=list(tokens))
    registry.stage(prefix_registry.Grant(req_id, q, q, 'key-%d' % q if q else None, checkpoint, [], request, q))
    registry.commit({req_id: q})
    return registry.grant_for(req_id)


def holder_with(registry):
    holder = ModuleType(prefix_registry.REGISTRY_KEY)
    holder.registry = registry
    return holder


# ------------------------------------------------------------------------------------------------
# The prefill capture (A1)
# ------------------------------------------------------------------------------------------------
class FakeLayerCapture(object):
    """target_features.LayerOutputCapture without the device: the capture's ledger is what is held."""

    def __init__(self, model, layer_ids, snapshot, release, storage_ids):
        self.closed = False

    @contextmanager
    def capture(self):
        yield self

    def outputs(self):
        return ()

    def close(self):
        self.closed = True


class RouteModel(object):
    """The model surface the capture binds: the chunk forward, the stock batched entry, and the prefix
    route, each running the chunks it is asked for through _forward_prefill_chunk_masked_tp."""

    def __init__(self, route=True):
        self.calls = []
        self.chunks = []

        def chunk(token_buf, valid_len, chunk_start, page_table, bucket, **kwargs):
            self.chunks.append((chunk_start, valid_len))
            return 'hidden'

        self._forward_prefill_chunk_masked_tp = chunk

        def prefill_paged_slots(token_ids_list, page_table, empty_slots, valid_lens=None):
            self.calls.append(('stock', list(empty_slots), 0))
            self.run(0, valid_lens[0])
            return 'logits'

        self.prefill_paged_slots = prefill_paged_slots
        if route:
            def route_call(token_ids_list, page_table, empty_slots, valid_lens=None, starts=None, req_ids=None):
                self.calls.append(('route', list(empty_slots), starts[0]))
                self.run(starts[0], valid_lens[0])
                return 'logits'

            self._qwen_prefix_prefill_slots = route_call

    def run(self, start, end):
        cursor = start
        while cursor < end:
            valid = min(CHUNK, end - cursor)
            self._forward_prefill_chunk_masked_tp('tokens', valid, cursor, None, (valid + 31) // 32 * 32)
            cursor += valid


class CaptureTests(unittest.TestCase):
    def capture(self, model, position, **kwargs):
        return window.PrefillWindowCapture(SimpleNamespace(deallocate=Mock()), model, position, (0,), **kwargs)

    def test_a_resumed_capture_counts_from_its_start_through_the_route(self):
        model = RouteModel()
        capture = self.capture(model, 12700, start=6144, prefix_route=True)
        self.assertTrue(capture.records_prefix_route)
        with patch.object(window, 'LayerOutputCapture', FakeLayerCapture), capture.capture():
            model._qwen_prefix_prefill_slots('tokens', 'pages', [2], valid_lens=[12700], starts=[6144],
                                             req_ids=['r'])
        self.assertTrue(capture.complete)
        self.assertEqual(capture.prefill_slot, 2)
        self.assertEqual([chunk['chunk_start'] for chunk in capture.chunks], [6144, 8192, 10240, 12288])
        self.assertEqual(window.validate_prefill_chunks(12700, capture.chunks, 6144),
                         [dict(start=10652, end=12288, rows=1636), dict(start=12288, end=12700, rows=412)])
        self.assertEqual(model.calls, [('route', [2], 6144)])
        self.assertFalse(hasattr(model.__dict__.get('_qwen_prefix_prefill_slots'), '__wrapped__'))
        self.assertNotIn('_qwen_dflash_prefill_capture', vars(model))
        capture.close()

    def test_a_cold_capture_with_the_route_records_its_slot(self):
        model = RouteModel()
        capture = self.capture(model, 5000, prefix_route=True)
        with patch.object(window, 'LayerOutputCapture', FakeLayerCapture), capture.capture():
            model._qwen_prefix_prefill_slots('tokens', 'pages', [1], valid_lens=[5000], starts=[0], req_ids=['r'])
        self.assertEqual((capture.complete, capture.prefill_slot), (True, 1))
        self.assertEqual([chunk['chunk_start'] for chunk in capture.chunks], [0, 2048, 4096])
        capture.close()

    def test_resume_starts_are_refused_off_a_boundary_or_inside_the_draft_window(self):
        model = RouteModel()
        for start in (1000, 2047, -2048, 12288, 10752, True, 2048.0):
            with self.subTest(start=start), self.assertRaisesRegex(ValueError, 'A resumed prefill must start'):
                self.capture(model, 12700, start=start, prefix_route=True)
        self.assertEqual(self.capture(model, 12700, start=10240, prefix_route=True).start, 10240)
        for start in (1000, 12288):
            with self.subTest(validate=start), self.assertRaises(ValueError):
                window.validate_prefill_chunks(12700, [], start)

    def test_a_resume_needs_the_route_wrapped(self):
        with self.assertRaisesRegex(ValueError, 'needs the prefix-reuse route'):
            self.capture(RouteModel(), 12700, start=6144)
        with self.assertRaisesRegex(ValueError, 'needs the prefix-reuse route'):
            self.capture(RouteModel(route=False), 12700, start=6144, prefix_route=True)
        self.assertFalse(self.capture(RouteModel(route=False), 12700, prefix_route=True).records_prefix_route)

    def test_a_route_call_at_the_wrong_start_is_refused_before_it_runs(self):
        for starts in ([0], [4096], [8192], [6144, 0], None, [True]):
            model = RouteModel()
            capture = self.capture(model, 12700, start=6144, prefix_route=True)
            with self.subTest(starts=starts), self.assertRaisesRegex(ValueError, 'must start at this capture'):
                with patch.object(window, 'LayerOutputCapture', FakeLayerCapture), capture.capture():
                    model._qwen_prefix_prefill_slots('tokens', 'pages', [0], valid_lens=[12700], starts=starts)
            self.assertEqual(model.calls, [], 'refused before the native route')

    def test_positional_starts_on_the_route_are_refused(self):
        model = RouteModel()
        capture = self.capture(model, 12700, start=6144, prefix_route=True)
        with self.assertRaisesRegex(ValueError, 'takes valid_lens, starts and req_ids as keywords'):
            with patch.object(window, 'LayerOutputCapture', FakeLayerCapture), capture.capture():
                model._qwen_prefix_prefill_slots('tokens', 'pages', [0], [12700], [6144])
        self.assertEqual(model.calls, [])

    def test_a_resumed_capture_refuses_the_stock_entries(self):
        model = RouteModel()
        capture = self.capture(model, 12700, start=6144, prefix_route=True)
        with self.assertRaisesRegex(ValueError, 'takes its first call only through _qwen_prefix_prefill_slots'):
            with patch.object(window, 'LayerOutputCapture', FakeLayerCapture), capture.capture():
                model.prefill_paged_slots('tokens', 'pages', [0], valid_lens=[12700])
        self.assertEqual(model.calls, [])

    def test_without_prefix_route_the_route_is_never_bound(self):
        model = RouteModel()
        capture = self.capture(model, 5000)
        self.assertFalse(capture.records_prefix_route)
        names = [name for _, name, _ in capture.bindings()]
        self.assertNotIn(window.PREFIX_ROUTE_ENTRY, names)
        self.assertIn(window.PREFIX_ROUTE_ENTRY, [name for _, name, _ in
                                                  self.capture(model, 5000, prefix_route=True).bindings()])


class CaptureParityTests(unittest.TestCase):
    """The default capture (start 0, no route) is the pre-sticky capture, call for call."""

    def drive(self, module, slots, route=True):
        model = RouteModel(route=route)

        def resumed(token_ids_list, page_table, empty_slots, starts, ends, valid_lens=None):
            model.calls.append(('range', list(empty_slots), starts[0]))
            model.run(starts[0], ends[0])
            return 'logits'

        model.prefill_paged_slots_range = resumed
        capture = module.PrefillWindowCapture(SimpleNamespace(deallocate=Mock()), model, CHUNK * len(slots), (0,))
        trace = []
        with patch.object(module, 'LayerOutputCapture', FakeLayerCapture), patch.object(module, '_log') as log:
            for index, slot in enumerate(slots):
                try:
                    with capture.segment():
                        if index == 0:
                            model.prefill_paged_slots('t', 'p', [slot], valid_lens=[CHUNK])
                        else:
                            model.prefill_paged_slots_range('t', 'p', [slot], [index * CHUNK], [(index + 1) * CHUNK],
                                                            valid_lens=[(index + 1) * CHUNK])
                    trace.append(('ok', capture.segment_slot))
                except ValueError as error:
                    trace.append(('raise', str(error)))
                    break
        return (trace, model.calls, model.chunks, capture.chunks, capture.prefill_slot, capture.complete,
                sorted(name for _, name, _ in capture.bindings()) if not capture.closed else None,
                [entry.args for entry in log.call_args_list])

    def test_the_default_capture_matches_the_parent(self):
        parent = parent_module('dflash_prefill_window.py')
        if parent is None:
            self.skipTest('no git history for %s' % PARENT)
        for slots in ([0], [1, 1, 1], [2, 2, 0, 0], [0, 3]):
            for route in (True, False):
                with self.subTest(slots=slots, route=route):
                    self.assertEqual(self.drive(window, slots, route), self.drive(parent, slots, route))
        for position in (170, 2048, 4093, 6144):
            chunks = [dict(chunk_start=start, valid_rows=min(CHUNK, position - start),
                           bucket=(min(CHUNK, position - start) + 31) // 32 * 32,
                           retained_rows=window.chunk_window(position, start, min(CHUNK, position - start))['rows'])
                      for start in range(0, position, CHUNK)]
            self.assertEqual(window.validate_prefill_chunks(position, chunks),
                             parent.validate_prefill_chunks(position, chunks))


if __name__ == '__main__':
    unittest.main()
