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
# The lifecycle (A3)
# ------------------------------------------------------------------------------------------------
class LifecycleFixture(unittest.TestCase):
    def fixture(self, cls=serving_lifecycle.FastServingLifecycle, prompt=4096, computed=0, chunk=None,
                records_route=True):
        worker, bridge, _, _ = worker_hook_tests.WorkerHookTests().fixture()
        worker.model_runner.execute_model.return_value = None
        worker.model_runner.sample_tokens.return_value = SimpleNamespace(req_ids=['request'], sampled_token_ids=[[10]])
        _, _, _, arguments = factory_tests.RequestFactoryTests().fixture()
        chunk = prompt - computed if chunk is None else chunk
        new = SimpleNamespace(req_id='request', prompt_token_ids=[1] * prompt, num_computed_tokens=computed,
                              mm_features=[], prompt_embeds=None, lora_request=None,
                              sampling_params=arguments['state'].sampling_params)
        scheduled = SimpleNamespace(finished_req_ids=set(), scheduled_new_reqs=[new],
                                    scheduled_cached_reqs=SimpleNamespace(req_ids=[]), scheduled_spec_decode_tokens={},
                                    num_scheduled_tokens={'request': chunk}, total_num_scheduled_tokens=chunk)
        capture = SimpleNamespace(capture=Mock(side_effect=lambda: nullcontext()), close=Mock(),
                                  segment=Mock(side_effect=lambda: nullcontext()), complete=False,
                                  records_prefix_route=records_route)

        def factory(state, features):
            features.close()
            return bridge

        build = Mock(side_effect=factory)
        capture_factory = Mock(return_value=capture)
        lifecycle = cls(worker, config=policy_tests.FastPolicyTests().fixture(), capture_factory=capture_factory,
                        bridge_factory=build, eos_ids=(99,), cancelled=lambda: False)
        # the runner's own methods, as the fixture made them (a hook replaces them once it attaches)
        return SimpleNamespace(lifecycle=lifecycle, worker=worker, bridge=bridge, capture=capture, build=build,
                               scheduled=scheduled, new=new, capture_factory=capture_factory,
                               runner_execute=worker.model_runner.execute_model,
                               runner_sample=worker.model_runner.sample_tokens)


class StickyAdmissionTests(LifecycleFixture):
    """A granted hit is admitted at R == Q; everything else the trim should have made impossible fails
    the engine, as the uncached clause's refusal always did."""

    def admit(self, prompt, computed, q, chunk=None, records_route=True, holder=True):
        registry = prefix_registry.PrefixRegistry(budget_bytes=1 << 30)
        tokens = [1] * prompt
        if q is not None:
            committed(registry, 'request', q, tokens)
        with sticky_env('1'):
            case = self.fixture(prompt=prompt, computed=computed, chunk=chunk, records_route=records_route)
        with registry_holder(holder_with(registry) if holder else None):
            try:
                case.result = case.worker.execute_model(case.scheduled)
                case.error = None
            except ValueError as error:
                case.error = error
        return case

    def test_a_granted_exact_extension_is_admitted_at_its_boundary(self):
        logger = Mock()
        with patch.dict(sys.modules, {'loguru': SimpleNamespace(logger=logger)}):
            case = self.admit(8192, 4096, 4096)
        info = logger.info
        self.assertIsNone(case.error)
        self.assertTrue(case.lifecycle.sticky)
        case.capture_factory.assert_called_once_with(8192, start=4096)
        case.capture.capture.assert_called_once_with()
        case.runner_execute.assert_called_once_with(case.scheduled)
        self.assertTrue(case.lifecycle.prefill_pending, 'the whole tail ran: the seed is due')
        self.assertEqual(case.lifecycle.request_id, 'request')
        self.assertIn("[PINDIAG] sticky admit req='request' Q=4096 P=8192 tail=4096",
                      [entry.args[0] for entry in info.call_args_list])
        # the seed is bridged as any other: the engine is built from the capture as on the cold path
        self.assertIs(case.worker.sample_tokens(None), case.runner_sample.return_value)
        case.build.assert_called_once()
        self.assertEqual(case.lifecycle.decoding_ids, ['request'])

    def test_the_largest_resume_leaves_the_draft_window_to_the_prefill(self):
        case = self.admit(6144, 4096, 4096)
        self.assertIsNone(case.error)
        case.capture_factory.assert_called_once_with(6144, start=4096)

    def assert_fault(self, case, words):
        self.assertIsNotNone(case.error)
        self.assertIn(words, str(case.error))
        self.assertTrue(case.lifecycle.failed, 'an impossible admission fails the engine')
        case.runner_execute.assert_not_called()

    def test_no_committed_grant_is_an_engine_fault(self):
        self.assert_fault(self.admit(8192, 4096, None), 'no committed prefix grant')

    def test_no_registry_in_the_process_is_an_engine_fault(self):
        self.assert_fault(self.admit(8192, 4096, None, holder=False), 'no committed prefix grant')

    def test_a_stale_grant_is_an_engine_fault(self):
        self.assert_fault(self.admit(8192, 4096, 2048), 'the committed grant is at Q=2048')
        self.assert_fault(self.admit(8192, 4096, 0), 'the committed grant is at Q=0')

    def test_a_misaligned_resume_is_an_engine_fault(self):
        case = self.admit(8192, 4032, None)
        self.assert_fault(case, 'R is not a 2048-token boundary')

    def test_a_resume_inside_the_draft_window_is_an_engine_fault(self):
        self.assert_fault(self.admit(8000, 6144, 6144), 'R is above the prompt minus 2048')

    def test_a_split_tail_is_an_engine_fault(self):
        self.assert_fault(self.admit(8192, 4096, 4096, chunk=2048), 'the step carries 2048 of the 4096 tokens after R')

    def test_a_capture_that_does_not_record_the_route_is_an_engine_fault(self):
        case = self.admit(8192, 4096, 4096, records_route=False)
        self.assertIn('does not record the prefix-reuse route', str(case.error))
        self.assertTrue(case.lifecycle.failed)
        case.runner_execute.assert_not_called()

    def test_a_cold_prompt_under_the_flag_is_the_uncached_path_and_reads_no_registry(self):
        with sticky_env('1'):
            case = self.fixture(prompt=4096)
        with registry_holder(Tripwire(prefix_registry.REGISTRY_KEY)):
            self.assertIsNone(case.worker.execute_model(case.scheduled))
        case.capture_factory.assert_called_once_with(4096)
        self.assertTrue(case.lifecycle.prefill_pending)

    def test_a_hit_is_refused_as_always_with_the_flag_off(self):
        registry = prefix_registry.PrefixRegistry(budget_bytes=1 << 30)
        committed(registry, 'request', 4096, [1] * 8192)
        for value in (None, '0'):
            with self.subTest(flag=value):
                with sticky_env(value):
                    case = self.fixture(prompt=8192, computed=4096)
                with registry_holder(Tripwire(prefix_registry.REGISTRY_KEY)):
                    with self.assertRaisesRegex(ValueError, 'Text-only uncached prompt, whole or first chunk, '
                                                            'required: computed=4096 chunk=4096'):
                        case.worker.execute_model(case.scheduled)
                self.assertFalse(case.lifecycle.sticky)
                case.capture_factory.assert_not_called()

    def test_a_bad_flag_value_refuses_the_lifecycle(self):
        with sticky_env('yes'), self.assertRaisesRegex(ValueError, 'QWEN_FAST_STICKY_SESSIONS must be 0 or 1'):
            self.fixture()

    def test_an_abort_mid_prefill_releases_the_resumed_capture(self):
        case = self.admit(8192, 4096, 4096)
        self.assertIsNone(case.error)
        # vLLM aborted it after its prefill step, before its seed was sampled: the prefill side goes
        case.lifecycle.prefill_pending = False
        finished = SimpleNamespace(finished_req_ids={'request'}, scheduled_new_reqs=[],
                                   scheduled_cached_reqs=SimpleNamespace(req_ids=[]), scheduled_spec_decode_tokens={},
                                   num_scheduled_tokens={}, total_num_scheduled_tokens=0)
        case.worker.execute_model(finished)
        self.assertIsNone(case.lifecycle.request_id)
        self.assertIsNone(case.lifecycle.capture)
        case.capture.close.assert_called_once_with()
        self.assertIsNone(serving_lifecycle.prefill_gate().held)
        self.assertFalse(case.lifecycle.failed)


def shape_of(entry):
    """A recorded call as (name, argument types, keyword names): the objects differ between two runs."""
    name, args, kwargs = entry
    return name, tuple(type(value).__name__ for value in args), tuple(sorted(kwargs))


class LifecycleParityTests(LifecycleFixture):
    """With the flag off (unset or 0) today's lifecycle makes exactly the pre-sticky lifecycle's calls."""

    def scenario(self, cls, value, shape):
        with sticky_env(value):
            case = self.fixture(cls=cls, **shape['fixture'])
        trace = []
        registry = prefix_registry.PrefixRegistry(budget_bytes=1 << 30)
        committed(registry, 'request', 4096, [1] * 8192)
        with registry_holder(Tripwire(prefix_registry.REGISTRY_KEY)):
            for kind, argument in shape['steps']:
                try:
                    if kind == 'execute':
                        scheduled = case.scheduled if argument is None else argument(case)
                        result = case.worker.execute_model(scheduled)
                    else:
                        result = case.worker.sample_tokens(None)
                    trace.append(('ok', kind, type(result).__name__))
                except ValueError as error:
                    trace.append(('raise', kind, str(error)))
        trace.append(('execute', [shape_of(entry) for entry in case.runner_execute.mock_calls]))
        trace.append(('sample', [shape_of(entry) for entry in case.runner_sample.mock_calls]))
        trace.append(('capture_factory', case.capture_factory.mock_calls))
        trace.append(('capture', case.capture.capture.mock_calls, case.capture.segment.mock_calls,
                      case.capture.close.mock_calls))
        trace.append(('bridge', len(case.build.mock_calls)))
        trace.append(('state', case.lifecycle.request_id, list(case.lifecycle.decoding_ids),
                      case.lifecycle.prefill_pending, case.lifecycle.failed, case.lifecycle.chunk_in_flight))
        return trace

    SHAPES = (
        dict(fixture=dict(prompt=4096), steps=[('execute', None), ('sample', None)]),
        dict(fixture=dict(prompt=8192, computed=4096), steps=[('execute', None)]),
        dict(fixture=dict(prompt=8192, computed=4032), steps=[('execute', None)]),
        dict(fixture=dict(prompt=4096, chunk=4000), steps=[('execute', None)]),
        dict(fixture=dict(prompt=4096), steps=[
            ('execute', None), ('sample', None),
            ('execute', lambda case: SimpleNamespace(
                finished_req_ids={'request'}, scheduled_new_reqs=[], scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
                scheduled_spec_decode_tokens={}, num_scheduled_tokens={}, total_num_scheduled_tokens=0))]),
    )

    def test_every_shape_matches_the_parent_call_for_call(self):
        parent = parent_module('serving_lifecycle.py')
        if parent is None:
            self.skipTest('no git history for %s' % PARENT)
        for index, shape in enumerate(self.SHAPES):
            for value in (None, '0'):
                with self.subTest(shape=index, flag=value):
                    self.assertEqual(self.scenario(serving_lifecycle.FastServingLifecycle, value, shape),
                                     self.scenario(parent.FastServingLifecycle, value, shape))

    def test_the_sticky_helper_is_never_called_with_the_flag_off(self):
        for shape in self.SHAPES:
            with patch.object(serving_lifecycle.FastServingLifecycle, '_granted_resume',
                              side_effect=AssertionError('sticky admission reached with the flag off')), \
                    patch.object(serving_lifecycle, 'committed_grant',
                                 side_effect=AssertionError('registry read with the flag off')):
                self.scenario(serving_lifecycle.FastServingLifecycle, None, shape)


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


# ------------------------------------------------------------------------------------------------
# The runtime factories (A3 signature, A8 telemetry)
# ------------------------------------------------------------------------------------------------
class RuntimeFactoryTests(unittest.TestCase):
    def run_attach(self, environ):
        seen = {}

        def probe(install, diagnostic):
            capture_factory = install.call_args.kwargs['capture_factory']
            bridge_factory = install.call_args.kwargs['bridge_factory']
            serving_runtime.ServingCacheOwner.return_value.physical_pages = 100
            capture_factory(4096)
            seen['cold'] = built.call_args
            if environ.get(STICKY) == '1':
                capture_factory(8192, start=4096)
                seen['resumed'] = built.call_args
            else:
                with self.assertRaisesRegex(ValueError, 'A prefill resumed at 4096 needs QWEN_FAST_STICKY_SESSIONS=1'):
                    capture_factory(8192, start=4096)
            state = SimpleNamespace(req_id='request-1', block_ids=([3, 4],), num_computed_tokens=4096,
                                    prompt_token_ids=[1] * 8192)
            diagnostic.reset_mock()
            with patch.object(serving_runtime, 'from_prefill', return_value=SimpleNamespace(engine=object(), close=Mock())), \
                    patch.object(serving_runtime, 'VerifierPageBinding'), \
                    patch.object(serving_runtime, 'FastRunnerBridge', return_value='bridge'):
                bridge_factory(state, 'capture')
            seen['diag'] = [entry.args for entry in diagnostic.call_args_list]

        with patch('dflash_prefill_window.PrefillWindowCapture') as built:
            runtime_tests.RuntimeAttachmentTests().exercise(packed=True, users=4, four_as_two=False, probe=probe,
                                                            extra_env=environ)
        return seen

    def test_flag_off_builds_the_capture_it_always_built_and_logs_no_build_time(self):
        with sticky_env(None):
            seen = self.run_attach({})
        from dflash_request_runtime import TARGET_TAPS

        self.assertEqual(seen['cold'].args[2:], (4096, TARGET_TAPS))
        self.assertEqual(seen['cold'].kwargs, {}, 'exactly the pre-sticky call')
        self.assertFalse(any(serving_runtime.STICKY_ENGINE_MARKER in str(args[0]) for args in seen['diag']))

    def test_flag_on_resumes_the_capture_wraps_the_route_and_times_the_build(self):
        seen = self.run_attach(dict(ON))
        self.assertEqual(seen['cold'].kwargs, dict(start=0, prefix_route=True))
        self.assertEqual(seen['cold'].args[2], 4096)
        self.assertEqual(seen['resumed'].kwargs, dict(start=4096, prefix_route=True))
        self.assertEqual(seen['resumed'].args[2], 8192)
        marks = [args for args in seen['diag'] if args[0].startswith(serving_runtime.STICKY_ENGINE_MARKER)]
        self.assertEqual(len(marks), 1, seen['diag'])
        template, request, ms, frontier, prompt = marks[0]
        self.assertEqual(template, serving_runtime.STICKY_ENGINE_MARKER + '{} ms={:.1f} frontier={} prompt={}')
        self.assertEqual((request, frontier, prompt), ('request-1', 4096, 8192))
        self.assertGreaterEqual(ms, 0.0)
        # the line parses as a gate would read it
        line = template.format(request, ms, frontier, prompt)
        self.assertRegex(line, r'^\[PINDIAG\] sticky engine built req=request-1 ms=\d+\.\d frontier=4096 prompt=8192$')


# ------------------------------------------------------------------------------------------------
# The policy and the contract (A5)
# ------------------------------------------------------------------------------------------------
class PolicyTests(unittest.TestCase):
    def config(self, caching, algorithm='sha256'):
        config = policy_tests.FastPolicyTests().fixture()
        config.cache_config.enable_prefix_caching = caching
        config.cache_config.prefix_caching_hash_algo = algorithm
        return config

    def test_the_prefix_cache_is_admitted_only_with_both_switches_and_sha256(self):
        both = {STICKY: '1', policy.PREFIX_REUSE_FLAG: '1'}
        with patch.dict(os.environ, both):
            self.assertEqual(policy.validate_fast_config(self.config(True))['scheduler_requests'], 1)
        refused = ({STICKY: '1'}, {policy.PREFIX_REUSE_FLAG: '1'}, {STICKY: '0', policy.PREFIX_REUSE_FLAG: '1'}, {})
        for environ in refused:
            with self.subTest(environ=environ), sticky_env(None), patch.dict(os.environ, environ), \
                    self.assertRaisesRegex(ValueError, 'no prefix cache or LoRA'):
                os.environ.pop(policy.PREFIX_REUSE_FLAG, None) if policy.PREFIX_REUSE_FLAG not in environ else None
                policy.validate_fast_config(self.config(True))
        with patch.dict(os.environ, both), self.assertRaisesRegex(ValueError, 'no prefix cache or LoRA'):
            policy.validate_fast_config(self.config(True, 'xxhash'))
        with patch.dict(os.environ, both), self.assertRaisesRegex(ValueError, 'no prefix cache or LoRA'):
            policy.validate_fast_config(self.config('yes'))

    def test_the_flag_is_read_strictly(self):
        self.assertEqual(policy.STICKY_SESSIONS_FLAG, STICKY)
        self.assertFalse(policy.sticky_sessions_enabled({}))
        self.assertFalse(policy.sticky_sessions_enabled({STICKY: '0'}))
        self.assertTrue(policy.sticky_sessions_enabled({STICKY: '1'}))
        for value in ('true', '', 'on', '2'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'must be 0 or 1'):
                policy.sticky_sessions_enabled({STICKY: value})

    def test_flag_off_the_prefix_cache_is_refused_without_reading_the_switches(self):
        with sticky_env(None), patch.object(policy, 'prefix_cache_admitted',
                                            side_effect=AssertionError('consulted with the cache off')):
            policy.validate_fast_config(self.config(False))


if __name__ == '__main__':
    unittest.main()
