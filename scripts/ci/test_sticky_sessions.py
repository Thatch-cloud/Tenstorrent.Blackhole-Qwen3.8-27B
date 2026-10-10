"""Sticky sessions, phase 1 (QWEN_FAST_STICKY_SESSIONS): the CPU tests of every path.

Phase 1 retains nothing of a finished request's engine, slot or end-of-turn state (none of it is what a
cold prefill writes); it resumes a continuation exactly from the last 2048-token boundary at or below
its prompt minus 2048 that the prefix-reuse grafts kept: vLLM's KV prefix (published below floor2048(P))
and a host GDN checkpoint (planned at floor2048(P) - 2048, because vLLM drops a DFlash hit's last
block). What is held here:

  flag off   serving_lifecycle, dflash_prefill_window, serving_packed_step.kv_guard and the scheduler
             graft make exactly the calls and decisions of their pre-sticky versions (the merge commit's,
             loaded from git; without that history a skip locally, a failure in CI), the registry is never
             read, the runtime builds the capture it always built, the policy refuses the prefix cache, and
             the contract refuses the fast path beside prefix reuse;
  warmup     the fast path's worker (the P8 patch's compile_or_warm_up_model, which returns before the
             plugin's warmup) runs the G1 model graft's warm under QWEN_PREFIX_REUSE=1 before start(): the
             restore path is chosen and the mid-loop captures declared, so the first resume runs and the
             first long request after boot plans C0; a model without the graft or a skipped warm is refused;
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

import make_octo_profiles
import make_parked_profiles
import profile_twins
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


def no_history(case):
    """A flag-off proof without the pre-sticky tree (PARENT, from git): a skip locally, a FAILURE in CI
    (GITHUB_ACTIONS=true), so a rebase or squash that loses PARENT cannot turn the call-for-call equivalence
    into green skips. Re-pin PARENT to the new pre-sticky commit then."""
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        case.fail('no git history for %s: the flag-off equivalence did not run (re-pin PARENT after a rebase or '
                  'squash)' % PARENT)
    case.skipTest('no git history for %s' % PARENT)


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
        # The lifecycle parks its prefill gate (held while a prefill is pending) under a fixed sys.modules key;
        # drop it after each test, or a later suite in the same process inherits a held gate.
        self.addCleanup(sys.modules.pop, serving_lifecycle.PREFILL_GATE_KEY, None)
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
            no_history(self)
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
            no_history(self)
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


def load_profiles():
    import json

    with open(HERE / 'qwen_c2_profiles.json', encoding='utf-8') as handle:
        return json.load(handle)


def profile(name):
    data = load_profiles()['profiles'][name]
    return dict(data, name=name)


class ProfileTests(unittest.TestCase):
    PAIRS = (('c2-packed-prefix', 'c2-packed'), ('c2-packed-prefix-gate', 'c2-packed-gate'),
             # tp4/packed-prefix: the eight-seat 262k twins (every other delta is the parent's: the levers, the pool)
             ('c2-packed-tp4-8x262k-prefix-gate', 'c2-packed-tp4-8x262k-best'),
             ('c2-packed-tp4-8x262k-prefix-time-gate', 'c2-packed-tp4-8x262k-best-time-gate'))

    def test_each_sticky_profile_is_its_twin_with_the_prefix_deltas_only(self):
        for name, twin in self.PAIRS:
            with self.subTest(profile=name):
                mine, theirs = profile(name), profile(twin)
                self.assertEqual(mine['env'], dict(theirs['env'], QWEN_PREFIX_REUSE='1', QWEN_PREFIX_STORE_GIB='8',
                                                   QWEN_FAST_STICKY_SESSIONS='1'))
                engine = dict(theirs['engine'])
                self.assertIs(engine.pop('no-enable-prefix-caching'), True)
                self.assertIs(engine.pop('no-enable-chunked-prefill'), True)
                engine.update({'enable-prefix-caching': True, 'enable-chunked-prefill': True,
                               'prefix-caching-hash-algo': 'sha256'})
                self.assertEqual(mine['engine'], engine)
                for key in set(theirs) | set(mine):
                    if key not in ('env', 'engine', 'description', 'name'):
                        self.assertEqual(mine.get(key), theirs.get(key), key)
                self.assertEqual(contract.prefix_reuse_problems(mine), [])
                self.assertTrue(contract.prefix_reuse(mine) and contract.sticky_sessions(mine))
        self.assertEqual(load_profiles()['default'], 'c2-packed-tp4', 'serving/tp4-s2: the image default is the four-card traffic profile')

    def test_only_the_sticky_profiles_set_the_switch(self):
        names = sorted(name for name, data in load_profiles()['profiles'].items() if STICKY in data['env'])
        names = [name for name in names if name not in profile_twins.twin_names()]
        self.assertEqual(names, ['c2-packed-prefix', 'c2-packed-prefix-gate', 'c2-packed-tp4-8x262k-prefix-gate',
                                 'c2-packed-tp4-8x262k-prefix-time-gate',
                                 'c2-packed-tp4-8x262k-ship-prefix', 'c2-packed-tp4-8x262k-ship-prefix-audit', 'c2-packed-tp4-8x262k-ship-prefix-audit-digests', 'c2-packed-tp4-8x262k-ship-prefix-dbf16', 'c2-packed-tp4-8x262k-ship-prefix-dckdefault', 'c2-packed-tp4-8x262k-ship-prefix-levern', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-nolna-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-audit-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-traffic', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-f1', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-ln', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-lean-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-nolna-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-pool-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-sdpa', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-audit-w1', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-er-traffic', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-lean', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-nolna-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-levern-w2-nof1-epochglobal', 'c2-packed-tp4-8x262k-ship-prefix-pool', 'c2-packed-tp4-8x262k-ship-prefix-w2', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-w2-audit-pool', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit-lean', 'c2-packed-tp4-8x262k-ship-prefix-w2-nof1-audit-pool'])

    def test_the_argv_turns_prefix_caching_on_beside_dflash(self):
        argv = contract.engine_arguments(profile('c2-packed-prefix'), '/snap')
        for flag in ('--enable-prefix-caching', '--enable-chunked-prefill', '--no-async-scheduling',
                     '--speculative-config'):
            self.assertIn(flag, argv)
        for flag in ('--no-enable-prefix-caching', '--no-enable-chunked-prefill'):
            self.assertNotIn(flag, argv)
        self.assertEqual(argv[argv.index('--prefix-caching-hash-algo') + 1], 'sha256')

    def test_the_contract_refuses_every_half_configuration(self):
        base = profile('c2-packed-prefix')

        def broken(env=None, engine=None, drop_env=(), drop_engine=()):
            data = dict(base, env=dict(base['env']), engine=dict(base['engine']))
            data['env'].update(env or {})
            data['engine'].update(engine or {})
            for key in drop_env:
                data['env'].pop(key)
            for key in drop_engine:
                data['engine'].pop(key)
            return contract.prefix_reuse_problems(data)

        cases = (
            (broken(drop_env=[STICKY]), 'cannot admit a prefix hit without QWEN_FAST_STICKY_SESSIONS=1'),
            (broken(drop_env=[STICKY]), 'refuses lookahead'),
            (broken(env={STICKY: 'yes'}), 'QWEN_FAST_STICKY_SESSIONS=\'yes\' is neither 1 nor 0'),
            (broken(env={'QWEN_PREFIX_REUSE': '0'}), 'needs QWEN_PREFIX_REUSE=1'),
            (broken(drop_env=['QWEN_PREFIX_REUSE']), 'needs QWEN_PREFIX_REUSE=1'),
            (broken(engine={'additional-config': {'tt': {}}}), 'does not run the fast path'),
            (broken(engine={'speculative-config': dict(base['engine']['speculative-config'], num_speculative_tokens=7)}),
             'sticky sessions serve only dflash with 15 proposals'),
            (broken(engine={'speculative-config': dict(base['engine']['speculative-config'], method='eagle')}),
             'sticky sessions serve only dflash with 15 proposals'),
            (broken(engine={'prefix-caching-hash-algo': 'xxhash'}), 'prefix-caching-hash-algo must be sha256'),
        )
        for problems, words in cases:
            with self.subTest(words=words):
                self.assertTrue(any(words in problem for problem in problems), problems)
        # sticky beside a profile with no prefix reuse at all (c2-packed plus the switch)
        packed = profile('c2-packed')
        packed['env'] = dict(packed['env'], QWEN_FAST_STICKY_SESSIONS='1')
        self.assertTrue(any('needs QWEN_PREFIX_REUSE=1' in problem for problem in contract.prefix_reuse_problems(packed)))

    def test_the_environment_drops_an_inherited_switch_under_every_other_profile(self):
        for name in ('c2-packed', 'general-prefix', 'exact'):
            with self.subTest(profile=name):
                environ = contract.apply_environment(profile(name), {STICKY: '1', 'QWEN_PREFIX_REUSE': '1'})
                self.assertNotIn(STICKY, environ)
        environ = contract.apply_environment(profile('c2-packed-prefix'), {})
        self.assertEqual((environ[STICKY], environ['QWEN_PREFIX_REUSE']), ('1', '1'))

    def test_the_flag_off_contract_matches_the_parent_on_every_other_profile(self):
        parent = parent_module('serving_c2_contract.py')
        if parent is None:
            no_history(self)
        for name in sorted(load_profiles()['profiles']):
            if name.startswith('c2-packed-prefix') or STICKY in load_profiles()['profiles'][name]['env']:
                continue            # the flag-on twins (tp4/packed-prefix's included) are what the flag changes
            with self.subTest(profile=name):
                self.assertEqual(contract.prefix_reuse_problems(profile(name)), parent.prefix_reuse_problems(profile(name)))
                self.assertEqual(contract.engine_arguments(profile(name), '/s'), parent.engine_arguments(profile(name), '/s'))


# ------------------------------------------------------------------------------------------------
# The scheduler graft (A4) on the fakes of test_qwen_prefix_scheduler_patch, with vLLM's DFlash drop
# ------------------------------------------------------------------------------------------------
class EagleManager(graft_fakes.Manager):
    """vLLM's KV cache manager with use_eagle set (dflash): the unitary coordinator drops a hit's last
    matched block (single_type_kv_cache_manager.find_longest_cache_hit, drop_eagle_block)."""

    def __init__(self, num_blocks):
        super(EagleManager, self).__init__(num_blocks)
        self.coordinator.eagle_group_ids = {0}

    def get_computed_blocks(self, request):
        blocks, _ = super(EagleManager, self).get_computed_blocks(request)
        hits = list(blocks.blocks[0])
        if hits:
            hits.pop()
        return graft_fakes.Blocks((hits,)), len(hits) * BLOCK


def dflash_scheduler(num_blocks=600, max_model_len=65536, lookahead=16, method='dflash', proposals=15, eagle=True):
    scheduler = graft_fakes.FakeScheduler(num_blocks=num_blocks, max_model_len=max_model_len)
    if eagle:
        scheduler.kv_cache_manager = EagleManager(num_blocks)
    scheduler.num_lookahead_tokens = lookahead
    scheduler.vllm_config.speculative_config = SimpleNamespace(method=method, num_speculative_tokens=proposals)
    return scheduler


def tokens(count, seed):
    return graft_fakes.tokens(count, seed)


class SchedulerGraftTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, graft_fakes.fake_vllm_modules())
        patcher.start()
        self.addCleanup(patcher.stop)
        environment = mock.patch.dict(os.environ, {'QWEN_PREFIX_STATS_PATH': ''})
        environment.start()
        self.addCleanup(environment.stop)
        self.logs = []

    def logger(self, message, *values):
        self.logs.append(message % values if values else message)

    def make(self, registry=None, environ=ON, **kwargs):
        scheduler = dflash_scheduler(**kwargs)
        registry = registry or prefix_registry.PrefixRegistry(budget_bytes=1 << 40)
        with graft_fakes.Quiet():
            registry.enable_mid_loop_capture()   # the model graft declares it at warmup
        installed = graft.install(scheduler, registry=registry, kill_switch_path=None, logger=self.logger,
                                  stats=graft.StatsExport(path='', logger=self.logger), environ=environ)
        self.model = graft_fakes.FakeGdnModel(registry)
        return scheduler, installed

    def step(self, scheduler):
        out = scheduler.schedule()
        registry = scheduler._qwen_prefix.registry
        rows = {}
        for data in out.scheduled_new_reqs:
            grant = registry.grant_for(data.req_id)
            rows[data.req_id] = (data.num_computed_tokens, grant)
            with graft_fakes.Quiet():
                self.model.prefill(data.req_id, scheduler.requests[data.req_id].all_token_ids, data.num_computed_tokens)
            # the lifecycle's own rule, on every admission the graft let through
            prompt = scheduler.requests[data.req_id].num_prompt_tokens
            if data.num_computed_tokens:
                self.assertEqual(data.num_computed_tokens % CHUNK, 0)
                self.assertLessEqual(data.num_computed_tokens, prompt - CHUNK, 'a resume inside the draft window')
        return rows

    def serve(self, scheduler, request):
        scheduler.add(request)
        rows = self.step(scheduler)
        scheduler.finish(request)
        return rows[request.request_id]

    # -- install ----------------------------------------------------------------------------------
    def test_the_dflash_lookahead_is_refused_without_the_flag(self):
        for environ in ({}, {STICKY: '0'}):
            with self.subTest(environ=environ), \
                    self.assertRaisesRegex(graft.PrefixInstallError, 'speculative lookahead is on$'):
                self.make(environ=environ)

    def test_only_dflash_with_fifteen_proposals_is_accepted_under_the_flag(self):
        for kwargs in (dict(lookahead=15), dict(lookahead=5, proposals=4), dict(method='eagle'),
                       dict(proposals=7)):
            with self.subTest(kwargs=kwargs), \
                    self.assertRaisesRegex(graft.PrefixInstallError, 'sticky sessions accept only dflash with 15'):
                self.make(**kwargs)
        scheduler, installed = self.make()
        self.assertTrue(installed.sticky and installed.drop_last)
        self.assertIn('install sticky=1 lookahead=16 drop_last=True ceiling=floor2048(P-2048)', self.logs)

    def test_the_flag_without_lookahead_still_installs_as_g1(self):
        scheduler, installed = self.make(lookahead=0, eagle=False)
        self.assertTrue(installed.sticky)
        self.assertFalse(installed.drop_last)

    # -- retain and extend --------------------------------------------------------------------------
    def test_a_turn_retains_its_prefix_and_the_checkpoint_the_drop_makes_reachable(self):
        scheduler, installed = self.make()
        registry = installed.registry
        turn = tokens(9000, 1)
        start, grant = self.serve(scheduler, graft_fakes.Request('n', turn))
        self.assertEqual((start, grant.q, grant.capture_positions()), (0, 0, [6144]))
        self.assertEqual(scheduler.kv_cache_manager.coordinator.calls[-1], ('n', 8192), 'published up to floor2048(P)')
        self.assertIsNotNone(registry.get(graft_fakes.Request('probe', turn).block_hashes[6144 // BLOCK - 1]))
        self.assertIsNone(registry.get(graft_fakes.Request('probe', turn).block_hashes[8192 // BLOCK - 1]),
                          'nothing at floor2048(P): no later hit could be granted there')

    def test_the_next_turn_resumes_at_the_checkpoint_and_plans_its_own(self):
        """The metered shape, scaled: previous prompt + a short answer + a new message."""
        scheduler, installed = self.make()
        turn = tokens(9000, 2)
        self.serve(scheduler, graft_fakes.Request('n', turn))
        extended = turn + tokens(115, 3) + tokens(3500, 4)
        start, grant = self.serve(scheduler, graft_fakes.Request('n1', extended))
        self.assertEqual((grant.h, grant.q, start), (8192 - BLOCK, 6144, 6144))
        self.assertEqual(grant.capture_positions(), [floor(len(extended)) - CHUNK])
        self.assertEqual(self.model.restored['n1'], 6144)
        self.assertEqual(self.model.chunks['n1'], (floor(len(extended)) - 6144) // CHUNK)
        # and the turn after that, from the checkpoint this one took
        again = extended + tokens(700, 5) + tokens(2000, 6)
        start, grant = self.serve(scheduler, graft_fakes.Request('n2', again))
        self.assertEqual((grant.q, start), (floor(len(extended)) - CHUNK, floor(len(extended)) - CHUNK))

    def test_a_small_extension_resumes_and_plans_nothing(self):
        scheduler, installed = self.make()
        turn = tokens(9000, 7)
        self.serve(scheduler, graft_fakes.Request('n', turn))
        start, grant = self.serve(scheduler, graft_fakes.Request('n1', turn + tokens(500, 8)))
        self.assertEqual((grant.q, start, grant.capture_positions()), (6144, 6144, []))

    def test_a_sibling_gets_the_gap_boundary(self):
        scheduler, installed = self.make()
        system = tokens(5000, 9)
        start, grant = self.serve(scheduler, graft_fakes.Request('a', system + tokens(5000, 10)))
        self.assertEqual(grant.capture_positions(), [6144], 'a checkpoints past the shared block only')
        start, grant = self.serve(scheduler, graft_fakes.Request('b', system + tokens(3500, 11)))
        # b shares 4992 tokens (78 blocks) with a, less the dropped block; no checkpoint below them yet,
        # so b plans the gap boundary floor2048(h) beside its own C0
        self.assertEqual((grant.h, grant.q), (4992 - BLOCK, 0))
        self.assertEqual(grant.capture_positions(), [4096, 8192 - CHUNK])
        start, grant = self.serve(scheduler, graft_fakes.Request('c', system + tokens(2500, 12)))
        self.assertEqual((grant.q, start), (4096, 4096))

    def test_a_short_fork_never_resumes_inside_its_draft_window(self):
        """A prompt that is a prefix of a longer cached conversation: vLLM's hit can reach past P - 2048,
        where a checkpoint exists (the longer turn took it); the ceiling keeps Q at or below
        floor2048(P - 2048), here with no checkpoint there, so the fork is cold."""
        scheduler, installed = self.make()
        long_turn = tokens(15000, 13)
        self.serve(scheduler, graft_fakes.Request('long', long_turn))
        self.assertIsNotNone(installed.registry.get(graft_fakes.Request('p', long_turn).block_hashes[12288 // BLOCK - 1]))
        start, grant = self.serve(scheduler, graft_fakes.Request('fork', long_turn[:12500]))
        self.assertGreater(floor(grant.h), floor(12500 - CHUNK), 'the hit itself reaches past the draft window')
        self.assertEqual((grant.q, start), (0, 0))

    # -- mismatch -> cold ---------------------------------------------------------------------------
    def test_a_divergence_below_the_checkpoint_is_cold(self):
        scheduler, installed = self.make()
        turn = tokens(9000, 14)
        self.serve(scheduler, graft_fakes.Request('n', turn))
        edited = list(turn)
        edited[100] += 1   # the template rewrote history early
        start, grant = self.serve(scheduler, graft_fakes.Request('n1', edited + tokens(3000, 15)))
        # block 0 still matches, and is the block vLLM drops: no hit at all
        self.assertEqual((grant.h, grant.q, start), (0, 0, 0))

    def test_another_salt_or_no_salt_never_hits(self):
        scheduler, installed = self.make()
        turn = tokens(9000, 16)
        self.serve(scheduler, graft_fakes.Request('n', turn))
        start, grant = self.serve(scheduler, graft_fakes.Request('other', turn + tokens(3000, 17), salt='tenant-b'))
        self.assertEqual((grant.h, grant.q, start), (0, 0, 0))
        start, grant = self.serve(scheduler, graft_fakes.Request('none', turn + tokens(3000, 18), salt=None))
        self.assertEqual((start, grant), (0, None))
        start, grant = self.serve(scheduler, graft_fakes.Request('same', turn + tokens(3000, 19)))
        self.assertEqual((grant.q, start), (6144, 6144), 'the owning salt still resumes')

    # -- eviction -------------------------------------------------------------------------------------
    def test_slot_demand_evicts_the_prefix_and_its_checkpoint_with_it(self):
        scheduler, installed = self.make(num_blocks=300)
        registry = installed.registry
        turn = tokens(9000, 20)
        self.serve(scheduler, graft_fakes.Request('n', turn))
        self.assertEqual(len(registry.entries), 1)
        # other conversations fill the pool: vLLM's LRU takes the free-but-cached prefix blocks
        for index in range(3):
            self.serve(scheduler, graft_fakes.Request('flood-%d' % index, tokens(9000, 30 + index), salt='tenant-%d' % index))
        self.assertGreater(registry.stats['evicted_coupled'], 0)
        start, grant = self.serve(scheduler, graft_fakes.Request('n1', turn + tokens(3000, 21)))
        self.assertEqual((grant.q, start), (0, 0), 'the continuation is cold, and exact')

    def test_host_store_pressure_evicts_the_oldest_checkpoint(self):
        registry = prefix_registry.PrefixRegistry(budget_bytes=1)   # one checkpoint's worth (nbytes=1 each)
        scheduler, installed = self.make(registry=registry)
        first, second = tokens(9000, 22), tokens(9000, 23)
        self.serve(scheduler, graft_fakes.Request('a', first))
        self.serve(scheduler, graft_fakes.Request('b', second, salt='tenant-b'))
        self.assertEqual(registry.stats['evicted_lru'], 1)
        start, grant = self.serve(scheduler, graft_fakes.Request('a1', first + tokens(3000, 24)))
        self.assertEqual((grant.h, grant.q, start), (8192 - BLOCK, 0, 0), 'KV kept, checkpoint gone: cold')
        self.assertEqual(registry.stats['kv_hit_without_checkpoint'], 1)
        start, grant = self.serve(scheduler, graft_fakes.Request('b1', second + tokens(3000, 25), salt='tenant-b'))
        self.assertEqual(start, 0, 'b\'s checkpoint went when a1 captured its own')

    def test_the_kill_switch_sends_every_continuation_cold(self):
        scheduler, installed = self.make()
        turn = tokens(9000, 26)
        self.serve(scheduler, graft_fakes.Request('n', turn))
        installed.kill_switch.engaged = True
        installed.registry.disable('kill switch (test)')
        start, grant = self.serve(scheduler, graft_fakes.Request('n1', turn + tokens(3000, 27)))
        self.assertEqual((start, grant), (0, None))

    def test_nothing_ages_out_on_the_engine_side(self):
        """The node's session cap and idle TTL are upstream (design 1); retention here is bounded by vLLM's
        pool and the host store alone, so a continuation long after its turn still resumes."""
        clock = [0.0]
        scheduler, installed = self.make()
        installed.kill_switch.clock = lambda: clock[0]
        turn = tokens(9000, 28)
        self.serve(scheduler, graft_fakes.Request('n', turn))
        clock[0] += 3600.0
        start, grant = self.serve(scheduler, graft_fakes.Request('n1', turn + tokens(3000, 29)))
        self.assertEqual(start, 6144)

    # -- abort --------------------------------------------------------------------------------------
    def test_an_aborted_granted_request_releases_its_grant_and_pin(self):
        scheduler, installed = self.make()
        registry = installed.registry
        turn = tokens(9000, 40)
        self.serve(scheduler, graft_fakes.Request('n', turn))
        request = graft_fakes.Request('n1', turn + tokens(3000, 41))
        scheduler.add(request)
        rows = self.step(scheduler)
        self.assertEqual(rows['n1'][0], 6144)
        self.assertEqual(registry.pins(), 1)
        scheduler.finish(request)   # aborted: vLLM frees it through _free_request
        self.assertIsNone(registry.grant_for('n1'))
        self.assertEqual(registry.pins(), 0)
        # the prefill ran before the abort: its own C0 (8192) was captured and outlives it
        start, grant = self.serve(scheduler, graft_fakes.Request('n2', turn + tokens(3000, 41)))
        self.assertEqual(start, 8192)

    def test_a_waiting_request_aborted_before_admission_leaves_nothing_staged(self):
        scheduler, installed = self.make(num_blocks=160)
        registry = installed.registry
        turn = tokens(9000, 42)
        self.serve(scheduler, graft_fakes.Request('n', turn))
        blocker = graft_fakes.Request('big', tokens(9000, 43), salt='tenant-z')
        scheduler.add(blocker)
        self.step(scheduler)
        waiting = graft_fakes.Request('n1', turn + tokens(3000, 44))
        scheduler.add(waiting)
        scheduler.schedule()     # no room: the grant is staged and dropped, never committed
        self.assertIsNone(registry.grant_for('n1'))
        scheduler.waiting.remove(waiting)
        scheduler._free_request(waiting)
        self.assertEqual(registry.pins(), 0)
        self.assertNotIn('n1', registry.staged)


def floor(value):
    return prefix_registry.floor_chunk(value)


class SchedulerParityTests(unittest.TestCase):
    """With the flag off the graft decides what the pre-sticky graft decided, step for step."""

    def setUp(self):
        patcher = mock.patch.dict(sys.modules, graft_fakes.fake_vllm_modules())
        patcher.start()
        self.addCleanup(patcher.stop)
        environment = mock.patch.dict(os.environ, {'QWEN_PREFIX_STATS_PATH': ''})
        environment.start()
        self.addCleanup(environment.stop)

    def scenario(self, module):
        scheduler = graft_fakes.FakeScheduler(num_blocks=300)
        registry = prefix_registry.PrefixRegistry(budget_bytes=1 << 40)
        with graft_fakes.Quiet():
            registry.enable_mid_loop_capture()
        logs = []
        module.install(scheduler, registry=registry, kill_switch_path=None,
                       logger=lambda message, *values: logs.append(message % values if values else message),
                       stats=module.StatsExport(path='', logger=lambda *a: None))
        model = graft_fakes.FakeGdnModel(registry)
        trace = []
        text = tokens(9000, 50)
        requests = [graft_fakes.Request('a', text), graft_fakes.Request('b', text + tokens(3000, 51)),
                    graft_fakes.Request('c', text[:5000] + tokens(4000, 52)), graft_fakes.Request('d', tokens(15000, 53)),
                    graft_fakes.Request('e', tokens(15000, 53)[:12500]), graft_fakes.Request('f', text, salt=None)]
        for request in requests:
            scheduler.add(request)
            out = scheduler.schedule()
            for data in out.scheduled_new_reqs:
                grant = registry.grant_for(data.req_id)
                trace.append((data.req_id, data.num_computed_tokens, grant.describe() if grant else None))
                with graft_fakes.Quiet():
                    model.prefill(data.req_id, scheduler.requests[data.req_id].all_token_ids, data.num_computed_tokens)
            scheduler.finish(request)
        # The counters the parent kept: the store policies' and the telemetry's are fed by this graft's trim and not by the parent's.
        stats = {name: registry.stats[name] for name in prefix_registry.LEGACY_STAT_NAMES}
        for name in ('capture_ms', 'restore_ms'):
            stats.pop(name)
        return trace, stats, [line for line in logs if not line.startswith('install ')]

    def test_the_flag_off_graft_matches_the_parent(self):
        parent = parent_module('qwen_prefix_scheduler_patch.py')
        if parent is None:
            no_history(self)
        for value in (None, '0'):
            with self.subTest(flag=value), sticky_env(value):
                self.assertEqual(self.scenario(graft), self.scenario(parent))

    def test_the_flag_off_install_rules_match_the_parent(self):
        parent = parent_module('qwen_prefix_scheduler_patch.py')
        if parent is None:
            no_history(self)
        cases = [dflash_scheduler(), dflash_scheduler(lookahead=5), graft_fakes.FakeScheduler()]
        broken = graft_fakes.FakeScheduler()
        broken.scheduler_config.async_scheduling = True
        broken.cache_config.prefix_caching_hash_algo = 'xxhash'
        cases.append(broken)
        for value in (None, '0'):
            with sticky_env(value):
                for index, scheduler in enumerate(cases):
                    with self.subTest(flag=value, case=index):
                        self.assertEqual(graft.install_problems(scheduler), parent.install_problems(scheduler))


# ------------------------------------------------------------------------------------------------
# The K/V guard (A6)
# ------------------------------------------------------------------------------------------------
class FakeBlock(object):
    def __init__(self, engines, rows=16):
        self.engines = list(engines)
        self.shape = SimpleNamespace(users=len(self.engines), rows_per_user=rows)

    def segment_of(self, engine):
        for index, candidate in enumerate(self.engines):
            if candidate is engine:
                return index
        raise ValueError('engine not bound')


def engine_with(blocks, width=68):
    pages = torch.full((1, width), blocks[0], dtype=torch.int32)
    pages[0, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    return SimpleNamespace(pages=pages)


class KvGuardTests(unittest.TestCase):
    def owners(self, shared_first=True):
        # A and B: same tenant, same cached first block 7 (their shared prefix); C and D: others.
        a = engine_with([7, 10, 11])
        b = engine_with([7 if shared_first else 8, 20, 21])
        c = engine_with([30, 31, 32, 33])
        d = engine_with([40, 41, 42, 43])
        return a, b, c, d

    def test_same_tenant_users_crossing_a_boundary_are_not_a_conflict(self):
        a, b, c, d = self.owners()
        block = FakeBlock([a, b, c, d])
        # both at 184: rows 184..199, the last eight past their bound blocks (192 // 64 = 3)
        owners = [(a, 184), (b, 184), (c, 100), (d, 100)]
        with sticky_env(None):
            self.assertEqual(serving_packed_step.kv_guard(owners, block),
                             'verify t2 kv tile rows shared: users 0,1 page 7 tile row 0')
        with sticky_env('1'):
            self.assertIsNone(serving_packed_step.kv_guard(owners, block))
            with patch.object(verify_trace_t2, 'log_once') as logged:
                self.assertIsNone(serving_packed_step.kv_shared_at_proposal(
                    [SimpleNamespace(engine=engine, session=SimpleNamespace(position=position))
                     for engine, position in owners], block))
            logged.assert_not_called()

    def test_a_real_conflict_is_still_caught(self):
        a, b, c, d = self.owners()
        b.pages[0, 1] = 10     # B's table names A's private block
        block = FakeBlock([a, b, c, d])
        with sticky_env('1'):
            self.assertEqual(serving_packed_step.kv_guard([(a, 70), (b, 70), (c, 100), (d, 100)], block),
                             'verify t2 kv tile rows shared: users 0,1 page 10 tile row 0')
        # and the shared first block itself, written at its own rows by both (never on a real path)
        with sticky_env('1'):
            self.assertEqual(serving_packed_step.kv_guard([(a, 0), (b, 0), (c, 100), (d, 100)], block),
                             'verify t2 kv tile rows shared: users 0,1 page 7 tile row 0')

    def test_an_unmappable_table_still_fails_closed(self):
        a, b, c, d = self.owners()
        b.pages = None
        with sticky_env('1'):
            self.assertIn('verify t2 kv tile rows unmapped',
                          serving_packed_step.kv_guard([(a, 70), (b, 70), (c, 100), (d, 100)], FakeBlock([a, b, c, d])))

    def test_the_sentinel_table(self):
        self.assertEqual(serving_packed_step.pad_sentinel_table(torch.tensor([[7, 10, 11, 7, 7]]), 2),
                         [[7, 10, 11, -3, -3]])
        self.assertEqual(serving_packed_step.pad_sentinel_table([[5, 6]], 0), [[5, 6]])
        self.assertIsNone(serving_packed_step.pad_sentinel_table(None, 0))

    def test_flag_off_the_guard_is_the_parent_guard(self):
        parent = parent_module('serving_packed_step.py')
        if parent is None:
            no_history(self)
        a, b, c, d = self.owners()
        e, f, g, h = self.owners(shared_first=False)
        cases = [([(a, 184), (b, 184), (c, 100), (d, 100)], FakeBlock([a, b, c, d])),
                 ([(e, 184), (f, 184), (g, 100), (h, 100)], FakeBlock([e, f, g, h])),
                 ([(a, 0), (b, 0), (c, 0), (d, 0)], FakeBlock([a, b, c, d]))]
        with sticky_env(None), patch.object(serving_packed_step, 'pad_sentinel_table',
                                            side_effect=AssertionError('mapped with the flag off')):
            for owners, block in cases:
                self.assertEqual(serving_packed_step.kv_guard(owners, block), parent.kv_guard(owners, block))


# ------------------------------------------------------------------------------------------------
# The writer audit (A7): every device write of a resumed request lands in its own blocks at or above R
# ------------------------------------------------------------------------------------------------
class WriterAuditTests(unittest.TestCase):
    """docs/sticky-sessions-writer-audit.md, as properties of the page tables the fast path builds."""

    R, P, BUDGET = 6144, 12700, 1024

    def request_blocks(self):
        shared = list(range(100, 100 + self.R // BLOCK))                 # vLLM's cached prefix [0, R)
        private = list(range(500, 500 + (self.P - self.R + BLOCK - 1) // BLOCK))
        return shared, private

    # The prefill route's own writes are held on the real staged route, not here:
    # ToyExactnessTests.test_the_resumed_route_writes_only_its_own_blocks_from_r.

    def test_the_engine_table_is_this_request_s_allocation_and_its_pad_is_its_first_block(self):
        shared, private = self.request_blocks()
        blocks = shared + private + list(range(900, 900 + self.BUDGET // BLOCK))
        pages = torch.full((1, 400), blocks[0], dtype=torch.int32)
        pages[0, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        serving_page_binding.validate_initial_capture_pages(pages, blocks, position=self.P, output_budget=self.BUDGET)
        self.assertEqual(serving_packed_step.pad_sentinel_table(pages, 0)[0][len(blocks)], -1)

    def binding(self, blocks):
        pages = torch.full((1, 400), blocks[0], dtype=torch.int32)     # as serving_runtime's bridge builds it
        pages[0, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        binding = object.__new__(serving_page_binding.VerifierPageBinding)
        binding.engine = SimpleNamespace(phase='idle', pages=pages)
        binding.operations, binding.mesh = Mock(), object()
        binding.physical_pages, binding.failed, binding.capacity = 4096, False, 400
        binding.blocks, binding.bindings = tuple(blocks), {}
        return binding

    def test_verify_rows_are_written_only_after_the_binding_covers_them(self):
        shared, private = self.request_blocks()
        blocks = shared + private
        binding = self.binding(blocks)
        for position in range(self.P, self.P + 256, 16):
            needed = (position + 16 + BLOCK - 1) // BLOCK
            if needed > len(blocks):
                with self.assertRaisesRegex(ValueError, 'must cover every scheduled verifier row'):
                    binding.refresh(blocks, position=position, rows=16)
                blocks = blocks + [1000 + len(blocks)]          # vLLM appends this round's block
            binding.refresh(blocks, position=position, rows=16)
            rows = verify_trace_t2.kv_tile_rows(range(position, position + 16), binding.engine.pages[0].tolist())
            pages = {page for page, _ in rows}
            self.assertFalse(pages & set(shared), position)
            self.assertNotIn(shared[0], pages, 'never the pad')
            self.assertNotIn(0, pages)


# ------------------------------------------------------------------------------------------------
# End to end on the G1 model graft's toy: the S2 capture over the real staged route (A1 + A2 + A4 plan)
# ------------------------------------------------------------------------------------------------
class ToyExactnessTests(unittest.TestCase):
    def setUp(self):
        import qwen_prefix_model_fixture as F
        import test_qwen_prefix_model_runtime as model_runtime

        self.F, self.model_runtime = F, model_runtime
        self.engine = model_runtime.Engine(traced=False)
        self.warm()
        toy = self.engine.toy
        model = self.engine.model

        # The real prefill_masked_bucket reaches the chunk forward (_forward_prefill_chunk_masked ->
        # _tp); the toy's does not, so route the tail through it as the model does.
        def masked_bucket(token_ids, page_table, actual_len, chunk_start=0, bucket=None, flex_sdpa=True,
                          vision_tokens=None, vis_row_offset=0):
            if chunk_start == 0:
                model._reset_gdn_state_for_new_sequence()
                model._build_request_rope(token_ids[:, :actual_len], vision_tokens)
            bucket = (actual_len + 31) // 32 * 32
            hidden = model._forward_prefill_chunk_masked_tp(token_ids, actual_len, chunk_start, page_table, bucket)
            return toy.logits(hidden.data[0, 0])

        model.prefill_masked_bucket = masked_bucket

    def warm(self):
        """The plugin's compile-only warmup_model_prefill call, where the model graft warms under
        general-prefix (FastPathWarmToyExactnessTests warms the fast path's way instead)."""
        self.engine.warm()

    def admit(self, req_id, tokens, q, h=None):
        """The scheduler graft's stage and commit under sticky sessions (its own plan, with the drop)."""
        F = self.F
        registry = self.engine.registry
        registry.begin_step()
        ids = F.as_ids(tokens)
        h = q if h is None else h
        request = SimpleNamespace(request_id=req_id, all_token_ids=ids, num_prompt_tokens=len(ids),
                                  num_tokens=len(ids), block_hashes=F.BlockHashes(ids))
        stand_in = SimpleNamespace(registry=registry, sticky=True, drop_last=True)
        plan, unplanned, drain = graft.SchedulerGraft.plan(stand_in, request, h, q)
        key = F.key_at(ids, q) if q else None
        checkpoint = registry.get(key) if q else None
        registry.stage(prefix_registry.Grant(req_id, q, h, key, checkpoint, plan, request, drain, unplanned))
        registry.commit({req_id: q})
        return [pos for pos, _ in plan]

    def prefill(self, req_id, tokens, row, q, slot, capture):
        with patch.object(window, 'LayerOutputCapture', FakeLayerCapture), capture.capture():
            return self.engine.prefill([(req_id, tokens, row, q, slot)])

    def capture(self, position, start=0, prefix_route=True):
        return window.PrefillWindowCapture(SimpleNamespace(deallocate=Mock()), self.engine.model, position, (0,),
                                           start=start, prefix_route=prefix_route)

    def test_a_resumed_turn_equals_a_cold_prefill_under_the_fast_path_capture(self):
        F, engine = self.F, self.engine
        base = F.prompt(12700, seed=91)
        first = base[:9000]
        self.assertEqual(self.admit('n', first, 0), [6144])
        row_n = engine.pool.row(9000)
        cold_capture = self.capture(9000)
        self.prefill('n', first, row_n, 0, 1, cold_capture)
        self.assertEqual((cold_capture.complete, cold_capture.prefill_slot), (True, 1))
        self.assertIsNotNone(engine.registry.get(F.key_at(first.tolist(), 6144)), 'C0 captured mid-loop')
        self.assertIsNone(engine.registry.get(F.key_at(first.tolist(), 8192)))

        self.assertEqual(self.admit('n1', base, 6144, h=8192 - BLOCK), [10240])
        row = engine.pool.row(12700, row_n[0, :6144 // BLOCK].tolist())
        engine.toy.segments.clear()
        hit_capture = self.capture(12700, start=6144)
        hit = self.prefill('n1', base, row, 6144, 2, hit_capture)
        self.assertEqual(engine.toy.segments[0][1], 6144, 'only [Q, P) ran')
        self.assertEqual([chunk['chunk_start'] for chunk in hit_capture.chunks], [6144, 8192, 10240, 12288])
        self.assertEqual((hit_capture.complete, hit_capture.prefill_slot), (True, 2))
        hit_slot = [(rec.clone(), [c.clone() for c in convs]) for rec, convs in engine.toy.slot(2)]

        engine.registry.begin_step()
        cold_row = engine.pool.row(12700)
        twin = self.capture(12700)
        cold = self.prefill('cold', base, cold_row, 0, 3, twin)
        self.assertTrue(torch.equal(hit, cold), 'logits')
        for a, b in zip(engine.toy.kv(row, 12700), engine.toy.kv(cold_row, 12700)):
            self.assertTrue(torch.equal(a, b), 'KV [0, P)')
        for (rec, convs), (rec2, convs2) in zip(hit_slot, engine.toy.slot(3)):
            self.assertTrue(torch.equal(rec, rec2) and all(torch.equal(x, y) for x, y in zip(convs, convs2)), 'GDN slot')
        # the two captures hold the same draft window of the same chunks
        self.assertEqual(window.validate_prefill_chunks(12700, hit_capture.chunks, 6144),
                         window.validate_prefill_chunks(12700, twin.chunks))

    def test_the_resumed_route_writes_only_its_own_blocks_from_r(self):
        """docs/sticky-sessions-writer-audit.md, the prefill route's row, on the real staged route: a prefill
        resumed at R writes KV into exactly its own blocks from R to its last - never a shared prefix block it
        reads, never the page-table pad (block 0), never another request's block."""
        F, engine = self.F, self.engine
        base = F.prompt(12700, seed=94)
        first = base[:9000]
        self.admit('w', first, 0)
        row_w = engine.pool.row(9000)
        self.prefill('w', first, row_w, 0, 1, self.capture(9000))
        self.admit('w1', base, 6144, h=8192 - BLOCK)
        shared = row_w[0, :6144 // BLOCK].tolist()
        row = engine.pool.row(12700, shared)
        before = [cache.data.clone() for pair in engine.model._paged_kv_caches for cache in pair]
        self.prefill('w1', base, row, 6144, 2, self.capture(12700, start=6144))
        written = set()
        for old, cache in zip(before, [cache for pair in engine.model._paged_kv_caches for cache in pair]):
            changed = (old != cache.data).reshape(old.shape[0], -1).any(dim=1)
            written |= set(torch.nonzero(changed).reshape(-1).tolist())
        own = set(row[0, 6144 // BLOCK:-(-12700 // BLOCK)].tolist())
        self.assertEqual(written, own)
        self.assertFalse(written & set(row_w[0].tolist()))
        self.assertNotIn(0, written)

    def test_the_route_still_refuses_under_a_capture_that_does_not_record_it(self):
        F, engine = self.F, self.engine
        tokens_ = F.prompt(5000, seed=92)
        self.admit('r', tokens_, 0)
        with self.assertRaisesRegex(AssertionError, 'does not run under the C2 fast path'):
            self.prefill('r', tokens_, engine.pool.row(5000), 0, 0, self.capture(5000, prefix_route=False))
        self.assertEqual([segment for segment in engine.toy.segments if segment[0] != 'warm-eager'], [])

    def test_a_route_call_that_does_not_resume_at_the_capture_start_never_reaches_the_device(self):
        F, engine = self.F, self.engine
        tokens_ = F.prompt(9000, seed=93)
        self.admit('r', tokens_, 0)
        engine.toy.segments.clear()
        with self.assertRaisesRegex(ValueError, 'must start at this capture'):
            self.prefill('r', tokens_, engine.pool.row(9000), 0, 0, self.capture(9000, start=4096))
        self.assertEqual(engine.toy.segments, [])


# ------------------------------------------------------------------------------------------------
# The fast path's worker warmup runs the model graft's warm (serving_startup.prefix_warm)
# ------------------------------------------------------------------------------------------------
PREFIX_ON = {'QWEN_PREFIX_REUSE': '1'}


def boot_fast_path_worker(engine, environ, wrapper=None):
    """The fast path's boot on the G1 toy, through the P8 worker's own compile_or_warm_up_model
    (serving_plugin_patch.patch_worker of the pinned plugin worker.py, compiled as the method it is): no
    registry holder exists yet, serving_startup.warmup runs with start() recording what the model and the
    holder hold when it is reached, and the scheduler graft's registry is then created as vLLM creates it,
    after the worker's warmup (qwen_prefix_registry.shared_registry, which honours the holder's mid-loop
    declaration). The plugin's TTModelRunner.warmup_model is a Mock: the fast path must never reach it."""
    import serving_startup
    import test_qwen_prefix_model_runtime as model_runtime
    import test_qwen_prefix_runner_patch as runner_tests

    method = runner_tests.compile_method(runner_tests.p8_worker().decode('utf-8'), 'TTWorker',
                                         'compile_or_warm_up_model', {})
    wrapper = engine.wrapper if wrapper is None else wrapper
    seen = SimpleNamespace(started=[], registry=None, result=None, error=None)

    def start(worker):
        holder = sys.modules.get(prefix_registry.REGISTRY_KEY)
        seen.started.append(dict(restore_mode=getattr(wrapper.model[0], '_qwen_prefix_restore_mode', None),
                                 mid_loop=getattr(holder, 'mid_loop_capture', None)))

    runner = SimpleNamespace(model=wrapper, warmup_model=Mock(name='TTModelRunner.warmup_model'))
    worker = SimpleNamespace(vllm_config=policy_tests.FastPolicyTests().fixture(), model_runner=runner,
                             enable_model_warmup=True)
    worker_base = ModuleType('vllm.v1.worker.worker_base')
    worker_base.CompilationTimes = lambda **times: SimpleNamespace(**times)
    with patch.dict(sys.modules, {'vllm_tt_plugin.qwen_fast_policy': policy,
                                  'vllm.v1.worker.worker_base': worker_base}), \
            patch.object(serving_startup, 'start', side_effect=start), model_runtime.prefix_env(environ):
        sys.modules.pop(prefix_registry.REGISTRY_KEY, None)
        try:
            seen.result = method(worker)
        except ValueError as error:
            seen.error = error
        else:
            seen.registry = prefix_registry.shared_registry()
    seen.warmup_model = runner.warmup_model
    return seen


class FastPathWarmupTests(unittest.TestCase):
    """The P8 worker's compile_or_warm_up_model returns serving_startup.warmup before the plugin's
    TTModelRunner.warmup_model, the only caller of warmup_model_prefill, where the G1 model graft chooses its
    GDN restore path and declares its mid-loop captures. Without them the first granted resume asserts inside
    the route (the engine dies with every live user) and the first long request after boot plans no C0. So
    under QWEN_PREFIX_REUSE=1 serving_startup.warmup runs the graft's warm itself, before start()."""

    def setUp(self):
        import test_qwen_prefix_model_runtime as model_runtime

        self.engine = model_runtime.Engine(traced=False)

    def test_the_worker_warms_the_graft_before_the_fast_path_starts(self):
        seen = boot_fast_path_worker(self.engine, PREFIX_ON)
        self.assertIsNone(seen.error)
        self.assertEqual(len(seen.started), 1)
        self.assertIn(seen.started[0]['restore_mode'], ('h2d', 'copy'), 'the restore path is chosen before start()')
        self.assertIs(seen.started[0]['mid_loop'], True, 'the mid-loop captures are declared on the holder')
        self.assertIs(seen.registry.mid_loop_capture, True, 'and reach the registry the scheduler graft creates')
        self.assertIsNotNone(self.engine.model._qwen_prefix_state_spec)
        seen.warmup_model.assert_not_called()
        self.assertEqual(len(self.engine.log.lines('[PINDIAG] prefix: model warm restore_mode=')), 1)

    def test_the_flag_off_boot_is_the_fast_path_s_own(self):
        for environ in ({}, {'QWEN_PREFIX_REUSE': '0'}):
            with self.subTest(environ=environ):
                engine = type(self.engine)(traced=False)
                seen = boot_fast_path_worker(engine, environ)
                self.assertIsNone(seen.error)
                self.assertEqual(seen.started, [dict(restore_mode=None, mid_loop=None)])
                self.assertIsNone(getattr(engine.model, '_qwen_prefix_restore_mode', None))
                self.assertIs(seen.registry.mid_loop_capture, False)
                seen.warmup_model.assert_not_called()

    def test_a_model_without_the_graft_is_refused_before_start(self):
        stock = SimpleNamespace(model=[SimpleNamespace()])
        seen = boot_fast_path_worker(self.engine, PREFIX_ON, wrapper=stock)
        self.assertIn('has no _qwen_prefix_warm', str(seen.error))
        self.assertEqual(seen.started, [])

    def test_a_skipped_warm_is_refused_before_start(self):
        self.engine.model.args.max_batch_size = 1
        seen = boot_fast_path_worker(self.engine, PREFIX_ON)
        self.assertIn('chose no GDN restore path', str(seen.error))
        self.assertEqual(seen.started, [])
        self.assertEqual(len(self.engine.log.lines('[PINDIAG] prefix: model warm skipped')), 1)

    def first_plan(self, registry):
        """The scheduler graft's grant for the first 9000-token request after boot, on the fast path's DFlash
        scheduler shape. -> (plan, unplanned, the sticky oracle's plan)."""
        import prefix_judge

        with patch.dict(sys.modules, graft_fakes.fake_vllm_modules()), \
                patch.dict(os.environ, {'QWEN_PREFIX_STATS_PATH': ''}):
            scheduler = dflash_scheduler()
            graft.install(scheduler, registry=registry, kill_switch_path=None, logger=lambda *values: None,
                          stats=graft.StatsExport(path='', logger=lambda *values: None), environ=ON)
            first = graft_fakes.Request('first', tokens(9000, 1))
            scheduler.add(first)
            scheduler.schedule()
            grant = registry.grant_for('first')
            expected = prefix_judge.Oracle(sticky=True).admit('salt', list(first.all_token_ids))['plan']
        return grant.capture_positions(), list(grant.unplanned), expected

    def test_the_first_long_request_after_boot_plans_c0(self):
        self.assertEqual(self.first_plan(boot_fast_path_worker(self.engine, PREFIX_ON).registry),
                         ([6144], [], [6144]))
        # the control: booted without the warm, the same request plans nothing and C0 is unplanned
        engine = type(self.engine)(traced=False)
        self.assertEqual(self.first_plan(boot_fast_path_worker(engine, {}).registry), ([], [6144], [6144]))


class FastPathWarmToyExactnessTests(ToyExactnessTests):
    """ToyExactnessTests with the model warmed the fast path's way (boot_fast_path_worker) instead of by the
    plugin's warmup_model_prefill: the resumed turn under the S2 capture still equals its cold twin."""

    def warm(self):
        seen = boot_fast_path_worker(self.engine, PREFIX_ON)
        self.assertIsNone(seen.error)
        self.engine.registry = seen.registry


if __name__ == '__main__':
    unittest.main()
