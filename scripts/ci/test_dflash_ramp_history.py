"""C7 in the ramp: the feature-history write of a request whose committed history is short of 2048 rows.

A prompt under 2048 tokens starts its drafter at history_rows = len(prompt), and every round's
publication grows it by the accepted prefix until it reaches 2048. On the way, the general branch
of DFlashDevice.prepare_publication - and its QWEN_FAST_ROUND_B1 twin, which serving runs - wrote
the next history by slice, concat, slice and pad at shapes history_rows sets, so no round reused the
last one's programs. Run 36211578069 (the solo arm): 'hist' was ~0.9-1.1 s of a ~1 s round, and a
4096-token answer to a 60-token prompt took ten minutes; from 2047 tokens up (history_rows 2048
after the first round) the same arm ran at 29-41 tok/s. Under C7's condition
(dflash_device.history_unread: a committed K/V cache, no audit reporter, a captured proposal -
every any-request engine serving_request_factory builds) nothing reads that history, so the ramp
now skips the write and marks the history stale, as the B1 fused steady branch already did.

Pinned here, on the host:
  - a ramp round under C7 makes no operation on either history buffer, flag on or off, sets
    history_stale, and keeps rows, prefix, position, the named spare and kv_history.prepare's
    call as the writing path has them; every state outside C7 still writes, op for op;
  - a ramp round under C7 issues no operation whose arguments depend on history_rows;
  - what the draft reads is unchanged: through a ramp into the steady state, a real
    DraftKVHistory's committed banks are bit-identical with and without the skip, and so are
    position, history_rows and published_rows after every commit, while the writing arm's history
    is the sliding window it always was;
  - every reader of the history's content refuses a stale one, and the served single-user trace
    (kv_history set, no reporter) hands the history to no operation, stale or not;
  - the sites that touch a device's history in the serving modules are the reviewed ones, and the
    served engine is built in C7.
"""

import ast
import os
import re
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import dflash_packed_proposal
import test_dflash_device_publish as publish_fixtures
import test_dflash_packed_proposal_trace as trace_fixtures
import test_draft_kv_history as history_fixtures
from dflash_device import DFlashDevice, history_unread


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FLAG = 'QWEN_FAST_ROUND_B1'
AUDIT_FLAG = 'QWEN_FAST_ROUND_B1_AUDIT'
OPERATION_NAMES = ('slice', 'pad', 'matmul', 'typecast', 'rms_norm', 'concat', 'copy', 'synchronize_device')
# The general branch's write, in its order: the valid-history slice, the concat with the projected
# rows, the window slice, the pad to 2048 rows and the copy into the spare.
GENERAL_WRITE = ['slice', 'concat', 'slice', 'pad', 'copy']
# (history_rows, prefix): the early ramp, the middle, the last ramp rows, and the rounds that cross
# into 2048 (history_rows + prefix >= 2048 > history_rows).
RAMP_ROUNDS = ((50, 3), (1000, 16), (2047 - 7, 7), (2047, 1), (2040, 8), (2035, 16))

_ISOLATION = []


def setUpModule():
    """The B1 marker and audit counts are once-per-process module state (test_dflash_round_b1 does
    the same): this module works on its own copies, so no later module finds the marker logged."""
    counts = dict(rounds=0, **dict.fromkeys(dflash_packed_proposal.ROUND_B1_AUDIT_COUNTS, 0))
    for name, value in (('_ROUND_B1_NOTED', []), ('_ROUND_B1_AUDIT', counts)):
        patcher = patch.object(dflash_packed_proposal, name, value)
        patcher.start()
        _ISOLATION.append(patcher)


def tearDownModule():
    while _ISOLATION:
        _ISOLATION.pop().stop()


@contextmanager
def round_b1(on):
    """QWEN_FAST_ROUND_B1 exactly on or exactly absent, and its audit absent."""
    with patch.dict(os.environ):
        os.environ.pop(FLAG, None)
        os.environ.pop(AUDIT_FLAG, None)
        if on:
            os.environ[FLAG] = '1'
        yield


def fake_kv_history():
    return SimpleNamespace(prepare=Mock(side_effect=lambda projected, prefix, position: SimpleNamespace(
        projected=projected, prefix=prefix, position=position)), discard=Mock(), commit=Mock(), audit=Mock())


def mock_device(operations, *, history_rows, kv_history='fake', progress=None, capture=True):
    """test_dflash_device_publish's device, in the state a served any-request drafter is in (C7)
    unless an option says otherwise."""
    device = publish_fixtures.build_device(operations, kv_history=fake_kv_history() if kv_history == 'fake' else kv_history)
    device.history_rows = history_rows
    device.progress = progress
    device.proposal_capture = object() if capture else None
    return device


def publish(on, *, history_rows, prefix=3, fused=False, merge_release=False, **device_options):
    """One prepare_publication on the mock runtime: its ordered calls, the names of the calls that
    touched either history buffer, the device and the pending record."""
    operations = publish_fixtures.fake_operations()
    order = Mock()
    for name in OPERATION_NAMES:
        order.attach_mock(getattr(operations, name), name)
    device = mock_device(operations, history_rows=history_rows, **device_options)
    stack = publish_fixtures.patched(operations)
    with round_b1(on), stack[0], stack[1], stack[2], stack[3]:
        pending = device.prepare_publication([publish_fixtures.make_feature_tap() for _ in range(5)], prefix,
            position=device.position, merge_release=merge_release, fused_steady_state=fused)
    buffers = (device.history, device.spare_history)
    touching = [call[0] for call in order.mock_calls
                if any(argument is buffer for argument in (*call[1], *call[2].values()) for buffer in buffers)]
    return list(order.mock_calls), touching, device, pending


def names(calls):
    return [call[0] for call in calls]


def simple(value):
    """Whether a call argument is plain data (an offset, extent, padding, dim, dtype name)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (tuple, list)):
        return all(simple(item) for item in value)
    return False


def geometry(calls):
    """Each call's name and its plain-data arguments: what decides the programs it builds."""
    return [(call[0], tuple(argument for argument in call[1] if simple(argument)),
             tuple(sorted((key, value) for key, value in call[2].items() if simple(value)))) for call in calls]


class RampWriteTests(unittest.TestCase):
    """The ramp's write under C7, on the mock runtime, flag on and off."""

    def test_a_ramp_round_under_c7_touches_neither_history_buffer(self):
        for on in (False, True):
            for fused in (False, True):
                for history_rows, prefix in RAMP_ROUNDS:
                    with self.subTest(b1=on, fused=fused, history_rows=history_rows, prefix=prefix):
                        calls, touching, device, pending = publish(on, history_rows=history_rows, prefix=prefix,
                                                                   fused=fused)
                        if fused and not on and history_rows + prefix >= 2048:
                            # The crossing round with fused_steady_state takes the fused branch, which
                            # skips under B1 alone (C7 steady); flag off it writes, as it always did.
                            self.assertEqual(touching, ['slice', 'copy'])
                            self.assertFalse(getattr(device, 'history_stale', False))
                            continue
                        self.assertEqual(touching, [], 'no operation reads or writes either history buffer')
                        self.assertTrue(device.history_stale)
                        self.assertIs(pending.history, device.spare_history, 'commit still swaps the same pair')
                        self.assertEqual((pending.rows, pending.prefix, pending.position, pending.status),
                                         (min(2048, history_rows + prefix), prefix, 100, 'prepared'))
                        device.kv_history.prepare.assert_called_once()
                        self.assertEqual(device.kv_history.prepare.call_args.args[1:], (prefix,))
                        self.assertEqual(device.kv_history.prepare.call_args.kwargs, dict(position=100))
                        self.assertEqual((pending.kv.prefix, pending.kv.position), (prefix, 100),
                                         'the pending record carries what kv_history.prepare returned')

    def test_the_skip_drops_exactly_the_general_write(self):
        """Against the writing path (the same round without a captured proposal): the five
        operations of the general branch go, every other operation stays in its order, and the
        bookkeeping is the writing path's."""
        for on in (False, True):
            for merge_release in (False, True):
                for history_rows, prefix in RAMP_ROUNDS:
                    with self.subTest(b1=on, merge_release=merge_release, history_rows=history_rows, prefix=prefix):
                        written = publish(on, history_rows=history_rows, prefix=prefix, merge_release=merge_release,
                                          capture=False)
                        skipped = publish(on, history_rows=history_rows, prefix=prefix, merge_release=merge_release)
                        self.assertEqual(written[1], ['slice', 'copy'], 'the valid-history slice and the spare copy')
                        write_names = names(written[0])
                        start = next(index for index, call in enumerate(written[0])
                                     if any(argument is written[2].history for argument in call[1]))
                        self.assertEqual(write_names[start:start + 5], GENERAL_WRITE)
                        self.assertEqual(names(skipped[0]), write_names[:start] + write_names[start + 5:])
                        self.assertFalse(getattr(written[2], 'history_stale', False))
                        for field in ('rows', 'prefix', 'position', 'status'):
                            self.assertEqual(getattr(skipped[3], field), getattr(written[3], field))
                        self.assertEqual(skipped[2].kv_history.prepare.call_args.args[1:],
                                         written[2].kv_history.prepare.call_args.args[1:])
                        self.assertEqual(skipped[2].kv_history.prepare.call_args.kwargs,
                                         written[2].kv_history.prepare.call_args.kwargs)

    def test_every_state_outside_c7_still_writes(self):
        states = dict(audit=dict(progress=lambda *a, **k: None), no_cache=dict(kv_history=None),
                      eager=dict(capture=False))
        for on in (False, True):
            for label, options in states.items():
                for history_rows, prefix in RAMP_ROUNDS:
                    with self.subTest(b1=on, state=label, history_rows=history_rows, prefix=prefix):
                        calls, touching, device, pending = publish(on, history_rows=history_rows, prefix=prefix,
                                                                   **options)
                        self.assertEqual(touching, ['slice', 'copy'])
                        self.assertIn(GENERAL_WRITE, [names(calls)[index:index + 5] for index in range(len(calls))])
                        self.assertFalse(getattr(device, 'history_stale', False))

    def test_the_steady_state_is_as_it_was(self):
        """At history_rows == 2048 nothing changed: the unfused (sequential) publication still
        writes, whatever C7 says, and the B1 fused branch skips under C7 alone."""
        for on in (False, True):
            with self.subTest(b1=on, fused=False):
                calls, touching, device, pending = publish(on, history_rows=2048, prefix=5)
                self.assertEqual(touching, ['slice', 'copy'])
                self.assertFalse(getattr(device, 'history_stale', False))
            with self.subTest(b1=on, fused=True):
                calls, touching, device, pending = publish(on, history_rows=2048, prefix=5, fused=True)
                self.assertEqual(touching, [] if on else ['slice', 'copy'])
                self.assertEqual(getattr(device, 'history_stale', False), on)

    def test_a_ramp_round_under_c7_depends_on_no_history_rows(self):
        """Every operation a skipped ramp round issues has the same plain-data arguments at any
        history_rows - the programs it needs are the prefix's, built once - where the writing path's
        do not (the control)."""
        for on in (False, True):
            for prefix in (1, 7, 16):
                with self.subTest(b1=on, prefix=prefix):
                    skipped = [geometry(publish(on, history_rows=rows, prefix=prefix)[0]) for rows in (50, 777, 2031)]
                    self.assertEqual(skipped[1], skipped[0])
                    self.assertEqual(skipped[2], skipped[0])
                    steady = geometry(publish(True, history_rows=2048, prefix=prefix, fused=True)[0]) if on else None
                    if steady is not None:
                        self.assertEqual(skipped[0], steady, 'the ramp round is the steady C7 round')
                    written = [geometry(publish(on, history_rows=rows, prefix=prefix, capture=False)[0])
                               for rows in (50, 777)]
                    self.assertNotEqual(written[0], written[1])

    def test_history_unread_is_the_steady_branch_s_condition(self):
        """The ramp's condition is the one the B1 fused steady branch tests inline, state for state:
        both skip exactly where history_unread holds."""
        for kv_history in ('fake', None):
            for progress in (None, lambda *a, **k: None):
                for capture in (True, False):
                    options = dict(kv_history=kv_history, progress=progress, capture=capture)
                    with self.subTest(kv_history=kv_history is not None, progress=progress is not None, capture=capture):
                        unread = history_unread(mock_device(publish_fixtures.fake_operations(), history_rows=50, **options))
                        self.assertEqual(unread, kv_history is not None and progress is None and capture)
                        steady = publish(True, history_rows=2048, fused=True, **options)
                        ramp = publish(True, history_rows=50, **options)
                        self.assertEqual(getattr(steady[2], 'history_stale', False), unread)
                        self.assertEqual(getattr(ramp[2], 'history_stale', False), unread)

    def test_a_bare_fixture_device_is_not_c7(self):
        """Fixtures that carry no progress or proposal_capture attribute (test_dflash_device_publish)
        keep today's write: getattr, never an AttributeError."""
        device = publish_fixtures.build_device(publish_fixtures.fake_operations(), kv_history=fake_kv_history())
        self.assertFalse(hasattr(device, 'proposal_capture'))
        self.assertFalse(history_unread(device))


def bank_bits(cache):
    return [pair[name].contiguous().view(torch.int16) for pair in cache.active for name in ('k', 'v')]


class RampLifeTests(unittest.TestCase):
    """A whole ramp into the steady state on real tensors: the DFlash publication over the
    DraftKVHistory test's torch runtime and a real DraftKVHistory, one drafter skipping (C7) and one
    writing (no captured proposal), stepped in lockstep with the same projected rows."""

    SCHEDULES = {
        # A 60-token prompt's first rounds.
        60: (5, 16, 1, 9),
        # The last ramp rounds, the round that crosses into 2048 (2042 + 12) and four steady rounds.
        1990: (16, 1, 7, 3, 16, 9, 12, 16, 2, 16, 11),
    }

    def device(self, operations, cache, history, *, capture, position):
        device = SimpleNamespace(operations=operations, mesh='mesh', closed=False, pending=None, position=position,
            history_rows=min(position, 2048), history=history, spare_history=torch.zeros_like(history),
            kv_history=cache, owned=[], borrowed=[], progress=None,
            proposal_capture=object() if capture else None, published_rows=0)
        device.temporaries = types.MethodType(DFlashDevice.temporaries, device)
        return device

    def live(self, *, on, fused, start, prefixes):
        case = history_fixtures.DraftKVHistoryTests('test_prepare_discard_and_commit_keep_all_layers_at_one_frontier')
        rows = min(start, 2048)
        initial = case.features(rows, start)
        padded = torch.nn.functional.pad(initial, (0, 0, 0, 2048 - rows))
        taps = [publish_fixtures.make_feature_tap() for _ in range(5)]
        with round_b1(on), patch('dflash_device.addresses', side_effect=lambda operations, value: (id(value),)), \
                patch('dflash_device.release_owned'), \
                case.fixture(initial, start, storage=history_fixtures.pooled_storage(layers=2)) as (cache_skip, ops_skip), \
                case.fixture(initial, start, storage=history_fixtures.pooled_storage(layers=2)) as (cache_write, ops_write):
            skip = self.device(ops_skip, cache_skip, padded.clone(), capture=True, position=start)
            write = self.device(ops_write, cache_write, padded.clone(), capture=False, position=start)
            buffers = (skip.history, skip.spare_history)
            window = initial
            for round_number, prefix in enumerate(prefixes):
                # A round the skip covers: the ramp, less the crossing round flag off with
                # fused_steady_state, which takes the fused branch and writes as it always did.
                ramp = skip.history_rows < 2048 and not (fused and not on and skip.history_rows + prefix >= 2048)
                projected = case.features(prefix, 7919 + skip.position)
                for device in (skip, write):
                    device.project_features = lambda features, count, retain=None, projected=projected: projected
                ops_skip.copy.reset_mock()
                ops_write.copy.reset_mock()
                published = []
                for device in (skip, write):
                    publication = DFlashDevice.prepare_publication(device, taps, prefix, position=device.position,
                                                                   fused_steady_state=fused)
                    published.append((publication.rows, publication.prefix, publication.position))
                    DFlashDevice.commit_publication(device, publication)
                with self.subTest(b1=on, fused=fused, start=start, round=round_number, prefix=prefix):
                    self.assertEqual(published[0], published[1])
                    for name in ('position', 'history_rows', 'published_rows'):
                        self.assertEqual(getattr(skip, name), getattr(write, name))
                    self.assertEqual((cache_skip.position, cache_skip.history_rows),
                                     (cache_write.position, cache_write.history_rows))
                    self.assertEqual(cache_skip.history_rows, skip.history_rows)
                    for mine, theirs in zip(bank_bits(cache_skip), bank_bits(cache_write), strict=True):
                        self.assertTrue(torch.equal(mine, theirs), 'the committed K/V banks the draft reads')
                    window = torch.cat((window, projected), dim=2)[..., -write.history_rows:, :]
                    self.assertTrue(torch.equal(write.history[..., :write.history_rows, :].view(torch.int16),
                                                window.contiguous().view(torch.int16)), 'the writing arm is today\'s')
                    self.assertEqual(torch.count_nonzero(write.history[..., write.history_rows:, :]).item(), 0)
                    self.assertTrue(skip.history_stale)
                    self.assertFalse(getattr(write, 'history_stale', False))
                    if ramp:
                        touched = [call for call in ops_skip.copy.call_args_list
                                   if any(argument is buffer for argument in call.args for buffer in buffers)]
                        self.assertEqual(touched, [], 'the skipping arm wrote neither history buffer')
                        self.assertTrue(any(call.args[1] is write.history for call in ops_write.copy.call_args_list),
                                        'the writing arm wrote its spare, now its history')
        return skip

    def test_the_draft_reads_the_same_banks_through_the_ramp(self):
        for on in (False, True):
            for fused in (False, True):
                for start, prefixes in self.SCHEDULES.items():
                    skip = self.live(on=on, fused=fused, start=start, prefixes=prefixes)
                    self.assertEqual(skip.history_rows, min(2048, start + sum(prefixes)))


class StaleReaderTests(unittest.TestCase):
    """Every reader of the history's content refuses a stale one; the served trace reads none."""

    def test_the_eager_proposal_refuses_a_stale_history_before_any_operation(self):
        operations = Mock()
        stale = SimpleNamespace(closed=False, pending=None, proposal_capture=None, max_drafts=15, history_stale=True,
                                operations=operations)
        with self.assertRaisesRegex(ValueError, 'stale'):
            DFlashDevice.propose(stale, 7, 3)
        self.assertEqual(operations.mock_calls, [])

    def test_commit_publication_refuses_to_audit_a_stale_history(self):
        publication = SimpleNamespace(status='prepared', position=100, kv=object(), history=object(), rows=61, prefix=2)
        auditing = SimpleNamespace(closed=False, pending=publication, position=100, kv_history=fake_kv_history(),
                                   history=object(), progress=lambda *a, **k: None, published_rows=0, history_stale=True)
        with self.assertRaisesRegex(ValueError, 'stale'):
            DFlashDevice.commit_publication(auditing, publication)
        auditing.kv_history.audit.assert_not_called()

    def capture(self, *, kv_history, progress, history_stale):
        """A single-user proposal trace over a ramp device (position 100, history_rows 100, the 256
        bucket), test_dflash_round_b1's construction."""
        from dflash_proposal_trace import PreparedDFlashProposal

        operations = trace_fixtures.fake_operations()
        device = SimpleNamespace(operations=operations, mesh=object(), position=100, history_rows=100, block_rows=16,
            progress=progress, history=object(), spare_history=object(), history_stale=history_stale,
            live_query_qk=False, native_proposal_attention=False, owned=[],
            temporaries=lambda protected: ([], lambda value: value))
        capture = PreparedDFlashProposal.__new__(PreparedDFlashProposal)
        capture.device, capture.operations, capture.mesh = device, operations, device.mesh
        capture.owned, capture.kv_history = [], kv_history
        placeholder = lambda: SimpleNamespace(dtype='bf16', layout='tile')
        bucket = SimpleNamespace(context=256, identifiers=placeholder(), mask=placeholder(),
            rope={name: (placeholder(), placeholder()) for name in ('q', 'k')}, history=placeholder(),
            cached_history=None if kv_history is None else [{name: placeholder() for name in ('k', 'v')}
                                                            for _ in kv_history.active])
        bucket.inputs, bucket.addresses = [], []
        return capture, bucket, device

    @staticmethod
    def kv_history():
        return SimpleNamespace(pending=None, position=100, history_rows=100, owned=[], borrowed=[],
                               active=[{'k': object(), 'v': object()} for _ in range(5)])

    def test_the_single_user_trace_refuses_to_copy_a_stale_history(self):
        """Its two copying states - no K/V cache, or a reporter - on both update paths."""
        for label, kv_history, progress in (('no-cache', None, None),
                                            ('reporter', self.kv_history(), lambda *a, **k: None)):
            for defer_finish in (False, True):
                with self.subTest(state=label, defer_finish=defer_finish):
                    capture, bucket, device = self.capture(kv_history=kv_history, progress=progress, history_stale=True)
                    with patch('dflash_proposal_trace.release_owned'), \
                            patch('dflash_proposal_trace.addresses', side_effect=lambda operations, value: (id(value),)), \
                            self.assertRaisesRegex(ValueError, 'stale'):
                        capture.update(bucket, 5, defer_finish=defer_finish)
                    for call in device.operations.slice.call_args_list:
                        self.assertNotIn(device.history, call.args)

    def test_the_served_single_user_trace_hands_the_history_to_no_operation(self):
        """kv_history set and no reporter - the any-request engine's trace, at every ladder rung
        and after a bucket switch alike: it copies the K/V banks and nothing else, stale or not,
        on the blocking and the prepared (early-draft, pipelined) paths."""
        for stale in (False, True):
            for defer_finish in (False, True):
                with self.subTest(stale=stale, defer_finish=defer_finish):
                    capture, bucket, device = self.capture(kv_history=self.kv_history(), progress=None,
                                                           history_stale=stale)
                    with patch('dflash_proposal_trace.release_owned'), \
                            patch('dflash_proposal_trace.addresses', side_effect=lambda operations, value: (id(value),)):
                        capture.update(bucket, 5, defer_finish=defer_finish)
                    operations = device.operations
                    self.assertEqual(operations.slice.call_count, 10, 'the five layers\' k and v banks')
                    for call in (*operations.slice.call_args_list, *operations.copy.call_args_list,
                                 *operations.copy_host_to_device_tensor.call_args_list):
                        self.assertFalse(any(argument is device.history or argument is device.spare_history
                                             for argument in (*call.args, *call.kwargs.values())))


def history_sites(path):
    """{(scope, receiver)} of every read of `.history` or `.spare_history` in a module."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    found = set()

    def visit(node, scope):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit(child, scope + (child.name,))
                continue
            if isinstance(child, ast.Attribute) and child.attr in ('history', 'spare_history'):
                found.add(('.'.join(scope), ast.unparse(child.value)))
            visit(child, scope)
    visit(tree, ())
    return found


class HistorySiteTests(unittest.TestCase):
    """Where the serving modules touch a drafter's feature history, each site reviewed for C7. A new
    site fails here until it is classified - and, if it reads the content, refuses a stale history."""

    SERVING = ('dflash_device.py', 'dflash_proposal_trace.py', 'dflash_packed_proposal.py',
               'dflash_packed_proposal_coordinator.py', 'quad_draft.py', 'pair_row_exact.py', 'fused_commit.py',
               'publish_prewarm.py', 'dflash_traced_publish.py', 'dflash_pipelined_publish.py',
               'dflash_request_runtime.py', 'serving_request_factory.py', 'serving_fast_request.py',
               'serving_worker_hook.py', 'early_draft.py', 'serving_sequential_step.py', 'serving_packed_step.py',
               'serving_packed_bridge.py', 'serving_runtime.py', 'serving_lifecycle.py', 'serving_buffer_pool.py',
               'packed_verifier.py', 'verify_prestage.py')
    PROTECTED = 'addresses only: protected from a temporary, never read'
    SITES = {
        ('dflash_device.py', 'DFlashDevice.__init__', 'self'):
            'builds the prefill history and projects the K/V banks from it, before any publication',
        ('dflash_device.py', 'DFlashDevice.__init__', 'self.pool_slot'): 'borrows the pooled pair',
        ('dflash_device.py', 'DFlashDevice.prepare_publication', 'self'):
            'the write: reads the history only on a branch C7 does not skip',
        ('dflash_device.py', 'DFlashDevice._prepare_publication_round_b1', 'self'):
            'the write: reads the history only on a branch C7 does not skip',
        ('dflash_device.py', 'DFlashDevice.commit_publication', 'self'):
            'swaps the pair; audits only with a reporter, refusing a stale history',
        ('dflash_device.py', 'DFlashDevice.commit_publication', 'publication'): 'the spare the record names',
        ('dflash_device.py', 'DFlashDevice.propose', 'self'):
            'the eager proposal, only without a captured proposal, refusing a stale history first',
        ('dflash_device.py', 'DFlashDevice.close', 'self'): 'teardown',
        ('dflash_device.py', 'ProposalAudit.not_owned', 'device'): 'addresses only, inside the eager proposal',
        ('dflash_packed_proposal.py', 'propose_packed', 'device'): PROTECTED,
        ('dflash_proposal_trace.py', 'PreparedDFlashProposal.__init__', 'bucket'): 'the bucket placeholder',
        ('dflash_proposal_trace.py', 'PreparedDFlashProposal.execute', 'bucket'): 'the bucket placeholder',
        ('dflash_proposal_trace.py', 'PreparedDFlashProposal.update', 'device'): PROTECTED,
        ('dflash_proposal_trace.py', 'PreparedDFlashProposal.update.copy_history_and_cache', 'device'):
            'copied only without a K/V cache or with a reporter, refusing a stale history',
        ('dflash_proposal_trace.py', 'PreparedDFlashProposal.update.copy_history_and_cache', 'bucket'):
            'the bucket placeholder',
        ('dflash_proposal_trace.py', 'PreparedPackedDFlashProposal._bucket', 'device'): PROTECTED,
        ('dflash_proposal_trace.py', 'PreparedPackedDFlashProposal._bucket', 'self.device_b'): PROTECTED,
        ('dflash_proposal_trace.py', 'PreparedPackedDFlashProposal._update', 'device_a'): PROTECTED,
        ('dflash_proposal_trace.py', 'PreparedPackedDFlashProposal._update', 'device_b'): PROTECTED,
        ('fused_commit.py', 'FusedCommit.prepare', 'drafter'): 'the spare its record names; it marks the history stale',
        ('quad_draft.py', 'PreparedQuadDFlashProposal._protected', 'device'): PROTECTED,
        ('serving_buffer_pool.py', 'HistorySlot.__init__', 'self'): 'allocation',
        ('serving_sequential_step.py', 'replicated_buffers', 'device'):
            'QWEN_FAST_SHARD_CHECK diagnostic (0 in the C2 image): compares the chips\' copies, equal when unwritten',
    }

    def test_the_sites_are_the_reviewed_ones(self):
        found = set()
        for name in self.SERVING:
            path = HERE / name
            self.assertTrue(path.is_file(), name)
            found.update((name, scope, receiver) for scope, receiver in history_sites(path))
        self.assertEqual(sorted(found - set(self.SITES)), [], 'new sites: classify each (and guard a reader)')
        self.assertEqual(sorted(set(self.SITES) - found), [], 'reviewed sites that no longer exist')

    def test_the_served_engine_is_built_in_c7(self):
        """serving_request_factory.from_prefill builds its drafter with a K/V cache, a deferred
        proposal capture and no reporter; the engine's before_capture captures the proposal unless
        QWEN_FAST_EAGER_PROPOSAL, which the C2 image does not set - nor QWEN_FAST_PROPOSAL_AUDIT."""
        tree = ast.parse((HERE / 'serving_request_factory.py').read_text(encoding='utf-8'))
        function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'from_prefill')
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == 'components.device']
        self.assertEqual(len(calls), 1)
        keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in calls[0].keywords}
        self.assertEqual((keywords['cache_history'], keywords['proposal_capture'], keywords['defer_proposal_capture']),
                         ('True', 'True', 'True'))
        self.assertNotIn('progress', keywords)
        dockerfile = (ROOT / 'docker' / 'qwen-c2-serving.Dockerfile').read_text(encoding='utf-8')
        self.assertIn('QWEN_FAST_ROUND_B1=1', dockerfile)
        for flag in ('QWEN_FAST_EAGER_PROPOSAL', 'QWEN_FAST_PROPOSAL_AUDIT'):
            self.assertIsNone(re.search(flag + '=1', dockerfile), flag)
            self.assertNotIn(flag, (HERE / 'qwen_c2_profiles.json').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
