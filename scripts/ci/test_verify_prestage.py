"""Round-fence plan H1a, the pre-stage (QWEN_FAST_PRESTAGE, QWEN_FAST_PRESTAGE_AUDIT; default off):
verify_prestage.BlockPrestage writes every input of the next packed verify but its tokens inside the
drafts' fence window, and verify() - while the fixture write epoch has not moved - skips
validate_bindings and writes only the buffers whose recomputed value differs, with no fence.

The block is test_packed_verifier's four-user fixture with test_padded_probe's fake device model
(every replay computes its outputs from what is staged), so a wrong staged byte is a wrong prediction
too. Here, over random users, page tables, idle sets and T2 on or off:
  - the snapshot plus the diff leaves the device holding exactly the {destination: value} map that
    stage_packed writes, and the predictions are the flag-off run's;
  - a page append for one user rewrites exactly that user's page-derived buffers plus the block-wide
    tables (and the tokens, always);
  - every other writer of fixture inputs bumps the epoch and the next verify takes the full path;
  - a failure inside the pre-stage leaves the round on today's path, never poisoned;
  - reader start and poison behave exactly as today;
  - padded idle fill is pre-staged and diffed like any live segment;
  - the audit reads back 8 buffers in rotation, and a stray write it finds is restaged before the trace.
Then the plumbing (hook, step, coordinator), the gate, the arm and the shipping lists; and, with every
flag off, each module H1a touches against its PARENT copy, call for call."""

import io
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from dflash_packed_proposal_coordinator import note_fenced, run_while_waiting
import packed_verifier
import serving_packed_step
import serving_worker_hook
import test_packed_verifier as tpv
import test_padded_probe as tpp
import verifier_engine
import verify_prestage
import verify_trace_t2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
# H1a's parent: the gate-only commit after 2f047a02; every module H1a touches is 2f047a02's there.
PARENT = '81acfec9'
PAGE_WIDTH = tpv.PAGE_WIDTH
NAMES = 'ABCD'
FAMILY_FIRST, FAMILY_LAST = 4096, 4352 - 16


def parent_module(relative):
    from test_padded_block import parent_module as load

    return load(relative, PARENT)


def clean_environment(**flags):
    environ = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
    environ.update(flags)
    return patch.dict(os.environ, environ, clear=True)


def table(values):
    return torch.tensor([list(values)], dtype=torch.int32)


def user_table(rng, index):
    """A page table of user `index`'s own pages: nonzero (never page 0) and disjoint from every
    other user's, as vLLM's append-only allocation leaves them."""
    return table(rng.randrange(100 * (index + 1), 100 * (index + 1) + 99) for column in range(PAGE_WIDTH))


class FlagTests(unittest.TestCase):
    def test_the_flags_are_zero_or_one_and_the_audit_needs_the_prestage(self):
        self.assertFalse(verify_prestage.enabled({}))
        self.assertTrue(verify_prestage.enabled({'QWEN_FAST_PRESTAGE': '1'}))
        self.assertFalse(verify_prestage.audit_enabled({'QWEN_FAST_PRESTAGE_AUDIT': '1'}))
        self.assertTrue(verify_prestage.audit_enabled({'QWEN_FAST_PRESTAGE': '1', 'QWEN_FAST_PRESTAGE_AUDIT': '1'}))
        self.assertTrue(verify_prestage.any_enabled({'QWEN_FAST_ROUND_FENCES': '1'}))
        self.assertFalse(verify_prestage.any_enabled({}))
        for name in ('QWEN_FAST_PRESTAGE', 'QWEN_FAST_PRESTAGE_AUDIT', 'QWEN_FAST_ROUND_FENCES'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify_prestage._flag(name, {name: 'yes'})

    def test_the_epoch_moves_on_every_bump_and_names_the_last(self):
        before = verify_prestage.epoch()
        verify_prestage.bump('test')
        self.assertEqual((verify_prestage.epoch(), verify_prestage.last_bump()), (before + 1, 'test'))

    def test_a_block_built_without_the_flag_has_no_prestage(self):
        fixture = tpv.FourUserFixture('run')
        fixture.run = lambda: None
        fixture.setUp()
        try:
            with clean_environment():
                block = fixture.build()
            self.assertIsNone(block.prestaged)
            self.assertFalse(block.round_fences)
            self.assertNotIn('prestage', block.describe())
        finally:
            fixture.doCleanups()


class PrestageFixture(tpp.ProbeFixture):
    """The four-user block with the fake device model; every H1a line captured."""

    def setUp(self):
        super().setUp()
        self.h1a = []
        for patcher in (patch.object(verify_prestage, 'log_line', side_effect=self.h1a.append),
                        patch.object(packed_verifier, 'diagnostic', side_effect=self.h1a.append),
                        patch.object(verify_trace_t2, 'log_line', side_effect=self.h1a.append)):
            patcher.start()
            self.addCleanup(patcher.stop)
        verify_trace_t2._LOGGED.clear()
        self.addCleanup(verify_trace_t2._LOGGED.clear)
        self.block = None

    def open_block(self, prestage=True, audit=False, padded=None, kv_chains=False):
        if self.block is not None:
            self.block.close()
        os.environ.pop('QWEN_FAST_PRESTAGE', None)
        os.environ.pop('QWEN_FAST_PRESTAGE_AUDIT', None)
        if prestage:
            os.environ['QWEN_FAST_PRESTAGE'] = '1'
        if audit:
            os.environ['QWEN_FAST_PRESTAGE_AUDIT'] = '1'
        verifier_engine.note_prefill()
        block = self.build(**({} if padded is None else dict(padded_min_users=padded)))
        if kv_chains:
            block.fixture.kv_chains = True
        self.model_hook = tpp.DeviceModel(self, block)
        self.block = block
        return block

    def entries(self, users, token_base=0):
        """Entries for `users` = {name: (position, table)}, in the order given."""
        made = []
        for name, (position, pages) in users.items():
            index = NAMES.index(name)
            owner = tpv.request(name, self.pool.slots[index], position, 1)
            owner.engine.pages = pages.clone()
            first = (token_base + 10 * index) % 80
            made.append(tpv.entry(owner, range(first, first + 16)))
        return made

    def staged_users(self, block, entries):
        """What the verify stages, in segment order: stage_packed's `users`."""
        segments = block.segments(entries)
        users = block.segment_users(entries, segments)
        if len(segments) < block.users:
            users = block.padded_users(users, segments)
        return users

    def round(self, block, users, token_base=0, prefixes=(9, 16, 0, 4)):
        """One verify of `users` and every live commit; (predictions, metrics, staged users)."""
        entries = self.entries(users, token_base)
        staged = self.staged_users(block, entries)
        predictions, metrics = block.verify(entries)
        for segment, prefix in zip(metrics['segments'], prefixes):
            block.commit_user(segment, prefix)
        return predictions, metrics, staged

    def window_requests(self, users, finished=()):
        requests = []
        for name, (position, pages) in users.items():
            carry = [list(snapshot) for snapshot in self.pool.slots[NAMES.index(name)].verifier.carry]
            requests.append(SimpleNamespace(session=SimpleNamespace(finished=False, position=position),
                                            engine=SimpleNamespace(carry=carry, pages=pages.clone())))
        for name in finished:
            carry = [list(snapshot) for snapshot in self.pool.slots[NAMES.index(name)].verifier.carry]
            requests.append(SimpleNamespace(session=SimpleNamespace(finished=True, position=0),
                                            engine=SimpleNamespace(carry=carry, pages=None)))
        return requests

    def window(self, block, users, finished=()):
        """The drafts' window as the coordinator runs it: the callable, the fence F9, `fenced`."""
        waiting = verify_prestage.WhileWaiting(block, self.window_requests(users, finished))
        run_while_waiting(waiting)
        self.ttnn.synchronize_device(self.model.mesh_device)
        note_fenced(waiting)
        return waiting

    def assert_device_holds(self, block, staged):
        """Every staging destination holds exactly what stage_packed would write for `staged`."""
        values, readers = packed_verifier.packed_values(block.operations, block.model, block.fixture, block.shape,
                                                        staged, guard=False)
        wrong = [index for index, (destination, value, dtype, layout) in enumerate(values)
                 if not (destination.value.dtype == value.dtype and torch.equal(destination.value, value))]
        self.assertEqual(wrong, [], 'destinations not holding the staged value')
        self.assertEqual([reader.start for reader in readers], [user[1] for user in staged])

    def marked(self, marker):
        return [line for line in self.h1a if line.startswith(marker + ' ')]

    def paths(self):
        return [re.search(r'path=(\w+)', line).group(1) for line in self.marked(verify_prestage.MARKER)]

    def reasons(self):
        return [re.search(r'reason=(\S+)', line).group(1) for line in self.marked(verify_prestage.MARKER)]

    def written(self, operation):
        """The destinations `operation` copied into, in order."""
        before = len(self.ttnn.host_copies)
        operation()
        return [destination for host, destination in self.ttnn.host_copies[before:]]


def base_users():
    return {name: (tpv.FourUserFixture.POSITIONS[index], table([tpv.FourUserFixture.PAGES[index]] * PAGE_WIDTH))
            for index, name in enumerate(NAMES)}


def advanced(users, steps):
    return {name: (position + steps, pages) for name, (position, pages) in users.items()}


class EquivalenceTests(PrestageFixture):
    """Random users, page tables, idle sets and T2 on or off: the pre-stage plus the diff is the
    full stage, byte for byte, and the predictions are the flag-off run's."""

    def schedule(self, rng, rounds, padded):
        positions = {name: rng.randrange(4100, 4180) for name in NAMES}
        tables = {name: user_table(rng, index) for index, name in enumerate(NAMES)}
        plan = []
        for number in range(rounds):
            live = NAMES
            if padded and number and rng.random() < 0.5:
                live = ''.join(sorted(rng.sample(NAMES, rng.choice((2, 3)))))
            window_tables = {name: tables[name].clone() for name in NAMES}
            # vLLM appends a block at execute time, after the window, for some rounds
            if number and rng.random() < 0.6:
                name = rng.choice(live)
                index = NAMES.index(name)
                column = min(PAGE_WIDTH - 1, (positions[name] + 15) // 64 + rng.randrange(0, 2))
                tables[name][0, column] = 100 * (index + 1) + 99
            plan.append(dict(live=live, token_base=rng.randrange(0, 90),
                             users={name: (positions[name], tables[name].clone()) for name in live},
                             window={name: (positions[name], window_tables[name]) for name in live}))
            for name in NAMES:
                positions[name] = min(FAMILY_LAST, positions[name] + rng.randrange(1, 17))
        return plan

    def run_schedule(self, plan, prestage, padded, kv_chains):
        block = self.open_block(prestage=prestage, padded=2 if padded else None, kv_chains=kv_chains)
        served = []
        for number, spec in enumerate(plan):
            if prestage and number:
                self.window(block, spec['window'], finished=[name for name in NAMES if name not in spec['live']])
            predictions, metrics, staged = self.round(block, spec['users'], spec['token_base'])
            if prestage:
                self.assert_device_holds(block, staged)
            served.append((predictions, metrics['segments']))
        block.close()
        self.block = None
        return served

    def test_random_rounds_stage_exactly_what_stage_packed_writes(self):
        for seed in range(8):
            padded, kv_chains = seed % 2 == 1, seed % 4 >= 2
            with self.subTest(seed=seed, padded=padded, kv_chains=kv_chains):
                plan = self.schedule(random.Random(seed), rounds=6, padded=padded)
                self.h1a.clear()
                on = self.run_schedule(plan, True, padded, kv_chains)
                paths = self.paths()
                off = self.run_schedule(plan, False, padded, kv_chains)
                self.assertEqual(on, off)
                self.assertEqual(paths, ['full'] + ['diff'] * (len(plan) - 1))


class DiffTests(PrestageFixture):
    def test_the_first_round_is_full_then_a_window_makes_the_next_a_diff_of_the_tokens_alone(self):
        block = self.open_block()
        users = base_users()
        self.round(block, users)
        self.assertEqual((self.paths(), self.reasons()), (['full'], ['no-snapshot']))
        synced = self.ttnn.synchronized
        self.round(block, advanced(users, 2))  # a full round: F1 and the last commit's fence
        full_fences = self.ttnn.synchronized - synced
        users = advanced(users, 2)
        nxt = advanced(users, 5)
        self.window(block, nxt)
        self.assertEqual(len(self.marked(verify_prestage.WINDOW_MARKER)), 1)
        self.assertRegex(self.marked(verify_prestage.WINDOW_MARKER)[0],
                         r'^\[PACKED-PRESTAGE-WINDOW\] round=3 buffers=[0-9]+ ms=[0-9.]+ live=4$')
        synced = self.ttnn.synchronized
        written = self.written(lambda: self.round(block, nxt))
        self.assertEqual(written[0], block.fixture.tokens)
        self.assertEqual(written[1:], [], 'nothing but the tokens differs')
        self.assertEqual(self.paths()[-1], 'diff')
        self.assertEqual(self.ttnn.synchronized - synced, full_fences - 1, 'no F1')

    def test_the_diff_skips_validate_bindings_which_the_window_ran(self):
        block = self.open_block()
        users = base_users()
        self.round(block, users)
        calls = []
        original = block.validate_bindings
        block.validate_bindings = lambda: calls.append('validate') or original()
        nxt = advanced(users, 3)
        self.window(block, nxt)
        self.assertEqual(calls, ['validate'])
        self.round(block, nxt)
        self.assertEqual(calls, ['validate'], 'the diff verify validated nothing more')
        self.round(block, advanced(nxt, 2))
        self.assertEqual(calls, ['validate', 'validate'], 'a full verify validates as today')

    def test_a_page_append_rewrites_that_users_page_buffers_and_the_block_wide_tables(self):
        block = self.open_block()
        users = base_users()
        self.round(block, users)
        nxt = advanced(users, 7)
        self.window(block, nxt)
        appended = dict(nxt)
        position, pages = appended['C']
        pages = pages.clone()
        pages[0, (position + 15) // 64 + 1] = 99
        appended['C'] = (position, pages)
        written = self.written(lambda: self.round(block, appended))
        fixture = block.fixture
        segment = 2  # C is admitted through slot 2
        rows = range(16 * segment, 16 * segment + 16)
        tile = next(tile for tile in fixture.cache_tiles if tile.rows[0] <= rows[0] < tile.rows[1])
        reader = fixture.replay_reader.readers[segment]
        expected = [fixture.tokens, fixture.pages, *[fixture.row_pages[row] for row in rows], tile.pages,
                    *[entry[1] for entry in reader.metadata]]
        self.assertEqual([id(value) for value in written], [id(value) for value in expected])
        self.assertEqual(self.paths()[-1], 'diff')
        self.assertIn('path=diff buffers=%d ' % len(expected), self.marked(verify_prestage.MARKER)[-1])

    def test_a_snapshot_serves_one_verify(self):
        block = self.open_block()
        users = base_users()
        self.round(block, users)
        self.window(block, advanced(users, 2))
        self.round(block, advanced(users, 2))
        self.round(block, advanced(users, 4))
        self.assertEqual(self.paths(), ['full', 'diff', 'full'])
        self.assertEqual(self.reasons()[-1], 'no-snapshot')

    def test_the_verify_metrics_carry_the_split_only_under_the_flag(self):
        block = self.open_block()
        users = base_users()
        predictions, metrics, staged = self.round(block, users)
        self.assertEqual(metrics['prestage']['path'], 'full')
        self.window(block, advanced(users, 1))
        predictions, metrics, staged = self.round(block, advanced(users, 1))
        self.assertEqual((metrics['prestage']['path'], metrics['prestage']['buffers']), ('diff', 1))
        self.assertGreater(metrics['prestage']['prestage_ms'], 0.0)
        self.assertEqual(metrics['staged_buffers'], 1)
        off = self.open_block(prestage=False)
        predictions, metrics, staged = self.round(off, users)
        self.assertNotIn('prestage', metrics)


class EpochTests(PrestageFixture):
    """Every other writer of fixture inputs bumps the epoch: the next verify takes the full path."""

    def bumped(self, writer):
        block = self.open_block()
        users = base_users()
        self.round(block, users)
        nxt = advanced(users, 3)
        self.window(block, nxt)
        writer(block, nxt)
        self.round(block, nxt)
        return self.paths()[-1], self.reasons()[-1]

    def test_the_full_stage_and_the_padded_probes_restage(self):
        def restage(block, nxt):
            import padded_probe

            padded_probe.stage(block, self.staged_users(block, self.entries(nxt)))

        self.assertEqual(self.bumped(restage), ('full', 'epoch:stage_packed'))

    def test_a_prefill_and_a_lever_n_chunk_through_the_lifecycle(self):
        import serving_lifecycle

        self.assertEqual(self.bumped(lambda block, nxt: serving_lifecycle.note_prefill()), ('full', 'epoch:prefill'))
        from unittest.mock import MagicMock

        lifecycle = SimpleNamespace(request_id='E', capture=MagicMock(), original_execute=Mock(return_value='chunk'),
                                    _displace_after_continuation=Mock(), _after_prefill_chunk=Mock(return_value='out'))
        scheduled = SimpleNamespace(scheduled_cached_reqs=SimpleNamespace(req_ids=['E'], num_computed_tokens=[2048]),
                                    total_num_scheduled_tokens=2048)
        self.assertEqual(self.bumped(lambda block, nxt: serving_lifecycle.FastServingLifecycle._continue_prefill(
            lifecycle, scheduled)), ('full', 'epoch:prefill-chunk'))
        lifecycle.original_execute.assert_called_with(scheduled)

    def test_the_hooks_pass_throughs_admissions_and_detaches(self):
        hook_class = serving_worker_hook.FastWorkerHook
        runner = SimpleNamespace(_pending_samples=())

        def hook():
            return SimpleNamespace(closed=False, runner=runner, bridges={'A': object()}, packed_step=object(),
                                   original_execute=Mock(return_value='stock'))

        new = SimpleNamespace(scheduled_new_reqs=[object()], total_num_scheduled_tokens=2048)
        chunk = SimpleNamespace(scheduled_new_reqs=[], total_num_scheduled_tokens=2048,
                                scheduled_cached_reqs=SimpleNamespace(req_ids=['E']))
        idle = SimpleNamespace(scheduled_new_reqs=[], total_num_scheduled_tokens=0,
                               scheduled_cached_reqs=SimpleNamespace(req_ids=['A']))
        for scheduled, reason in ((new, 'prefill'), (chunk, 'prefill-chunk'), (idle, 'bookkeeping')):
            with self.subTest(reason=reason):
                owner = hook()
                self.assertEqual(self.bumped(lambda block, nxt: hook_class._execute(owner, runner, scheduled)),
                                 ('full', 'epoch:' + reason))
                owner.original_execute.assert_called_once_with(scheduled)
        bridge = SimpleNamespace(runner=runner, request=SimpleNamespace(session=SimpleNamespace(request_id='B')),
                                 close=Mock())
        owner = hook()
        self.assertEqual(self.bumped(lambda block, nxt: hook_class.attach(owner, bridge)), ('full', 'epoch:admission'))
        self.assertEqual(self.bumped(lambda block, nxt: hook_class.detach(owner, 'B')), ('full', 'epoch:detach'))

    def test_a_verify_in_between_consumes_the_snapshot(self):
        block = self.open_block()
        users = base_users()
        self.round(block, users)
        self.window(block, advanced(users, 1))
        self.round(block, advanced(users, 2))  # a different verify than the window prepared for
        self.round(block, advanced(users, 3))
        self.assertEqual(self.paths(), ['full', 'diff', 'full'])


class FailureTests(PrestageFixture):
    def failing_copies(self, after):
        original = self.ttnn.copy_host_to_device_tensor
        state = dict(copies=0)

        def copy(host, destination):
            state['copies'] += 1
            if state['copies'] > after:
                raise RuntimeError('copy failed')
            original(host, destination)

        self.ttnn.copy_host_to_device_tensor = copy
        return lambda: setattr(self.ttnn, 'copy_host_to_device_tensor', original)

    def test_a_failure_inside_the_prestage_leaves_the_round_on_todays_path(self):
        block = self.open_block()
        users = base_users()
        self.round(block, users)
        starts = [reader.start for reader in block.fixture.replay_reader.readers]
        nxt = advanced(users, 6)
        restore = self.failing_copies(3)
        self.window(block, nxt)  # never raises: the coordinator hands the failure to drop()
        restore()
        self.assertIsNone(block.prestaged.snapshot)
        self.assertFalse(block.fixture.replay_reader.failed, 'a pre-stage never poisons')
        self.assertEqual([reader.start for reader in block.fixture.replay_reader.readers], starts)
        self.assertRegex(self.marked(verify_prestage.WINDOW_MARKER)[-1],
                         r'^\[PACKED-PRESTAGE-WINDOW\] round=2 dropped=prestage-failed:RuntimeError detail=')
        predictions, metrics, staged = self.round(block, nxt)
        self.assertEqual((self.paths()[-1], self.reasons()[-1]), ('full', 'prestage-failed:RuntimeError'))
        self.assert_device_holds(block, staged)

    def test_a_window_the_block_cannot_serve_drops_and_says_why(self):
        block = self.open_block()
        users = base_users()
        self.round(block, users)
        self.window(block, {'A': users['A']}, finished='BCD')  # one live user: not a round of this block
        self.assertIn('dropped=prestage-failed:ValueError', self.marked(verify_prestage.WINDOW_MARKER)[-1])
        block.verify(self.entries(users))  # verified and uncommitted: not idle
        self.window(block, users)
        self.assertIn('dropped=prestage-failed:ValueError', self.marked(verify_prestage.WINDOW_MARKER)[-1])
        self.assertIsNone(block.prestaged.snapshot)


class ReaderTests(PrestageFixture):
    def test_the_window_never_moves_a_start_and_the_diff_sets_every_one(self):
        block = self.open_block()
        users = base_users()
        self.round(block, users)
        readers = block.fixture.replay_reader.readers
        before = [reader.start for reader in readers]
        nxt = advanced(users, 9)
        self.window(block, nxt)
        self.assertEqual([reader.start for reader in readers], before)
        self.assertEqual([reader.positions.value[0].item() for reader in readers], [position for position, pages in
                                                                                   nxt.values()])
        self.round(block, nxt)
        self.assertEqual([reader.start for reader in readers], [position for position, pages in nxt.values()])

    def test_a_failed_diff_write_poisons_the_readers_and_fails_the_round_as_today(self):
        outcomes = []
        for prestage in (True, False):
            block = self.open_block(prestage=prestage)
            users = base_users()
            self.round(block, users)
            nxt = advanced(users, 2)
            if prestage:
                self.window(block, nxt)
            original = self.ttnn.copy_host_to_device_tensor
            self.ttnn.copy_host_to_device_tensor = Mock(side_effect=RuntimeError('copy failed'))
            entries = self.entries(nxt)
            with self.assertRaisesRegex(RuntimeError, 'copy failed'):
                block.verify(entries)
            self.ttnn.copy_host_to_device_tensor = original
            outcomes.append((block.phase, block.fixture.replay_reader.failed,
                             [entry['request'].session.fail_verification.call_count for entry in entries]))
            block.phase = 'idle'  # let the fixture close it
            self.assertIsNone(block.prestaged.snapshot if prestage else None)
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0], ('failed', True, [1, 1, 1, 1]))


class PaddedTests(PrestageFixture):
    def test_a_three_live_window_prestages_the_idle_fill_and_the_verify_diffs_it(self):
        block = self.open_block(padded=2)
        users = base_users()
        self.round(block, users)
        three = {name: users[name] for name in 'ABD'}
        nxt = advanced(three, 4)
        self.window(block, nxt, finished='C')
        self.assertRegex(self.marked(verify_prestage.WINDOW_MARKER)[-1], r' live=3$')
        written = self.written(lambda: self.round(block, nxt))
        predictions, metrics, staged = None, None, self.staged_users(block, self.entries(nxt))
        self.assertEqual(self.paths()[-1], 'diff')
        self.assertEqual(written, [block.fixture.tokens])
        self.assert_device_holds(block, staged)
        idle = block.idle_inputs((0, 1, 3))[2]
        self.assertEqual(staged[2][1], idle[1])

    def test_a_window_for_four_then_a_padded_verify_rewrites_the_idle_segment(self):
        block = self.open_block(padded=2)
        users = base_users()
        self.round(block, users)
        nxt = advanced(users, 1)
        self.window(block, nxt)
        three = {name: nxt[name] for name in 'ABD'}
        predictions, metrics, staged = self.round(block, three)
        self.assertEqual(self.paths()[-1], 'diff')
        self.assert_device_holds(block, staged)


class T2Tests(PrestageFixture):
    def test_the_kv_guard_stays_at_verify_time(self):
        block = self.open_block(kv_chains=True)
        users = base_users()
        self.round(block, users)
        shared = advanced(users, 1)
        position, pages = shared['B']
        shared['B'] = (shared['A'][0], shared['A'][1].clone())  # B on A's pages and rows
        self.window(block, shared)  # the window has no guard: its pages may predate the refresh
        self.assertIsNotNone(block.prestaged.snapshot)
        copies = len(self.ttnn.host_copies)
        with self.assertRaisesRegex(ValueError, 'disjoint cache tile rows'):
            block.verify(self.entries(shared))
        self.assertEqual(len(self.ttnn.host_copies), copies, 'refused before any copy')
        self.assertTrue(any(line.startswith(verify_trace_t2.KV_SHARED + ' site=stage_packed') for line in self.h1a))
        block.phase = 'idle'

    def test_a_conflict_only_in_the_window_is_the_verifys_ground_truth_to_clear(self):
        block = self.open_block(kv_chains=True)
        users = base_users()
        self.round(block, users)
        nxt = advanced(users, 1)
        stale = dict(nxt)
        stale['B'] = (nxt['B'][0], nxt['A'][1].clone())
        self.window(block, stale)
        predictions, metrics, staged = self.round(block, nxt)
        self.assertEqual(self.paths()[-1], 'diff')
        self.assert_device_holds(block, staged)


class AuditTests(PrestageFixture):
    def test_every_write_reads_back_eight_buffers_in_rotation(self):
        block = self.open_block(audit=True)
        users = base_users()
        self.round(block, users)
        for step in range(1, 4):
            self.window(block, advanced(users, step))
            self.round(block, advanced(users, step))
        lines = self.marked(verify_prestage.AUDIT_MARKER)
        self.assertEqual(len(lines), 4)
        firsts = [int(re.search(r'first=([0-9]+)', line).group(1)) for line in lines]
        self.assertEqual(firsts, [0, 8, 16, 24])
        self.assertTrue(all(' checked=8 ' in line and line.endswith('mismatches=0') for line in lines))
        self.assertEqual(block.prestaged.counts['mismatches'], 0)

    def test_first_names_where_the_rotation_starts_when_it_wraps(self):
        block = self.open_block(audit=True)
        users = base_users()
        self.round(block, users)
        count = len(packed_verifier.packed_values(block.operations, block.model, block.fixture, block.shape,
                                                  self.staged_users(block, self.entries(users)), guard=False)[0])
        block.prestaged.cursor = count - 3  # three before the end, five after the wrap
        self.window(block, advanced(users, 2))
        self.round(block, advanced(users, 2))
        line = self.marked(verify_prestage.AUDIT_MARKER)[-1]
        self.assertIn(' checked=8 first=%d ' % (count - 3), line)
        self.assertEqual(block.prestaged.cursor, 5)

    def test_a_stray_write_the_diff_misses_is_found_and_restaged_before_the_trace(self):
        block = self.open_block(audit=True)
        users = base_users()
        self.round(block, users)
        nxt = advanced(users, 2)
        self.window(block, nxt)
        block.prestaged.cursor = 0  # the positions buffer (index 1) is in this round's eight
        block.fixture.positions.value = torch.full_like(block.fixture.positions.value, 7)  # no bump
        predictions, metrics, staged = self.round(block, nxt)
        line = self.marked(verify_prestage.AUDIT_MARKER)[-1]
        self.assertRegex(line, r'path=diff checked=8 first=0 mismatches=2 at=1\.0,1\.1$')
        self.assert_device_holds(block, staged)
        reference = self.open_block(prestage=False)
        self.round(reference, users)
        expected, unused, unused_staged = self.round(reference, nxt)
        self.assertEqual(predictions, expected, 'the round stayed exact')


class PlumbingTests(unittest.TestCase):
    def test_the_step_builds_a_window_only_for_one_block_with_a_flag(self):
        block = SimpleNamespace(prestaged=None, round_fences=False)
        self.assertIsNone(serving_packed_step.PackedStep(block).while_waiting([]))
        block.round_fences = True
        waiting = serving_packed_step.PackedStep(block).while_waiting(['r'])
        self.assertIsInstance(waiting, verify_prestage.WhileWaiting)
        self.assertEqual((waiting.block, waiting.requests), (block, ['r']))
        self.assertIsNone(serving_packed_step.PackedStep([block, block]).while_waiting([]))
        self.assertIsNotNone(serving_packed_step.PackedStep(SimpleNamespace(prestaged=object())).while_waiting([]))

    def hook_drafts(self, module, environ, rows=16, window=True):
        """The hook's _drafts over a fake coordinator: its prepare calls and the drafts."""
        from test_serving_worker_hook import WorkerHookTests

        case = WorkerHookTests('test_committed_block_bypasses_baseline_forward_and_sampler')
        worker, bridge, events, scheduled = case.fixture()
        made = Mock(return_value='window') if window else None
        packed_step = SimpleNamespace(proposal_rows=Mock(return_value=rows), **({'while_waiting': made} if window else {}))
        with clean_environment(**environ):
            hook = module.FastWorkerHook(worker, bridge, cancelled=lambda: False, packed_step=packed_step)
            original = hook.bridges
            bridges = {name: SimpleNamespace(
                request=SimpleNamespace(session=SimpleNamespace(request_id=name, pending=None, finished=False)),
                drafts=Mock(return_value=SimpleNamespace(req_ids=[name], draft_token_ids=[[1]]))) for name in 'ab'}
            hook.bridges = bridges
            hook._packed_coordinator = SimpleNamespace(prepare=Mock(return_value=[]), close=Mock())
            outputs = ModuleType('vllm.v1.outputs')
            outputs.DraftTokenIds = SimpleNamespace
            try:
                with patch.dict('sys.modules', {'vllm.v1.outputs': outputs}):
                    worker.take_draft_token_ids()
            finally:
                hook.bridges = original
            prepare = hook._packed_coordinator.prepare
            hook.close()
        return ([(call.args[0] is not None and [b.request.session.request_id for b in call.args[0]], call.kwargs)
                 for call in prepare.call_args_list],
                [bridges[name].drafts.call_args_list for name in 'ab'], made and made.call_args_list)

    def test_the_hook_passes_the_window_only_for_a_block_round_with_a_flag(self):
        base = {'QWEN_FAST_PIPELINED_PROPOSALS': '1', 'QWEN_FAST_PACKED_PROPOSAL': '1'}
        calls, drafts, made = self.hook_drafts(serving_worker_hook, dict(base, QWEN_FAST_PRESTAGE='1'))
        self.assertEqual(calls, [(['a', 'b'], {'while_waiting': 'window'})])
        self.assertEqual(len(made), 1)
        calls, drafts, made = self.hook_drafts(serving_worker_hook, dict(base, QWEN_FAST_ROUND_FENCES='1',
                                                                         QWEN_FAST_PAIRS_PACKED_ONLY='1'))
        self.assertEqual(calls, [(['a', 'b'], {'packed_round': True, 'while_waiting': 'window'})])
        calls, drafts, made = self.hook_drafts(serving_worker_hook, dict(base, QWEN_FAST_PRESTAGE='1'), rows=None)
        self.assertEqual((calls, made), ([(['a', 'b'], {})], []))
        calls, drafts, made = self.hook_drafts(serving_worker_hook, base)
        self.assertEqual((calls, made), ([(['a', 'b'], {})], []))

    def test_flag_off_the_hooks_drafts_are_the_parents_call_for_call(self):
        parent = parent_module('serving_worker_hook.py')
        if parent is None:
            self.skipTest('no git history for %s' % PARENT)
        base = {'QWEN_FAST_PIPELINED_PROPOSALS': '1', 'QWEN_FAST_PACKED_PROPOSAL': '1'}
        for environ in (base, dict(base, QWEN_FAST_PAIRS_PACKED_ONLY='1')):
            for rows in (16, None):
                with self.subTest(environ=environ, rows=rows):
                    self.assertEqual(self.hook_drafts(serving_worker_hook, environ, rows, window=False),
                                     self.hook_drafts(parent, environ, rows, window=False))

    def test_flag_off_the_hooks_pass_throughs_are_the_parents(self):
        parent = parent_module('serving_worker_hook.py')
        if parent is None:
            self.skipTest('no git history for %s' % PARENT)
        runner = SimpleNamespace(_pending_samples=())
        new = SimpleNamespace(scheduled_new_reqs=[object()], total_num_scheduled_tokens=2048)
        for module in (serving_worker_hook, parent):
            owner = SimpleNamespace(closed=False, runner=runner, bridges={}, original_execute=Mock(return_value='stock'))
            with clean_environment():
                self.assertEqual(module.FastWorkerHook._execute(owner, runner, new), 'stock')
            owner.original_execute.assert_called_once_with(new)


class GateTests(unittest.TestCase):
    ON = {'QWEN_FAST_PRESTAGE': '1', 'QWEN_FAST_PRESTAGE_AUDIT': '1', 'QWEN_FAST_ROUND_FENCES': '1',
          'QWEN_FAST_PACKED_AUDIT': '1', 'QWEN_FAST_PACKED_PROPOSAL': '1', 'QWEN_FAST_PIPELINED_PROPOSALS': '1'}

    def log(self, rounds=12, full=(1,), mismatch=None, fence='f9'):
        lines = ['[PINDIAG] verify prestage engaged users=4 audit=1', '[PINDIAG] round fences engaged users=4']
        for number in range(1, rounds + 1):
            path = 'full' if number in full else 'diff'
            if number > 1:
                lines.append('[PACKED-PRESTAGE-WINDOW] round=%d buffers=143 ms=3.10 live=4' % number)
            lines.append('[PACKED-PRESTAGE] round=%d path=%s buffers=%d reason=%s live=4' % (
                number, path, 144 if path == 'full' else 1, 'no-snapshot' if path == 'full' else '-'))
            lines.append('[PACKED-PRESTAGE-AUDIT] round=%d path=%s checked=8 first=0 mismatches=%d' % (
                number, path, 2 if number == mismatch else 0))
            lines.append('[PACKED-FENCES] round=%d fence=%s validated=1 replay_ms=0.00 commit_sync_ms=0.03 path=%s '
                         'prestage_ms=3.10 diff_ms=0.20 write_ms=0.40' % (number, 'first' if number == 1 else fence, path))
        return chr(10).join(lines)

    def test_a_clean_arm_passes_and_is_summarised(self):
        import lever_n_m3native_gate as gate

        report = gate.flag_marker_report(self.ON, 4, self.log())
        self.assertEqual(report['missing'], [])
        summary = report['round_fence_h1a']
        self.assertEqual((summary['rounds'], summary['diff'], summary['full'], summary['four_live']), (12, 11, 1, 12))
        self.assertEqual((summary['audit_mismatches'], summary['fence_kinds']), (0, {'first': 1, 'f9': 11}))
        self.assertEqual(summary['full_reasons'], {'no-snapshot': 1})

    def test_missing_lines_a_mismatch_a_low_diff_share_or_no_f9_fail(self):
        import lever_n_m3native_gate as gate

        missing = gate.flag_marker_report(self.ON, 4, '')['missing']
        for marker in (gate.PRESTAGE_ENGAGED_MARKER, gate.PRESTAGE_WINDOW_MARKER, gate.PRESTAGE_MARKER,
                       gate.PRESTAGE_AUDIT_MARKER, gate.FENCES_ENGAGED_MARKER, gate.FENCES_MARKER):
            self.assertTrue(any(marker in line for line in missing), marker)
        self.assertTrue(any('no mismatch' in line for line in gate.flag_marker_report(self.ON, 4, self.log(mismatch=5))['missing']))
        low = gate.flag_marker_report(self.ON, 4, self.log(full=range(1, 6)))['missing']
        self.assertTrue(any('diff path in 7 of 12 four-live rounds' in line for line in low), low)
        none = gate.flag_marker_report(self.ON, 4, self.log(full=range(1, 13)))['missing']
        self.assertTrue(any('no verify took the diff path' in line for line in none))
        nof9 = gate.flag_marker_report(self.ON, 4, self.log(fence='replay'))['missing']
        self.assertTrue(any("no replay armed by the drafts' fence" in line for line in nof9))
        lone = gate.flag_marker_report({'QWEN_FAST_PRESTAGE_AUDIT': '1'}, 4, '')['missing']
        self.assertTrue(any('audits nothing' in line for line in lone))

    def test_the_gate_reads_the_lines_the_modules_write(self):
        import lever_n_m3native_gate as gate

        self.assertEqual((gate.PRESTAGE_ENGAGED_MARKER, gate.FENCES_ENGAGED_MARKER),
                         (verify_prestage.ENGAGED_MARKER, verify_prestage.FENCES_ENGAGED_MARKER))
        for marker, module_marker in ((gate.PRESTAGE_WINDOW_MARKER, verify_prestage.WINDOW_MARKER),
                                      (gate.PRESTAGE_MARKER, verify_prestage.MARKER),
                                      (gate.PRESTAGE_AUDIT_MARKER, verify_prestage.AUDIT_MARKER),
                                      (gate.FENCES_MARKER, verify_prestage.FENCES_MARKER)):
            self.assertEqual(marker, module_marker + ' round=')
        line = serving_packed_step.FENCES_LINE.format(**serving_packed_step.fences_fields(
            dict(replay_fence='f9', validated_this_round=True, replay_fence_ms=0.0,
                 prestage=dict(path='diff', prestage_ms=3.1, diff_ms=0.2, write_ms=0.4)),
            SimpleNamespace(rounds=7, commit_block_ms=[0.0, 0.1, 0.2, 0.03]), (0, 1, 2, 3)))
        self.assertEqual(gate.FENCES_LINE.search(line).groups(),
                         ('7', 'f9', '1', '0.00', '0.03', 'diff', '3.10', '0.20', '0.40'))

    def test_flag_off_the_report_is_the_parents(self):
        import lever_n_m3native_gate as gate
        from test_acceptance_report import fixture

        parent = parent_module('lever_n_m3native_gate.py')
        if parent is None:
            self.skipTest('no git history for %s' % PARENT)
        for tag in ('v155', 'v185'):
            log, streams = fixture(tag)
            for environ in ({}, {'QWEN_FAST_PADDED_BLOCK': '1', 'QWEN_FAST_PAIR_MASK_REFRESH': '1'},
                            {'QWEN_FAST_VERIFY_T2': '1', 'QWEN_FAST_GDN_USER_BATCH': '1', 'QWEN_FAST_PACKED_AUDIT': '1'}):
                with self.subTest(tag=tag, environ=environ):
                    self.assertEqual(gate.flag_marker_report(environ, 4, log), parent.flag_marker_report(environ, 4, log))


class ParentTests(unittest.TestCase):
    """With every flag off, the modules H1a touches against their PARENT copies (skipped without
    git history): test_padded_block's three parent comparisons re-pointed at PARENT, and the
    verifier's rounds call for call (test_padded_probe's FlagOffTests record)."""

    def repointed(self, name):
        import test_padded_block

        original = test_padded_block.parent_module
        if parent_module('packed_verifier.py') is None:
            self.skipTest('no git history for %s' % PARENT)
        case = test_padded_block.ParentTests(name)
        with patch.object(test_padded_block, 'parent_module', lambda relative, commit=None: original(relative, PARENT)):
            getattr(case, name)()

    def test_the_steps_answers_and_rounds_are_the_parents(self):
        self.repointed('test_the_steps_answers_and_rounds_are_the_parents')

    def test_the_real_blocks_round_is_the_parents(self):
        self.repointed('test_the_real_blocks_round_is_the_parents')

    def test_the_verifier_rounds_are_the_parents_call_for_call(self):
        self.repointed('test_the_verifier_rounds_are_the_parents_call_for_call')

    def test_the_lifecycles_prefill_note_still_displaces_the_resident_engine(self):
        import serving_lifecycle

        verifier_engine._resident = object()
        serving_lifecycle.note_prefill()
        self.assertIsNone(verifier_engine._resident)


class ArmTests(unittest.TestCase):
    """M3NATIVE_PRESTAGE, _PRESTAGE_AUDIT and _ROUND_FENCES cross as QWEN_FAST_<name>=1 right after
    the padded block's lines; the arm refuses what could not engage, before the docker run."""

    LINES = ('${M3NATIVE_PRESTAGE:+-e QWEN_FAST_PRESTAGE=1}',
             '${M3NATIVE_PRESTAGE_AUDIT:+-e QWEN_FAST_PRESTAGE_AUDIT=1}',
             '${M3NATIVE_ROUND_FENCES:+-e QWEN_FAST_ROUND_FENCES=1}')

    @staticmethod
    def text():
        from test_m3native_arm_env import arm_text

        return arm_text()

    def bash(self, script, **environ):
        bash = shutil.which('bash')
        if bash is None:
            self.skipTest('no bash')
        try:
            return subprocess.run([bash, '-c', script], env=dict(PATH=os.environ.get('PATH', ''), **environ),
                                  capture_output=True, text=True, timeout=60)
        except OSError as error:
            self.skipTest('bash unusable: %s' % error)

    def test_the_three_switches_cross_after_the_padded_block_before_the_entrypoint(self):
        text = self.text()
        lines = text.split(chr(10))
        start = next(number for number, line in enumerate(lines) if self.LINES[0] in line)
        self.assertEqual(lines[start - 1].strip(), '${M3NATIVE_PADDED_BLOCK_MIN_USERS:+-e QWEN_FAST_PADDED_BLOCK_MIN_USERS='
                                                   '$M3NATIVE_PADDED_BLOCK_MIN_USERS} ' + chr(92))
        for offset, expected in enumerate(self.LINES):
            with self.subTest(line=expected):
                self.assertEqual(text.count(expected), 1)
                self.assertEqual(lines[start + offset].strip(), expected + ' ' + chr(92))
                self.assertLess(text.index(expected), text.index('--entrypoint python3'))
        self.assertIn('${M3NATIVE_PIPELINED_PUBLISH:+-e QWEN_FAST_PIPELINED_PUBLISH=1}', text)

    def test_unset_nothing_crosses(self):
        script = 'printf "%s|" ' + ' '.join(self.LINES) + chr(10)
        self.assertEqual(self.bash(script).stdout.strip('|'), '')
        self.assertEqual(self.bash(script, M3NATIVE_PRESTAGE='1', M3NATIVE_PRESTAGE_AUDIT='1', M3NATIVE_ROUND_FENCES='1').stdout,
                         '-e|QWEN_FAST_PRESTAGE=1|-e|QWEN_FAST_PRESTAGE_AUDIT=1|-e|QWEN_FAST_ROUND_FENCES=1|')

    def test_the_arm_refuses_what_could_not_engage(self):
        lines = self.text().split(chr(10))
        start = next(number for number, line in enumerate(lines) if line.startswith('# Round-fence plan H1a'))
        end = next(number for number in range(start, len(lines)) if lines[number] == 'fi'
                   and lines[number - 1].startswith('  echo "round-fence plan H1a'))
        script = chr(10).join(['users="${M3NATIVE_USERS:-4}"'] + lines[start:end + 1] + ['echo passed'])
        window = dict(M3NATIVE_PACKED_PROPOSAL='1', M3NATIVE_PIPELINED_PROPOSALS='1')
        cases = [(dict(), 0), (dict(window, M3NATIVE_PRESTAGE='1'), 0),
                 (dict(window, M3NATIVE_PRESTAGE='1', M3NATIVE_PRESTAGE_AUDIT='1', M3NATIVE_ROUND_FENCES='1'), 0),
                 (dict(M3NATIVE_ROUND_FENCES='1'), 0),
                 (dict(window, M3NATIVE_PRESTAGE='yes'), 1), (dict(M3NATIVE_ROUND_FENCES='0'), 1),
                 (dict(window, M3NATIVE_PRESTAGE_AUDIT='1'), 1),
                 (dict(M3NATIVE_PRESTAGE='1'), 1), (dict(M3NATIVE_PRESTAGE='1', M3NATIVE_PACKED_PROPOSAL='1'), 1),
                 (dict(window, M3NATIVE_PRESTAGE='1', M3NATIVE_USERS='1'), 1),
                 (dict(M3NATIVE_ROUND_FENCES='1', M3NATIVE_SEQUENTIAL_USERS='4'), 1)]
        for environ, code in cases:
            with self.subTest(environ=environ):
                result = self.bash(script, **environ)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual('passed' in result.stdout, code == 0)

    def test_the_names_are_read_where_the_arm_sends_them(self):
        for name, modules in (('QWEN_FAST_PRESTAGE', ('verify_prestage.py', 'serving_worker_hook.py',
                                                      'lever_n_m3native_gate.py')),
                              ('QWEN_FAST_PRESTAGE_AUDIT', ('verify_prestage.py', 'lever_n_m3native_gate.py')),
                              ('QWEN_FAST_ROUND_FENCES', ('verify_prestage.py', 'serving_worker_hook.py',
                                                          'lever_n_m3native_gate.py'))):
            for module in modules:
                with self.subTest(name=name, module=module):
                    self.assertIn("'%s'" % name, (HERE / module).read_text(encoding='utf-8'))


class ShippingTests(unittest.TestCase):
    RUNTIME = ('verify_prestage.py', 'packed_verifier.py', 'gdn_records.py', 'serving_packed_step.py',
               'serving_worker_hook.py', 'serving_lifecycle.py', 'dflash_packed_proposal_coordinator.py')

    def test_every_runtime_module_h1a_touches_is_in_both_image_copy_lists(self):
        from test_serving_image_copy_closure import context_modules, dockerfile_modules, dockerfile_text

        docker, context = dockerfile_modules(dockerfile_text()), context_modules()
        for name in self.RUNTIME:
            with self.subTest(module=name):
                self.assertIn(name, docker)
                self.assertIn(name, context)

    def test_the_suites_run_in_the_cpu_workflow(self):
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text(encoding='utf-8')
        self.assertRegex(workflow, r'python -B -m unittest [^\n]*\btest_verify_prestage\b[^\n]*\btest_round_fences\b')

    def test_the_new_files_are_lf(self):
        for name in ('verify_prestage.py', 'test_verify_prestage.py', 'test_round_fences.py'):
            with self.subTest(name=name):
                self.assertNotIn(b'\r\n', (HERE / name).read_bytes())


if __name__ == '__main__':
    unittest.main()
