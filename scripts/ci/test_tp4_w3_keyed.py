"""tp4/w3 lever 1b: the keyed staging diff (QWEN_FAST_TP4_PRESTAGE_KEYED, default off, byte-identical off).

The pre-stage's snapshot also keeps the KEY its values were computed from (each segment's start and a clone of its page table, and the identity of
every staging destination holder). A verify whose key equals it builds the tokens alone and writes the tokens buffer alone; any other key takes
the value diff, unchanged. Here, on test_verify_prestage's four-user block with its fake device model (every replay computes its outputs from what is
staged, so a wrong staged byte is a wrong prediction too):

  - the flag grammar, and that off nothing of the lever exists (lines, counts, `last`, writes, fences are the control's);
  - a keyed round writes the tokens alone and leaves the device holding exactly what stage_packed writes, over random schedules and padded rounds,
    with the predictions the flag-off run's; its key is checked on every way to differ (a start, one table entry, the live set, the padded fill,
    a destination holder, an epoch bump) and a differing key takes the value diff;
  - the checks the full build makes still run (the T2 K/V guard on the snapshot's own values, each reader's validate, the tokens' signature) and
    the helper that builds the tokens is the one packed_host_inputs calls;
  - the audits: the host comparison of every key-derived value (QWEN_FAST_PRESTAGE_AUDIT or the full audit), the full read-back on a keyed round
    and the restage before the trace; the extent readers' block too;
  - the markers and the smoke rules (w3_keyed_smoke), and that the existing parsers still read the lines the lever extends."""

import ast
import os
from pathlib import Path
import random
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import c2_smoke_check
import packed_verifier
import test_packed_extent_block as tpe
import test_tp4_hostgap as hostgap
import test_verify_prestage as tvp
import verifier_inputs
import verify_prestage
import verify_trace_t2
import w3_keyed_smoke

HERE = Path(__file__).resolve().parent
FLAG = 'QWEN_FAST_TP4_PRESTAGE_KEYED'
NAMES = tvp.NAMES
PAGE_WIDTH = tvp.PAGE_WIDTH


def normalised(lines):
    """Log lines with their timings blanked: two runs of the same schedule differ only there."""
    return [re.sub(r'\d+\.\d+', '#', line) for line in lines if not line.startswith('[PINDIAG] gc ')]


class KeyedFixture(tvp.PrestageFixture):
    def open_keyed(self, keyed=True, audit=False, full_audit=False, padded=None, kv_chains=False, hostgap_log=False, flag=None):
        for name in (FLAG, 'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT', 'QWEN_FAST_TP4_HOSTGAP_LOG'):
            os.environ.pop(name, None)
        if keyed:
            os.environ[FLAG] = '1'
        elif flag is not None:
            os.environ[FLAG] = flag
        if full_audit:
            os.environ['QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT'] = '1'
        if hostgap_log:
            os.environ['QWEN_FAST_TP4_HOSTGAP_LOG'] = '1'
        return self.open_block(prestage=True, audit=audit, padded=padded, kv_chains=kv_chains)

    def prestage_lines(self):
        return self.marked(verify_prestage.MARKER)

    def keyed_rounds(self):
        return [' keyed=1' in line for line in self.prestage_lines()]

    def stale_then_round(self, block, window, verify):
        self.window(block, window)
        return self.round(block, verify)

    def ready(self, **options):
        """A block that has served its first (full) round, the users it served, and the next round's frontier."""
        block = self.open_keyed(**options)
        users = tvp.base_users()
        self.round(block, users)
        return block, users, tvp.advanced(users, 3)


class FlagTests(unittest.TestCase):
    def test_the_flag_is_zero_or_one_off_by_default_and_inert_without_the_prestage(self):
        self.assertFalse(verify_prestage.keyed_enabled({}))
        self.assertFalse(verify_prestage.keyed_enabled({FLAG: '0', 'QWEN_FAST_PRESTAGE': '1'}))
        self.assertTrue(verify_prestage.keyed_enabled({FLAG: '1', 'QWEN_FAST_PRESTAGE': '1'}))
        self.assertFalse(verify_prestage.keyed_enabled({FLAG: '1'}), 'alone it keys nothing')
        for value in ('yes', '2', 'true', ''):
            with self.subTest(value=value), self.assertRaises(ValueError):
                verify_prestage.keyed_enabled({FLAG: value, 'QWEN_FAST_PRESTAGE': '1'})
        self.assertEqual(verify_prestage.KEYED_FLAG, FLAG)

    def test_a_bad_value_raises_at_the_attach_of_several_blocks_beside_the_other_hostgap_flags(self):
        blocks = [hostgap.stub_block('A'), hostgap.stub_block('B')]
        environ = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
        with patch.object(verify_prestage, 'log_line'):
            self.assertIsNone(verify_prestage.engage_two_block(blocks, dict(environ, **{FLAG: '1'})))
            with self.assertRaises(ValueError):
                verify_prestage.engage_two_block(blocks, dict(environ, **{FLAG: 'yes'}))

    def test_a_bad_value_raises_when_the_block_is_built(self):
        with patch.dict(os.environ, {'QWEN_FAST_PRESTAGE': '1', FLAG: 'maybe'}):
            with self.assertRaises(ValueError):
                verify_prestage.BlockPrestage(SimpleNamespace(), audit=False)

    def test_the_flag_is_not_one_of_the_hostgap_family_the_other_suites_enumerate(self):
        self.assertNotIn(FLAG, verify_prestage.HOSTGAP_FLAGS)


class FlagOffTests(KeyedFixture):
    """With the flag unset or 0 nothing of the lever exists: the same writes, fences, log lines, counts and `last` as the control."""

    def schedule_run(self, flag):
        plan = tvp.EquivalenceTests.schedule(None, random.Random(11), 6, True)
        block = self.open_keyed(keyed=False, flag=flag, padded=2, hostgap_log=True)
        self.h1a.clear()
        copies, synced = len(self.ttnn.host_copies), self.ttnn.synchronized
        served = []
        for number, spec in enumerate(plan):
            if number:
                self.window(block, spec['window'], finished=[name for name in NAMES if name not in spec['live']])
            predictions, metrics, staged = self.round(block, spec['users'], spec['token_base'])
            served.append((predictions, metrics['segments'], dict(metrics['prestage'], prestage_ms=0, diff_ms=0, write_ms=0)))
        writes = [(host.dtype, tuple(host.shape), destination.name if hasattr(destination, 'name') else id(destination) and 'd')
                  for host, destination in self.ttnn.host_copies[copies:]]
        result = (served, writes, self.ttnn.synchronized - synced, normalised(self.h1a), dict(block.prestaged.counts),
                  sorted(block.prestaged.last))
        block.close()
        self.block = None
        return result

    def test_unset_and_zero_give_the_same_run_and_no_keyed_text_anywhere(self):
        unset, zero = self.schedule_run(None), self.schedule_run('0')
        self.assertEqual(unset, zero)
        served, writes, fences, lines, counts, last = unset
        self.assertFalse([line for line in lines if 'keyed' in line.lower()])
        self.assertNotIn('keyed', ''.join(counts))
        self.assertEqual(last, ['buffers', 'diff_ms', 'path', 'prestage_ms', 'reason', 'write_ms'])
        self.assertTrue(any(' path=diff ' in line for line in lines))

    def test_the_block_state_is_the_controls(self):
        block = self.open_keyed(keyed=False)
        self.assertFalse(block.prestaged.keyed)
        self.assertNotIn('keyed', block.prestaged.counts)
        self.assertEqual(sorted(block.prestaged.last), ['buffers', 'diff_ms', 'path', 'prestage_ms', 'reason', 'write_ms'])
        self.assertFalse([line for line in self.h1a if 'keyed' in line.lower()])
        users = tvp.base_users()
        self.round(block, users)
        self.window(block, tvp.advanced(users, 2))
        self.assertIsNone(block.prestaged.snapshot.key, 'no key is kept off')

    def test_the_value_diff_is_what_runs_off(self):
        block, users, nxt = self.ready(keyed=False)
        self.window(block, nxt)
        with patch.object(verify_prestage.BlockPrestage, 'keyed_values') as keyed:
            written = self.written(lambda: self.round(block, nxt))
        keyed.assert_not_called()
        self.assertEqual(written, [block.fixture.tokens])
        self.assertEqual(self.paths()[-1], 'diff')


class KeyedRoundTests(KeyedFixture):
    def test_a_keyed_round_writes_the_tokens_alone_and_the_device_holds_the_full_stage(self):
        block, users, nxt = self.ready()
        self.window(block, nxt)
        staged_users = self.staged_users(block, self.entries(nxt))
        with patch.object(packed_verifier, 'packed_values', side_effect=AssertionError('no full build on a keyed round')) as full:
            written = self.written(lambda: self.round(block, nxt))
        full.assert_not_called()
        self.assertEqual(written, [block.fixture.tokens])
        self.assertEqual(self.paths()[-1], 'diff')
        line = self.prestage_lines()[-1]
        self.assertRegex(line, r'^\[PACKED-PRESTAGE\] round=2 path=diff buffers=1 reason=- live=4 keyed=1$')
        self.assert_device_holds(block, staged_users)
        self.assertEqual((block.prestaged.counts['keyed'], block.prestaged.counts['diff']), (1, 1))
        self.assertEqual(block.prestaged.last['keyed'], 1)
        self.assertEqual(self.keyed_rounds(), [False, True])

    def test_the_first_round_and_a_round_after_a_consumed_snapshot_are_full_and_unkeyed(self):
        block, users, nxt = self.ready()
        self.round(block, nxt)
        self.assertEqual((self.paths(), self.keyed_rounds()), (['full', 'full'], [False, False]))
        self.assertEqual(block.prestaged.last['keyed'], 0)

    def test_the_keyed_diff_skips_validate_bindings_and_the_fence_like_the_value_diff(self):
        counts = {}
        for keyed in (True, False):
            block, users, nxt = self.ready(keyed=keyed)
            calls = []
            original = block.validate_bindings
            block.validate_bindings = lambda original=original, calls=calls: calls.append('validate') or original()
            self.window(block, nxt)
            synced = self.ttnn.synchronized
            self.round(block, nxt)
            counts[keyed] = (calls, self.ttnn.synchronized - synced)
        self.assertEqual(counts[True], counts[False])
        self.assertEqual(counts[True][0], ['validate'], 'the window validated; the verify did not')

    def test_random_schedules_stage_exactly_what_stage_packed_writes_and_predict_what_the_value_diff_predicts(self):
        keyed_total = 0
        for seed in range(8):
            padded, kv_chains = seed % 2 == 1, seed % 4 >= 2
            with self.subTest(seed=seed, padded=padded, kv_chains=kv_chains):
                plan = tvp.EquivalenceTests.schedule(None, random.Random(seed), 6, padded)
                runs = {}
                for keyed in (True, False):
                    block = self.open_keyed(keyed=keyed, padded=2 if padded else None, kv_chains=kv_chains)
                    self.h1a.clear()
                    served = []
                    for number, spec in enumerate(plan):
                        if number:
                            self.window(block, spec['window'], finished=[name for name in NAMES if name not in spec['live']])
                        predictions, metrics, staged = self.round(block, spec['users'], spec['token_base'])
                        if number:
                            self.assert_device_holds(block, staged)
                        served.append((predictions, metrics['segments']))
                    runs[keyed] = (served, self.paths(), self.keyed_rounds())
                    block.close()
                    self.block = None
                self.assertEqual(runs[True][0], runs[False][0])
                self.assertEqual(runs[True][1], runs[False][1])
                expected = [False] + [all(spec['window'][name][0] == spec['users'][name][0]
                                          and torch.equal(spec['window'][name][1], spec['users'][name][1]) for name in spec['live'])
                                      for spec in plan[1:]]
                self.assertEqual(runs[True][2], expected)
                self.assertEqual(runs[False][2], [False] * len(plan))
                keyed_total += sum(expected)
        self.assertGreater(keyed_total, 4, 'the schedules exercise the keyed path')

    def test_a_padded_window_for_four_then_the_same_three_live_is_keyed_only_when_the_fill_matches(self):
        block = self.open_keyed(padded=2)
        users = tvp.base_users()
        self.round(block, users)
        nxt = tvp.advanced(users, 1)
        three = {name: nxt[name] for name in 'ABD'}
        self.window(block, three, finished=['C'])
        predictions, metrics, staged = self.round(block, three)
        self.assertTrue(self.keyed_rounds()[-1], 'the same three live, the same idle fill: keyed')
        self.assert_device_holds(block, staged)
        # a window for four, a verify of three: the idle fill replaces a live user's start and table
        nxt2 = tvp.advanced(nxt, 1)
        self.window(block, nxt2)
        predictions, metrics, staged = self.round(block, {name: nxt2[name] for name in 'ABD'})
        self.assertFalse(self.keyed_rounds()[-1])
        self.assertEqual(self.paths()[-1], 'diff')
        self.assert_device_holds(block, staged)
        # a window for three (C idle), a verify of a different three (B idle): every fill moves
        nxt3 = tvp.advanced(nxt2, 1)
        self.window(block, {name: nxt3[name] for name in 'ABD'}, finished=['C'])
        predictions, metrics, staged = self.round(block, {name: nxt3[name] for name in 'ACD'})
        self.assertFalse(self.keyed_rounds()[-1])
        self.assert_device_holds(block, staged)


class KeyBreakerTests(KeyedFixture):
    """Every way the key can differ takes the value diff, and the value diff leaves the device holding the full stage."""

    def check_not_keyed(self, block, verify, path='diff', reason='-'):
        staged = self.staged_users(block, self.entries(verify))
        self.round(block, verify)
        self.assertFalse(self.keyed_rounds()[-1])
        self.assertEqual(self.paths()[-1], path)
        self.assertEqual(self.reasons()[-1], reason)
        self.assertEqual(block.prestaged.last['keyed'], 0)
        if path == 'diff':
            self.assert_device_holds(block, staged)

    def test_a_start_that_moved(self):
        block, users, nxt = self.ready()
        self.window(block, nxt)
        moved = dict(nxt)
        moved['C'] = (nxt['C'][0] + 1, nxt['C'][1])
        self.check_not_keyed(block, moved)
        self.assertEqual(block.prestaged.counts['keyed'], 0)

    def test_one_page_table_entry_that_moved(self):
        block, users, nxt = self.ready()
        self.window(block, nxt)
        appended = dict(nxt)
        position, pages = appended['B']
        pages = pages.clone()
        pages[0, (position + 15) // 64 + 1] = 99
        appended['B'] = (position, pages)
        self.check_not_keyed(block, appended)

    def test_a_table_of_another_dtype_with_equal_values_is_not_the_same_key(self):
        block, users, nxt = self.ready()
        self.window(block, nxt)
        wrong = dict(nxt)
        wrong['A'] = (nxt['A'][0], nxt['A'][1].to(torch.int64))
        with self.assertRaisesRegex(ValueError, 'int32 page table'):
            self.round(block, wrong)

    def test_the_live_set_that_changed(self):
        block = self.open_keyed(padded=2)
        users = tvp.base_users()
        self.round(block, users)
        nxt = tvp.advanced(users, 2)
        self.window(block, {name: nxt[name] for name in 'ABD'}, finished=['C'])
        self.check_not_keyed(block, nxt)

    def test_an_epoch_bump_between_the_window_and_the_verify_takes_the_full_stage(self):
        block, users, nxt = self.ready()
        self.window(block, nxt)
        verify_prestage.bump('test')
        with patch.object(verify_prestage.BlockPrestage, 'keyed_values') as keyed:
            self.round(block, nxt)
        keyed.assert_not_called()
        self.assertEqual((self.paths()[-1], self.reasons()[-1]), ('full', 'epoch:test'))
        self.assertFalse(self.keyed_rounds()[-1])

    def test_a_destination_holder_that_changed_takes_the_destinations_full_path_as_the_value_diff_does(self):
        results = {}
        for keyed in (True, False):
            block, users, nxt = self.ready(keyed=keyed)
            self.window(block, nxt)
            # a replaced cache tile positions buffer: a destination the snapshot did not name
            tile = block.fixture.cache_tiles[0]
            tile.positions = self.ttnn.allocate(tuple(tile.positions.shape), 'int32', 'row_major',
                                                torch.zeros(tuple(tile.positions.shape), dtype=torch.int32))
            self.round(block, nxt)
            results[keyed] = (self.paths()[-1], self.reasons()[-1], self.keyed_rounds()[-1])
        self.assertEqual(results[True], ('full', 'destinations', False))
        self.assertEqual(results[False][:2], ('full', 'destinations'))

    def test_a_reader_buffer_that_changed_is_a_different_structure(self):
        block, users, nxt = self.ready()
        self.window(block, nxt)
        key = block.prestaged.snapshot.key
        fixture = block.fixture
        self.assertTrue(verify_prestage.key_matches(key, fixture, self.staged_users(block, self.entries(nxt))))
        reader = fixture.replay_reader.readers[1]
        original = reader.metadata[0]
        reader.metadata[0] = (original[0], self.ttnn.allocate(tuple(original[1].shape), 'int32', 'row_major',
                                                              torch.zeros(tuple(original[1].shape), dtype=torch.int32)), *original[2:])
        self.assertFalse(verify_prestage.key_matches(key, fixture, self.staged_users(block, self.entries(nxt))))
        reader.metadata[0] = original
        self.assertTrue(verify_prestage.key_matches(key, fixture, self.staged_users(block, self.entries(nxt))))
        # a number the values are sliced by
        capacity = reader.capacity
        reader.capacity = capacity + 64
        self.assertFalse(verify_prestage.key_matches(key, fixture, self.staged_users(block, self.entries(nxt))))
        reader.capacity = capacity
        self.assertTrue(verify_prestage.key_matches(key, fixture, self.staged_users(block, self.entries(nxt))))

    def test_the_key_holds_a_clone_of_the_table_not_the_engines(self):
        block, users, nxt = self.ready()
        requests = self.window_requests(nxt)
        waiting = verify_prestage.WhileWaiting(block, requests)
        waiting()
        key = block.prestaged.snapshot.key
        table = requests[0].engine.pages
        before = key[1][0][1].clone()
        table[0, 0] = 12345      # vLLM's refresh at execute time writes the engine's table in place
        self.assertTrue(torch.equal(key[1][0][1], before))
        self.assertIsNot(key[1][0][1], table)

    def test_key_matches_refuses_a_missing_user_a_short_list_and_no_key(self):
        block, users, nxt = self.ready()
        self.window(block, nxt)
        key = block.prestaged.snapshot.key
        staged = self.staged_users(block, self.entries(nxt))
        fixture = block.fixture
        self.assertTrue(verify_prestage.key_matches(key, fixture, staged))
        self.assertFalse(verify_prestage.key_matches(key, fixture, staged[:3]))
        self.assertFalse(verify_prestage.key_matches(key, fixture, [None] + staged[1:]))
        self.assertFalse(verify_prestage.key_matches(None, fixture, staged))
        self.assertFalse(verify_prestage.key_matches(key, fixture, [(staged[0][0], True, staged[0][2])] + staged[1:]))


class ChecksStillRunTests(KeyedFixture):
    def test_the_kv_guard_runs_on_the_snapshots_own_values_for_a_keyed_round(self):
        results = {}
        for keyed in (True, False):
            block = self.open_keyed(keyed=keyed, kv_chains=True)
            users = tvp.base_users()
            self.round(block, users)
            shared = tvp.advanced(users, 1)
            shared['B'] = (shared['A'][0], shared['A'][1].clone())
            self.window(block, shared)
            copies = len(self.ttnn.host_copies)
            with self.assertRaisesRegex(ValueError, 'disjoint cache tile rows') as caught:
                block.verify(self.entries(shared))
            results[keyed] = str(caught.exception)
            self.assertEqual(len(self.ttnn.host_copies), copies, 'refused before any copy')
            self.assertTrue(any(line.startswith(verify_trace_t2.KV_SHARED + ' site=stage_packed') for line in self.h1a))
            block.phase = 'idle'
        self.assertEqual(results[True], results[False])

    def test_the_kv_audit_line_is_logged_on_a_keyed_round_as_on_a_value_diff(self):
        lines = {}
        for keyed in (True, False):
            with patch.dict(os.environ, {'QWEN_FAST_VERIFY_T2': '1', 'QWEN_FAST_VERIFY_T2_AUDIT': '1'}):
                block, users, nxt = self.ready(keyed=keyed, kv_chains=True)
                self.h1a.clear()
                self.window(block, nxt)
                self.round(block, nxt)
            lines[keyed] = [line for line in self.h1a if 'kv_rows_per_user' in line]
        self.assertTrue(lines[True])
        self.assertEqual(lines[True][-1], lines[False][-1])

    def test_every_readers_validate_still_runs_with_its_start_and_a_failure_raises(self):
        block, users, nxt = self.ready()
        self.window(block, nxt)
        readers = block.fixture.replay_reader.readers
        seen = []
        for index, reader in enumerate(readers):
            original = reader.validate
            reader.validate = (lambda start, original=original, index=index: seen.append((index, start)) or original(start))
        self.round(block, nxt)
        starts = [nxt[name][0] for name in NAMES]
        self.assertEqual(sorted(seen)[:4], sorted((index, starts[index]) for index in range(4))[:4])
        self.assertTrue(self.keyed_rounds()[-1])
        # a reader that refuses the start: the round raises before any copy, keyed or not
        self.window(block, tvp.advanced(nxt, 2))
        readers[2].validate = Mock(side_effect=ValueError('a start the family refuses'))
        copies = len(self.ttnn.host_copies)
        with self.assertRaisesRegex(ValueError, 'a start the family refuses'):
            block.verify(self.entries(tvp.advanced(nxt, 2)))
        self.assertEqual(len(self.ttnn.host_copies), copies)
        block.phase = 'idle'

    def test_a_token_the_vocabulary_refuses_raises_as_on_the_value_diff(self):
        for keyed in (True, False):
            block, users, nxt = self.ready(keyed=keyed)
            self.window(block, nxt)
            entries = self.entries(nxt)
            entries[0]['ticket'].tokens = tuple(range(1_000_000, 1_000_016))
            copies = len(self.ttnn.host_copies)
            with self.assertRaisesRegex(ValueError, 'Token outside target vocabulary'):
                block.verify(entries)
            self.assertEqual(len(self.ttnn.host_copies), copies)
            block.phase = 'idle'

    def test_a_tokens_buffer_whose_signature_moved_raises(self):
        block, users, nxt = self.ready()
        self.window(block, nxt)
        block.fixture.tokens.dtype = 'int32'
        with self.assertRaisesRegex(ValueError, 'preserve every captured tensor signature'):
            block.verify(self.entries(nxt))
        block.phase = 'idle'

    def test_the_helper_that_builds_the_tokens_is_the_one_both_paths_call(self):
        block, users, nxt = self.ready()
        calls = []
        real = packed_verifier.packed_host_tokens
        with patch.object(packed_verifier, 'packed_host_tokens', side_effect=lambda *args: calls.append(args) or real(*args)):
            self.window(block, nxt)
            self.assertEqual(len(calls), 1, 'the pre-stage builds through it (packed_host_inputs)')
            self.round(block, nxt)
        self.assertEqual(len(calls), 2, 'the keyed verify builds through it too, and builds nothing else')
        self.assertTrue(self.keyed_rounds()[-1])
        self.assertEqual(calls[1][2], block.model.args.vocab_size)
        source = (HERE / 'packed_verifier.py').read_text(encoding='utf-8')
        self.assertEqual(source.count('torch.tensor(user_tokens'), 1, 'one place builds a user tokens tensor')

    def test_the_factored_host_inputs_are_the_old_build(self):
        block, users, nxt = self.ready()
        shape = block.shape
        rng = random.Random(3)
        for case in range(5):
            segments = [([rng.randrange(0, 100) for row in range(shape.rows_per_user)], rng.randrange(4096, 4300),
                         tvp.user_table(rng, index)) for index in range(shape.users)]
            tokens, positions, cos, sin, pages = packed_verifier.packed_host_inputs(segments, shape, 64, 1.0e6, 1000)
            pieces = [verifier_inputs.host_inputs(list(t), s, 64, 1.0e6) for t, s, table in segments]
            self.assertTrue(torch.equal(tokens, torch.cat([piece[0] for piece in pieces])))
            self.assertEqual(tokens.dtype, torch.int32)
            self.assertTrue(torch.equal(positions, torch.cat([piece[1] for piece in pieces])))
            self.assertTrue(torch.equal(cos, torch.cat([piece[2] for piece in pieces], dim=1)))
            self.assertTrue(torch.equal(sin, torch.cat([piece[3] for piece in pieces], dim=1)))
            self.assertTrue(torch.equal(pages, torch.cat([table.repeat(shape.rows_per_user, 1) for t, s, table in segments])))
            self.assertTrue(torch.equal(tokens, packed_verifier.packed_host_tokens(segments, shape, 1000)))
        with self.assertRaisesRegex(ValueError, 'One .tokens, start, pages. per segment required'):
            packed_verifier.packed_host_tokens(segments[:3], shape, 1000)
        with self.assertRaisesRegex(ValueError, 'Token outside target vocabulary'):
            packed_verifier.packed_host_tokens([(list(range(2000, 2016)), 4100, segments[0][2])] * 4, shape, 1000)
        with self.assertRaisesRegex(ValueError, 'int32 page table'):
            packed_verifier.packed_host_tokens([(list(range(16)), 4100, segments[0][2].to(torch.int64))] * 4, shape, 1000)

    def test_packed_values_is_unchanged_by_the_factoring(self):
        block, users, nxt = self.ready(keyed=False)
        staged = self.staged_users(block, self.entries(nxt))
        values, readers = packed_verifier.packed_values(block.operations, block.model, block.fixture, block.shape, staged)
        self.assertEqual(readers, tuple(block.fixture.replay_reader.readers))
        # destinations in the order the snapshot keeps: tokens, positions, cos, sin, pages are the first five
        fixture = block.fixture
        self.assertEqual([value[0] for value in values[:5]], [fixture.tokens, fixture.positions, fixture.cos, fixture.sin, fixture.pages])


class AuditTests(KeyedFixture):
    def keyed_audit_lines(self):
        return self.marked(verify_prestage.KEYED_AUDIT_MARKER)

    def test_the_audited_arm_compares_every_value_but_the_tokens_and_finds_no_difference(self):
        block, users, nxt = self.ready(audit=True)
        self.window(block, nxt)
        self.round(block, nxt)
        (line,) = self.keyed_audit_lines()
        self.assertRegex(line, r'^\[PACKED-PRESTAGE-KEYED-AUDIT\] round=2 checked=[1-9][0-9]* keyed_mismatches=0$')
        self.assertEqual(block.prestaged.counts['keyed_mismatches'], 0)
        self.assertGreater(block.prestaged.counts['keyed_audited'], 100, 'the whole non-token list')

    def test_no_keyed_audit_line_for_an_unkeyed_round_or_an_unaudited_arm(self):
        block, users, nxt = self.ready(audit=True)
        self.round(block, nxt)      # no window: a full round
        self.assertEqual(self.keyed_audit_lines(), [])
        block, users, nxt = self.ready()
        self.window(block, nxt)
        self.round(block, nxt)
        self.assertEqual(self.keyed_audit_lines(), [])

    def test_a_value_the_key_does_not_see_is_found_and_the_round_is_restaged_in_full_before_the_trace(self):
        # the rope base is a block constant, not in the key: changed between the window and the verify, a keyed round would stage stale rotary tables
        block, users, nxt = self.ready(audit=True)
        self.window(block, nxt)
        block.model.args.rope_theta = block.model.args.rope_theta * 2
        predictions, metrics, staged = self.round(block, nxt)
        (line,) = self.keyed_audit_lines()
        self.assertRegex(line, r'keyed_mismatches=[1-9][0-9]* at=2,3')      # cos and sin are values 2 and 3
        self.assertGreater(block.prestaged.counts['keyed_mismatches'], 0)
        self.assert_device_holds(block, staged)      # restaged in full with the new base before the trace

    def test_the_full_audit_reads_every_destination_back_on_a_keyed_round_and_restages(self):
        block, users, nxt = self.ready(full_audit=True)
        self.window(block, nxt)
        staged = self.staged_users(block, self.entries(nxt))
        copies = len(self.ttnn.host_copies)
        predictions, metrics, unused = self.round(block, nxt)
        full = self.marked(verify_prestage.FULL_AUDIT_MARKER)
        keyed_lines = [line for line in full if ' path=diff ' in line]
        self.assertEqual(len(keyed_lines), 1)
        self.assertRegex(keyed_lines[0], r'^\[PACKED-PRESTAGE-FULLAUDIT\] block=- round=2 path=diff buffers=[0-9]+ checked=[0-9]+ mismatches=0 keyed=1$')
        self.assertEqual(len(self.keyed_audit_lines()), 1, 'the host comparison runs under the full audit too')
        self.assertTrue(self.keyed_rounds()[-1])
        self.assert_device_holds(block, staged)
        self.assertGreater(len(self.ttnn.host_copies) - copies, 100, 'the diff round is staged in full after the audit')

    def test_the_full_audit_finds_a_buffer_the_keyed_write_left_stale(self):
        block, users, nxt = self.ready(full_audit=True)
        self.window(block, nxt)
        block.fixture.cos.value = torch.zeros_like(block.fixture.cos.value)     # the device buffer, after the window wrote it
        predictions, metrics, staged = self.round(block, nxt)
        keyed_lines = [line for line in self.marked(verify_prestage.FULL_AUDIT_MARKER) if ' path=diff ' in line]
        self.assertRegex(keyed_lines[0], r'mismatches=[1-9][0-9]* at=2\.0,2\.1 keyed=1$')
        self.assert_device_holds(block, staged)

    def test_the_rotating_device_audit_still_runs_beside_the_keyed_one(self):
        block, users, nxt = self.ready(audit=True)
        self.window(block, nxt)
        block.prestaged.cursor = 0
        block.fixture.positions.value = torch.full_like(block.fixture.positions.value, 7)
        predictions, metrics, staged = self.round(block, nxt)
        self.assertRegex(self.marked(verify_prestage.AUDIT_MARKER)[-1], r'mismatches=2 at=1\.0,1\.1$')
        self.assertEqual(len(self.keyed_audit_lines()), 1)
        self.assert_device_holds(block, staged)


class MarkerTests(KeyedFixture):
    def test_the_engaged_line_is_logged_once_per_block_with_the_flag_and_never_without(self):
        self.open_keyed()
        self.assertEqual(len(self.marked(verify_prestage.KEYED_ENGAGED_MARKER)), 1)
        self.assertEqual(self.marked(verify_prestage.KEYED_ENGAGED_MARKER)[0], '[PINDIAG] verify prestage keyed engaged users=4 audit=0')
        self.h1a.clear()
        self.open_keyed(keyed=False)
        self.assertEqual(self.marked(verify_prestage.KEYED_ENGAGED_MARKER), [])
        self.assertEqual(len(self.marked(verify_prestage.ENGAGED_MARKER)), 1, 'the prestage line is unchanged')

    def test_the_hostgap_verify_line_ends_with_keyed_only_under_the_flag_and_the_smoke_parsers_still_match(self):
        for keyed in (True, False):
            block, users, nxt = self.ready(keyed=keyed, hostgap_log=True)
            self.h1a.clear()
            self.window(block, nxt)
            self.round(block, nxt)
            (line,) = self.marked(packed_verifier.verify_prestage.HOSTGAP_VERIFY_MARKER)
            if keyed:
                self.assertRegex(line, r' readback_cpu_ms=[0-9.]+ keyed=1$')
            else:
                self.assertRegex(line, r' readback_cpu_ms=[0-9.]+$')
            self.assertIsNotNone(c2_smoke_check.HOSTGAP_VERIFY_LINE.search(line))
            prestage = self.prestage_lines()[-1]
            self.assertIsNotNone(c2_smoke_check.HOSTGAP_PRESTAGE_LINE.search(prestage))
            self.assertIsNotNone(__import__('lever_n_m3native_gate').PRESTAGE_LINE.search(prestage))

    def test_the_facts_the_hostgap_smoke_reads_are_the_same_with_the_flag(self):
        facts = {}
        for keyed in (True, False):
            block, users, nxt = self.ready(keyed=keyed, hostgap_log=True, full_audit=True)
            self.h1a.clear()
            for step in (3, 6):
                self.window(block, tvp.advanced(users, step))
                self.round(block, tvp.advanced(users, step))
            facts[keyed] = c2_smoke_check.hostgap_facts('\n'.join(self.h1a))
        for name in ('hostgap_live4_verifies', 'hostgap_live4_diff', 'hostgap_full_audits', 'hostgap_full_audit_mismatches'):
            self.assertEqual(facts[True][name], facts[False][name], name)


class ExtentBlockTests(tpe.ExtentFixture):
    """The extent readers' block (S2): the readers' word, cur_pos and full-width tables are functions of (start, table) too."""

    def keyed_block(self, keyed):
        self.env(QWEN_FAST_PRESTAGE='1', QWEN_FAST_PRESTAGE_AUDIT='1', **({FLAG: '1'} if keyed else {}))
        block = self.build()
        first = (20218, 30000, 40000, 50000)
        block.verify(self.round(first))
        self.finish(block)
        return block, first

    def prestage_for(self, block, starts):
        users = [((0,) * 16, start, torch.full((1, tpe.WIDTH), 7 + segment, dtype=torch.int32)) for segment, start in enumerate(starts)]
        block.prestaged.prestage(users)

    def test_a_keyed_round_writes_the_tokens_alone_and_the_audit_is_clean(self):
        block, first = self.keyed_block(True)
        second = (20226, 30005, 40005, 50005)
        self.prestage_for(block, second)
        marked = len(self.ttnn.host_copies)
        block.verify(self.round(second))
        written = [destination for host, destination in self.ttnn.host_copies[marked:]]
        self.assertEqual(block.prestaged.last['keyed'], 1)
        self.assertEqual(written, [block.fixture.tokens])
        self.assertEqual(block.prestaged.counts['keyed'], 1)
        self.assertEqual(block.prestaged.counts['keyed_mismatches'], 0)
        self.assertGreater(block.prestaged.counts['keyed_audited'], 20)
        self.assertEqual([reader.start for reader in self.readers(block)], list(second))
        self.finish(block)

    def test_a_start_that_moved_takes_the_value_diff_which_rewrites_the_readers_word_and_cur_pos(self):
        block, first = self.keyed_block(True)
        second = (20226, 30005, 40005, 50005)
        self.prestage_for(block, second)
        moved = (20226, 30005, 40006, 50005)
        marked = len(self.ttnn.host_copies)
        block.verify(self.round(moved))
        written = [destination for host, destination in self.ttnn.host_copies[marked:]]
        self.assertEqual(block.prestaged.last['keyed'], 0)
        self.assertEqual(block.prestaged.last['path'], 'diff')
        self.assertGreater(len(written), 1)
        self.assertEqual(block.prestaged.counts['keyed'], 0)
        self.finish(block)


class SmokeRuleTests(unittest.TestCase):
    ENV = {FLAG: '1'}
    ENGAGED = '[PINDIAG] verify prestage keyed engaged users=4 audit=0'

    @staticmethod
    def diff_lines(total, keyed):
        return ['[PACKED-PRESTAGE] round=%d path=diff buffers=1 reason=- live=4%s' % (number, ' keyed=1' if number < keyed else '')
                for number in range(total)]

    def text(self, *parts):
        return '\n'.join(parts[0] if len(parts) == 1 and isinstance(parts[0], list) else parts)

    def test_a_clean_flagged_log_has_no_problem(self):
        text = self.text([self.ENGAGED] + self.diff_lines(40, 30))
        self.assertEqual(w3_keyed_smoke.problems(self.ENV, text), [])

    def test_a_profile_without_the_flag_logs_none_of_it(self):
        self.assertEqual(w3_keyed_smoke.problems({}, self.text(self.diff_lines(40, 0))), [])
        for extra in (self.ENGAGED, self.diff_lines(1, 1)[0],
                      '[PACKED-PRESTAGE-KEYED-AUDIT] round=3 checked=100 keyed_mismatches=0',
                      '[PACKED-PRESTAGE-FULLAUDIT] block=A round=3 path=diff buffers=150 checked=600 mismatches=0 keyed=1',
                      '[PACKED-HOSTGAP-VERIFY] block=A round=3 live=4 path=diff reason=- bind_ms=0.00 readback_cpu_ms=1.00 keyed=1'):
            with self.subTest(extra=extra[:60]):
                found = w3_keyed_smoke.problems({}, self.text([extra]))
                self.assertTrue(found and 'without %s' % FLAG in found[0])

    def test_a_bad_flag_value_is_a_problem(self):
        self.assertTrue(w3_keyed_smoke.problems({FLAG: 'yes'}, ''))

    def test_the_engaged_line_is_required_with_the_flag(self):
        found = w3_keyed_smoke.problems(self.ENV, self.text(self.diff_lines(10, 10)))
        self.assertTrue(found and 'no engaged line' in found[0])

    def test_the_keyed_share_of_the_diff_rounds_is_judged_from_twenty_rounds(self):
        low = w3_keyed_smoke.problems(self.ENV, self.text([self.ENGAGED] + self.diff_lines(40, 5)))
        self.assertTrue(low and 'only 5 of 40 diff verifies were keyed' in low[0])
        edge = w3_keyed_smoke.problems(self.ENV, self.text([self.ENGAGED] + self.diff_lines(40, 12)))
        self.assertEqual(edge, [])
        short = w3_keyed_smoke.problems(self.ENV, self.text([self.ENGAGED] + self.diff_lines(19, 0)))
        self.assertEqual(short, [], 'fewer than 20 diff rounds are not judged')
        full_only = [line.replace('path=diff', 'path=full') for line in self.diff_lines(40, 0)]
        self.assertEqual(w3_keyed_smoke.problems(self.ENV, self.text([self.ENGAGED] + full_only)), [], 'full rounds are not diff rounds')

    def test_a_keyed_mismatch_is_a_problem_on_any_profile(self):
        line = '[PACKED-PRESTAGE-KEYED-AUDIT] round=4 checked=140 keyed_mismatches=2 at=2,3'
        for env in (self.ENV, {}):
            found = w3_keyed_smoke.problems(env, self.text([self.ENGAGED, line] + self.diff_lines(40, 30)))
            self.assertTrue(any('audit found a difference' in item and 'keyed_mismatches=2' in item for item in found))
        full = '[PACKED-PRESTAGE-FULLAUDIT] block=B round=4 path=diff buffers=150 checked=600 mismatches=3 at=1.0 keyed=1'
        found = w3_keyed_smoke.problems(self.ENV, self.text([self.ENGAGED, full] + self.diff_lines(40, 30)))
        self.assertTrue(any('full pre-stage audit found a difference on a keyed round' in item for item in found))
        unkeyed = full.replace(' keyed=1', '')
        self.assertEqual(w3_keyed_smoke.problems(self.ENV, self.text([self.ENGAGED, unkeyed] + self.diff_lines(40, 30))), [])

    def test_an_audited_arm_needs_a_keyed_audit_line(self):
        for flag in ('QWEN_FAST_PRESTAGE_AUDIT', 'QWEN_FAST_TP4_TWO_BLOCK_PRESTAGE_AUDIT'):
            env = dict(self.ENV, **{flag: '1'})
            base = [self.ENGAGED] + self.diff_lines(40, 30)
            found = w3_keyed_smoke.problems(env, self.text(base))
            self.assertTrue(found and 'no keyed audit line' in found[0])
            ok = w3_keyed_smoke.problems(env, self.text(base + ['[PACKED-PRESTAGE-KEYED-AUDIT] round=4 checked=140 keyed_mismatches=0']))
            self.assertEqual(ok, [])

    def test_the_module_is_stdlib_only_and_python37_syntax(self):
        source = (HERE / 'w3_keyed_smoke.py').read_text(encoding='utf-8')
        tree = ast.parse(source, feature_version=(3, 7))
        imported = {alias.name.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imported |= {node.module.split('.')[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertEqual(imported, {'re'})


class HygieneTests(unittest.TestCase):
    """The public repo: no host, address, serial, registry name, digest or home path in what this lever adds."""

    def test_no_new_file_names_a_host_a_path_or_a_digest(self):
        banned = re.compile('|'.join(('blackhole-[A-Za-z0-9]{8,}', 'thatch' + r'\.local', r'\d{1,3}(\.\d{1,3}){3}', 'sha256:[0-9a-f]{16}',
                                      '[0-9a-f]{40,}', '/dev/' + 'tenstorrent', '/ho' + 'me/', 'zo' + 't' + r'\.')))
        for path in (HERE / 'w3_keyed_smoke.py', HERE.parent.parent / 'docs' / 'tp4-w3-keyed.md'):
            with self.subTest(path=path.name):
                self.assertEqual([match.group(0) for match in banned.finditer(path.read_text(encoding='utf-8'))], [])

    def test_the_serving_module_keeps_the_words_the_extent_test_bans_out_and_its_packed_values_calls(self):
        source = (HERE / 'verify_prestage.py').read_text(encoding='utf-8')
        self.assertNotIn('extent', source.replace('extent storage', '').replace('extent_storage', ''))
        self.assertEqual(source.count('packed_values('), 5)


if __name__ == '__main__':
    unittest.main()
