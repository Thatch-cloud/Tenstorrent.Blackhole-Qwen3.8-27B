"""Variable-user packed rounds M1: PackedVerifierEngine.idle_inputs and the G-pad probe
(padded_probe.py, QWEN_FAST_PADDED_PROBE=1).

The block is test_packed_verifier's own four-user fixture (the real stage_packed and the real
per-user replay readers over a fake ttnn), with one addition: a fake device model. Every
replay of the verify trace computes the outputs from what is staged - the logits and ids
rows, the five taps and every segment's retained GDN states - so the probe's comparisons are
of real functions of the inputs. Separable by default (a row depends on its own segment's
inputs and carry); tests switch in a cross-row term, a trace that writes an idle carry, and a
nondeterministic replay.

With the flag off the verify round is proved byte for byte the pinned commit's
(test_dflash_proposal_trace.PINNED_COMMIT) over the same fixture."""

from itertools import count
import os
import re
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import packed_verifier
import padded_probe
import test_packed_verifier as tpv
from test_dflash_proposal_trace import PINNED_COMMIT, Normalizer, pinned_module
import verifier_engine
import verify_trace_t2

LINE = re.compile(r'\[PINDIAG\] padded probe round=([0-9]+) live=([0-9,]+) exact=([a-z0-9]+) trace_ms=(\S+) '
                  r'idle_carry_intact=(\S+) idle=(\S+) differ=(\S+)')


class DeviceModel:
    """What a replay of the verify trace writes, from what is staged. `cross_row` adds a term
    that couples every row to every other; `write_idle_carry` makes the trace advance the carry
    of any segment staged idle (all tokens 1); `drift` makes every replay differ."""

    def __init__(self, test, block):
        self.test, self.block = test, block
        self.cross_row = self.write_idle_carry = self.drift = False
        self.replays = 0
        ttnn = test.ttnn
        logits = block.output[0]
        logits.chips = [torch.zeros(1, 1, block.block_rows, 8), torch.zeros(1, 1, block.block_rows, 8)]
        for tap in block.taps:
            tap.chips = [torch.zeros(1, 1, block.block_rows, 4), torch.zeros(1, 1, block.block_rows, 4)]
        for layer, (state, result, carries) in enumerate(block.fixture.retained.records):
            for piece in result['segment_results']:
                piece['states'] = ttnn.allocate((16, 3), 'f32', 'tile', torch.zeros(16, 3))
                piece['packed_conv_states'] = [ttnn.allocate((4, 2), 'f32', 'tile', torch.zeros(4, 2)) for _ in range(4)]
        for segment, carry in enumerate(block.carries):
            for layer, snapshot in enumerate(carry):
                for part, value in enumerate(snapshot):
                    value.value = torch.full((5,), float(segment * 1000 + layer * 10 + part))

    def __call__(self):
        block = self.block
        fixture = block.fixture
        self.replays += 1
        tokens = fixture.tokens.value[:, 0].to(torch.float64)
        positions = fixture.positions.value.to(torch.float64)
        pages = fixture.pages.value[:, 0].to(torch.float64)
        base = tokens * 100000 + positions * 10 + pages
        if self.cross_row:
            base = base + tokens.sum()
        if self.drift:
            base = base + self.replays
        rows = block.block_rows
        block.output[0].chips = [(base + chip)[None, None, :, None].repeat(1, 1, 1, 8).to(torch.float32)
                                 for chip in range(2)]
        block.output[1].value = (base.to(torch.int64) % 1000003).to(torch.int32)
        for index, tap in enumerate(block.taps):
            tap.chips = [(base + index + 0.25 * chip)[None, None, :, None].repeat(1, 1, 1, 4).to(torch.float32)
                         for chip in range(2)]
        for layer, (state, result, carries) in enumerate(block.fixture.retained.records):
            for segment, piece in enumerate(result['segment_results']):
                start, stop = 16 * segment, 16 * segment + 16
                carry = float(block.carries[segment][layer][0].value[0])
                piece['states'].value = (base[start:stop, None] + layer + carry).repeat(1, 3).to(torch.float32)
                for tap, conv in enumerate(piece['packed_conv_states']):
                    conv.value = (base[stop - 4:stop, None] + tap).repeat(1, 2).to(torch.float32)
        if self.write_idle_carry:
            for segment in range(block.users):
                if bool((fixture.tokens.value[16 * segment:16 * segment + 16, 0] == 1).all()):
                    block.carries[segment][0][0].value = block.carries[segment][0][0].value + 1
        assert rows == len(tokens)


class ProbeFixture(tpv.FourUserFixture):
    def setUp(self):
        super().setUp()
        self.lines = []
        patcher = patch.object(padded_probe, 'log_line', side_effect=self.lines.append)
        patcher.start()
        self.addCleanup(patcher.stop)
        environment = patch.dict(os.environ, {name: value for name, value in os.environ.items()
                                              if not name.startswith('QWEN_FAST_')}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)
        padded_probe._STATE.update(rounds=0, hits=0)
        ttnn = self.ttnn

        def to_torch(shard):
            chips = getattr(shard.tensor, 'chips', None)
            return chips[shard.tensor.shards.index(shard)] if chips is not None else shard.tensor.value

        ttnn.to_torch = to_torch
        self.model_hook = None
        original = ttnn.execute_trace

        def execute_trace(mesh, trace, cq_id=0, blocking=True):
            original(mesh, trace, cq_id=cq_id, blocking=blocking)
            if self.model_hook is not None and trace == self.model_hook.block.trace:
                self.model_hook()

        ttnn.execute_trace = execute_trace

    def probed_block(self, probe=True):
        if probe:
            os.environ['QWEN_FAST_PADDED_PROBE'] = '1'
        block = self.build()
        self.model_hook = DeviceModel(self, block)
        return block

    def serve(self, block, rounds, prefixes=(9, 16, 0, 4)):
        """`rounds` rounds of the four users, each committed; returns every round's predictions."""
        served = []
        for _ in range(rounds):
            predictions, metrics = block.verify(self.four())
            served.append(predictions)
            for segment, prefix in zip(metrics['segments'], prefixes):
                block.commit_user(segment, prefix)
        return served

    def parsed(self):
        return [LINE.search(line).groups() for line in self.lines if LINE.search(line)]


class IdleInputsTests(ProbeFixture):
    def test_two_idle_segments_sit_on_page_zero_in_disjoint_tile_rows(self):
        block = self.build()
        family_start = block.replay_capacity - 256
        idle = block.idle_inputs((0, 2))
        self.assertEqual(sorted(idle), [1, 3])
        for index, segment in enumerate((1, 3)):
            tokens, start, table = idle[segment]
            self.assertEqual(tokens, (1,) * 16)
            self.assertEqual(start, family_start + 32 * index)
            self.assertEqual((tuple(table.shape), table.dtype), ((1, tpv.PAGE_WIDTH), torch.int32))
            self.assertFalse(bool(table.any()), 'an all-zero table: every position on page 0')
        rows = [verify_trace_t2.kv_tile_rows(range(start, start + 16), table[0]) for _, start, table in idle.values()]
        self.assertEqual(rows, [{(0, 0)}, {(0, 1)}])
        # beside two live users on their own pages, the T2 guard passes
        users = [(None, 4100, torch.full((1, tpv.PAGE_WIDTH), 7, dtype=torch.int32)), idle[1],
                 (None, 4150, torch.full((1, tpv.PAGE_WIDTH), 13, dtype=torch.int32)), idle[3]]
        self.assertIsNone(verify_trace_t2.kv_conflict([(range(start, start + 16), table[0]) for _, start, table in users]))

    def test_a_third_idle_segment_is_refused_and_would_share_a_tile_row(self):
        block = self.build()
        for live in ((0,), (3,), ()):
            with self.subTest(live=live), self.assertRaisesRegex(ValueError, 'At most 2 idle segments'):
                block.idle_inputs(live)
        # why: the j % 2 layout puts the third on the first one's tile row, which the chained
        # K/V write refuses
        first = block.replay_capacity - 256
        table = torch.zeros((1, tpv.PAGE_WIDTH), dtype=torch.int32)
        conflict = verify_trace_t2.kv_conflict([(range(first, first + 16), table[0]),
                                                (range(first + 32, first + 48), table[0]),
                                                (range(first, first + 16), table[0])])
        self.assertEqual((conflict['users'], conflict['page'], conflict['tile_row']), ((0, 2), 0, 0))

    def test_live_segments_must_be_distinct_segments_of_the_block(self):
        block = self.build()
        for live in ((0, 0, 1), (0, 4), (0, -1), ('0', 1)):
            with self.subTest(live=live), self.assertRaisesRegex(ValueError, 'distinct segments'):
                block.idle_inputs(live)
        self.assertEqual(block.idle_inputs((0, 1, 2, 3)), {})
        self.assertEqual(sorted(block.idle_inputs((1, 2, 3))), [0])

    def test_idle_inputs_stage_through_the_real_readers_and_the_t2_backstop(self):
        block = self.build()
        entries = self.four()
        live = block.segment_users(entries, block.segments(entries))
        idle = block.idle_inputs((0, 1))
        padded = [live[0], live[1], idle[2], idle[3]]
        block.fixture.kv_chains = True
        packed_verifier.stage_packed(self.ttnn, self.model, block.fixture, block.shape, padded)
        reader = block.fixture.replay_reader
        first = block.replay_capacity - 256
        self.assertEqual([own.positions.value.tolist()[0] for own in reader.readers], [4100, 4200, first, first + 32])
        self.assertEqual(block.fixture.tokens.value[32:, 0].tolist(), [1] * 32)
        self.assertEqual(block.fixture.pages.value[32:].unique().tolist(), [0])
        for own in reader.readers[2:]:
            for bundle, table, mask, config in own.metadata:
                self.assertFalse(bool(table.value.any()))
        # a third idle segment, forced past idle_inputs, is what the chained write's backstop refuses
        with patch.object(verify_trace_t2, 'log_line'), self.assertRaisesRegex(ValueError, 'disjoint cache tile rows'):
            packed_verifier.stage_packed(self.ttnn, self.model, block.fixture, block.shape,
                                         [live[0], idle[3], idle[2], (idle[2][0], first, idle[2][2])])

    def test_the_two_user_block_takes_one_idle_segment(self):
        class TwoUsers(tpv.BlockFixture):
            def runTest(self):
                pass

        fixture = TwoUsers()
        fixture.setUp()
        try:
            block = fixture.build()
            idle = block.idle_inputs((1,))
            self.assertEqual(list(idle), [0])
            self.assertEqual(idle[0][1], block.replay_capacity - 256)
            self.assertEqual(padded_probe.patterns_for(2), ((0,), (1,)))
        finally:
            fixture.doCleanups()


class FlagOffTests(ProbeFixture):
    def run_rounds(self, module, rounds=3):
        """Build with `module`'s engine, serve `rounds` rounds, close; the fake ttnn's record."""
        tpv.FakeModelBatch.instances, tpv.FakeFeatures.instances = [], []
        self.traces = count(1)
        verifier_engine.note_prefill()
        ttnn = self.ttnn
        marks = dict(hosts=len(ttnn.hosts), copies=len(ttnn.host_copies), executed=len(ttnn.executed),
                     released=len(ttnn.released), deallocated=len(ttnn.deallocated), uploads=len(ttnn.device_uploads),
                     zeroed=len(ttnn.zeroed), synchronized=ttnn.synchronized)

        def capture_operation(operations, mesh, operation):
            return 'trace%d' % next(self.traces), operation()

        with patch.object(module, 'ModelBatch', tpv.FakeModelBatch), \
                patch.object(module, 'PreparedTargetFeatures', tpv.FakeFeatures), \
                patch.object(module, 'prepare', Mock(side_effect=lambda mesh, layers, prefix: Mock(name='publication'))), \
                patch.object(module, 'capture_operation', Mock(side_effect=capture_operation)), \
                patch.object(module, 'sample_rows', Mock(side_effect=lambda *args, **options: self.ids)):
            block = module.PackedVerifierEngine(ttnn, self.model, self.helpers, 'sampler', pool=self.pool,
                shared_weights=self.weights, shape=self.shape(), feature_taps=tpv.TAPS)
            self.model_hook = DeviceModel(self, block)
            served = []
            for _ in range(rounds):
                predictions, metrics = block.verify(self.four())
                served.append((predictions, metrics['segments'], metrics['staged_buffers']))
                for segment, prefix in zip(metrics['segments'], (9, 16, 0, 4)):
                    block.commit_user(segment, prefix)
            block.close()
            self.model_hook = None
        normalize = Normalizer()
        return dict(served=served, synchronized=ttnn.synchronized - marks['synchronized'],
                    hosts=[normalize((host.value, host.dtype, host.layout)) for host in ttnn.hosts[marks['hosts']:]],
                    copies=[normalize(pair) for pair in ttnn.host_copies[marks['copies']:]],
                    executed=list(ttnn.executed[marks['executed']:]),
                    blocking=list(ttnn.execute_blocking[marks['executed']:]),
                    released=list(ttnn.released[marks['released']:]),
                    deallocated=[normalize(value) for value in ttnn.deallocated[marks['deallocated']:]],
                    uploads=[normalize(value) for value in ttnn.device_uploads[marks['uploads']:]],
                    zeroed=[normalize(value) for value in ttnn.zeroed[marks['zeroed']:]])

    def test_the_round_matches_the_pinned_source_byte_for_byte(self):
        pinned = pinned_module('packed_verifier.py', 'packed_verifier_pinned')
        if pinned is None:
            self.skipTest('no git history for %s' % PINNED_COMMIT)
        before = self.run_rounds(pinned)
        today = self.run_rounds(packed_verifier)
        self.assertGreater(len(before['copies']), 400)
        self.assertEqual(today, before)
        self.assertEqual(self.lines, [])

    def test_off_the_probe_module_is_never_reached(self):
        block = self.probed_block(probe=False)
        with patch.object(padded_probe, 'after_readback') as probe:
            self.serve(block, 3)
        probe.assert_not_called()
        self.assertFalse(block.padded_probe)


class ProbeRoundTests(ProbeFixture):
    def test_non_probe_rounds_replay_once_and_check_page_zero(self):
        block = self.probed_block()
        executed = len(self.ttnn.executed)
        self.serve(block, 2)
        self.assertEqual([trace for trace in self.ttnn.executed[executed:] if trace == block.trace], [block.trace] * 2)
        self.assertEqual(self.parsed(), [])
        self.assertEqual(padded_probe._STATE, dict(rounds=2, hits=0))

    def test_round_three_replays_every_pattern_and_the_round_itself(self):
        block = self.probed_block()
        reference = self.serve(block, 2)
        executed = len(self.ttnn.executed)
        predictions = self.serve(block, 1)[0]
        verifies = [trace for trace in self.ttnn.executed[executed:] if trace == block.trace]
        # the round's own replay, three padded patterns (two refused), the restaged round
        self.assertEqual(len(verifies), 5)
        rows = self.parsed()
        self.assertEqual([(r, live, exact, intact, idle, differ) for r, live, exact, ms, intact, idle, differ in rows], [
            ('3', '0', 'refused', '-', '1,2,3', '-'),
            ('3', '1', 'refused', '-', '0,2,3', '-'),
            ('3', '0,1', '1', '1', '2,3', '-'),
            ('3', '1,3', '1', '1', '0,2', '-'),
            ('3', '0,1,2', '1', '1', '3', '-'),
            ('3', '0,1,2,3', '1', '1', '-', '-')])
        self.assertTrue(all(float(ms) >= 0 for r, live, exact, ms, *rest in rows if exact == '1'))
        self.assertIn('reason=At_most_2_idle_segments', self.lines[1])
        self.assertIn('[PINDIAG] padded probe page0 rounds=3 hits=0', self.lines)
        # the probe never changes what is served: the same entries every round, the same rows
        self.assertEqual(predictions, reference[0])

    def test_the_probe_restages_all_live_inputs_before_the_commit(self):
        block = self.probed_block()
        self.serve(block, 2)
        seen = []
        retained = block.fixture.retained
        commit = retained.commit_user.side_effect

        def at_commit(segment, prefix, **options):
            fixture = block.fixture
            seen.append(dict(tokens=fixture.tokens.value.clone(), positions=fixture.positions.value.clone(),
                             pages=fixture.pages.value.clone(),
                             words=[own.positions.value[0].item() for own in fixture.replay_reader.readers],
                             starts=fixture.replay_reader.starts, last=self.ttnn.executed[-1],
                             logits=[chip.clone() for chip in block.output[0].chips]))
            return commit(segment, prefix, **options)

        retained.commit_user.side_effect = at_commit
        entries = self.four()
        predictions, metrics = block.verify(entries)
        for segment, prefix in zip(metrics['segments'], (9, 16, 0, 4)):
            block.commit_user(segment, prefix)
        first = seen[0]
        for user in range(4):
            rows = slice(16 * user, 16 * user + 16)
            self.assertEqual(first['tokens'][rows, 0].tolist(), list(range(self.TOKENS[user], self.TOKENS[user] + 16)))
            self.assertEqual(first['positions'][rows].tolist(),
                             list(range(self.POSITIONS[user], self.POSITIONS[user] + 16)))
            self.assertTrue(bool((first['pages'][rows] == self.PAGES[user]).all()))
        self.assertEqual(first['words'], list(self.POSITIONS))
        self.assertEqual(first['starts'], self.POSITIONS)
        self.assertEqual(first['last'], block.trace, 'the last replay before the commit is the restaged round')
        # and what the commit reads is the round's own replay, bit for bit
        self.model_hook()
        self.assertTrue(all(torch.equal(mine, theirs) for mine, theirs in zip(first['logits'], block.output[0].chips)))

    def test_a_cross_row_trace_is_reported_and_the_round_still_served(self):
        block = self.probed_block()
        self.serve(block, 2)
        self.model_hook.cross_row = True
        entries = self.four()
        predictions, metrics = block.verify(entries)
        rows = {live: (exact, differ) for r, live, exact, ms, intact, idle, differ in self.parsed()}
        for live in ('0,1', '1,3', '0,1,2'):
            self.assertEqual(rows[live], ('0', 'ids:0+1,logits:0+1,taps:0+1,states:0+1'))
        self.assertEqual(rows['0,1,2,3'], ('1', '-'), 'deterministic: the restaged round is the round')
        self.assertEqual(block.phase, 'verified')
        for segment in range(4):
            block.commit_user(segment, 1)

    def test_a_trace_that_writes_an_idle_carry_fails_the_round(self):
        block = self.probed_block()
        self.serve(block, 2)
        self.model_hook.write_idle_carry = True
        entries = self.four()
        with self.assertRaisesRegex(AssertionError, 'could not show the round restored'):
            block.verify(entries)
        self.assertEqual(block.phase, 'failed')
        for item in entries:
            item['request'].session.fail_verification.assert_called_once_with(item['request_id'], item['ticket'])
        rows = {live: intact for r, live, exact, ms, intact, idle, differ in self.parsed()}
        self.assertEqual(rows['0,1'], '0')
        self.assertEqual(rows['0,1,2,3'], '0', 'every carry against the probe start')

    def test_a_nondeterministic_replay_fails_the_round(self):
        block = self.probed_block()
        self.serve(block, 2)
        self.model_hook.drift = True
        entries = self.four()
        with self.assertRaisesRegex(AssertionError, 'could not show the round restored'):
            block.verify(entries)
        final = [row for row in self.parsed() if row[1] == '0,1,2,3'][0]
        self.assertEqual(final[2], '0')
        self.assertEqual(block.phase, 'failed')

    def test_page_zero_in_a_live_table_is_reported_every_round_and_stops_the_probe(self):
        block = self.probed_block()
        executed = len(self.ttnn.executed)
        for _ in range(3):
            entries = self.four()
            entries[1]['request'].engine.pages[0, 64] = 0     # A's page index 64: inside [0, (4100 + 79) // 64)
            predictions, metrics = block.verify(entries)
            for segment in metrics['segments']:
                block.commit_user(segment, 2)
        hits = [line for line in self.lines if line.startswith(padded_probe.PAGE0_HIT)]
        self.assertEqual(hits, ['[PINDIAG] padded probe page0 hit round=%d segment=0 position=4100 page_index=64' % r
                                for r in (1, 2, 3)])
        self.assertEqual([row[2] for row in self.parsed()], ['refused'])
        self.assertEqual(len([trace for trace in self.ttnn.executed[executed:] if trace == block.trace]), 3)
        self.assertEqual(padded_probe._STATE, dict(rounds=3, hits=3))

    def test_a_page_zero_entry_past_the_used_range_is_not_a_hit(self):
        users = [((1,) * 16, 4100, torch.full((1, tpv.PAGE_WIDTH), 7, dtype=torch.int32))]
        users[0][2][0, 65:] = 0
        self.assertEqual(padded_probe.page_zero_hits(users, 16), [])
        users[0][2][0, 64] = 0
        self.assertEqual(padded_probe.page_zero_hits(users, 16), [(0, 4100, 64)])
        self.assertEqual(padded_probe.page_zero_hits([None], 16), [])


class ShippingTests(unittest.TestCase):
    def test_the_probe_and_the_verifier_reach_the_image(self):
        from test_serving_image_copy_closure import copied_modules, dockerfile_text

        copied = copied_modules(dockerfile_text())
        for name in ('padded_probe.py', 'packed_verifier.py'):
            with self.subTest(module=name):
                self.assertIn(name, copied)


if __name__ == '__main__':
    unittest.main()
