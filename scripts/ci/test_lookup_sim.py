"""optimisation/lookup: the offline prompt-lookup estimate and the tape builder, on synthetic tapes (no logged data is read)."""
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
LOOKUP = HERE.parent.parent / 'optimisation' / 'lookup'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(LOOKUP))

import build_tapes  # noqa: E402
import lookup_sim  # noqa: E402

BODY = list(range(100, 140))                    # a 40-token passage the prompt holds and the answer copies


def tape(prompt, out, rounds, **extra):
    record = dict(id='t', set='swe', arm='A1', cluster='c', weight=1.0, plen=len(prompt), prompt=prompt, out=out, rounds=rounds,
                  thinking=False, think_end=None, tool_at=None)
    record.update(extra)
    return record


def copying_turn():
    """The answer is out[0] then the passage; DFlash2 logged short rounds of 3 tokens, so a lookup that finds the passage wins."""
    prompt = [7, 8, 9] + BODY + [5, 5, 5]
    out = [4] + BODY
    rounds, off = [], 1
    while off < len(out):
        em = min(3, len(out) - off)
        rounds.append([off, em, 1, None])
        off += em
    return tape(prompt, out, rounds)


class HelperTests(unittest.TestCase):
    def test_lcp(self):
        self.assertEqual(lookup_sim.lcp((1, 2, 3), [1, 2, 9, 4], 0), 2)
        self.assertEqual(lookup_sim.lcp((1, 2, 3), [0, 1, 2, 3], 1), 3)
        self.assertEqual(lookup_sim.lcp((), [1], 0), 0)

    def test_edge_left_never_crosses_the_next_256_edge_or_the_end(self):
        self.assertEqual(lookup_sim.edge_left(250, 1, 100), 100 if 251 + 100 <= 257 else 257 - 251)
        self.assertEqual(lookup_sim.edge_left(1000, 1, 3), 3)
        self.assertEqual(lookup_sim.edge_left(256, 1, 100), 100)          # s = 257 = 1 mod 256: the next edge is 513

    def test_region(self):
        turn = dict(thinking=True, think_end=10, tool_at=20)
        self.assertEqual([lookup_sim.region_of(turn, i) for i in (3, 10, 25)], ['reasoning', 'content', 'tool'])


class TurnTests(unittest.TestCase):
    def test_lookup_facts_find_the_passage_from_the_prompt(self):
        turn = lookup_sim.Turn(copying_turn(), ns=(3,))
        match, k, accepted = turn.look[3][4]               # start 4: history ends ...BODY[0:3]; the passage follows
        self.assertGreaterEqual(match, 3)
        self.assertEqual(k, 15)
        self.assertEqual(accepted, 15)

    def test_without_the_prompt_the_lookup_has_nothing_to_copy(self):
        record = copying_turn()
        record['prompt'] = []
        turn = lookup_sim.Turn(record, ns=(3,))
        self.assertTrue(all(row is None or row == (0, 0, 0) for row in turn.look[3]))

    def test_the_tape_marks_hits_misses_and_the_logged_accepted_count(self):
        turn = lookup_sim.Turn(tape([1, 2], [9, 1, 2, 3, 4], [[1, 3, 1, None], [4, 1, 1, None]]), ns=(2,))
        self.assertEqual(turn.status[1:3], [0, 0])         # accepted drafts of the first round
        self.assertEqual(turn.status[3], 1)                # its first rejection
        self.assertEqual(turn.rec[1], 2)
        self.assertEqual(turn.rec[4], 0)
        self.assertEqual(turn.spans(), [(1, 5)])

    def test_a_sequential_entry_breaks_the_span(self):
        record = tape([1, 2], [9, 1, 2, 3, 4, 5], [[1, 2, 1, None], [None, 0, 0, None], [4, 2, 1, None]])
        self.assertEqual(lookup_sim.Turn(record, ns=(2,)).spans(), [(1, 3), (4, 6)])


class PolicyTests(unittest.TestCase):
    def run_policy(self, mode, name, gates=(2, 4, 8)):
        turn = lookup_sim.Turn(copying_turn(), ns=(3,))
        table, _ = lookup_sim.simulate([turn], 3, mode, 0.65, 0, True, gates)
        return table[name][('all',)]

    def test_dflash_replays_the_logged_rounds_in_both_modes(self):
        for mode in ('recorded', 'walk'):
            cell = self.run_policy(mode, 'dflash')
            self.assertEqual(cell[2], 14, mode)              # 40 tokens in rounds of 3: 13 full + a tail of 1
            self.assertAlmostEqual(cell[0] / cell[1], 40 / 14.0)

    def test_a_gated_lookup_beats_dflash_on_a_copying_turn_and_counts_its_rounds(self):
        base = self.run_policy('walk', 'dflash')
        gated = self.run_policy('walk', 'lookup>=4')
        self.assertGreater(gated[0] / gated[1], base[0] / base[1])
        self.assertGreater(gated[3], 0)
        self.assertGreaterEqual(gated[4], 1)                 # wins
        self.assertLess(gated[2], base[2])                   # fewer rounds: the walk conserves tokens

    def test_max_is_never_below_the_better_policy_in_recorded_mode(self):
        best = self.run_policy('recorded', 'max')
        for name in ('dflash', 'lookup>=2', 'lookup>=8'):
            other = self.run_policy('recorded', name)
            self.assertGreaterEqual(best[0], other[0] - 1e-9, name)

    def test_a_gate_above_every_match_is_dflash(self):
        base = self.run_policy('walk', 'dflash')
        never = lookup_sim.simulate([lookup_sim.Turn(copying_turn(), ns=(3,))], 3, 'walk', 0.65, 0, True, (64,))[0]['lookup>=64'][('all',)]
        self.assertEqual(never[:3], base[:3])

    def test_the_spliced_ticket_is_at_least_the_pure_lookups(self):
        record = copying_turn()
        turn = lookup_sim.Turn(record, ns=(3,))
        pure = lookup_sim.simulate([turn], 3, 'recorded', 0.65, 0, False, (3,))[0]['lookup>=3'][('all',)]
        spliced = lookup_sim.simulate([turn], 3, 'recorded', 0.65, 0, True, (3,))[0]['lookup>=3'][('all',)]
        self.assertGreaterEqual(spliced[0], pure[0])

    def test_report_text(self):
        class Args:
            mode, q, seeds, pure, detail = 'walk', 0.65, 1, False, True
            ns, gates = [3], [4, 8]
        text = lookup_sim.report([lookup_sim.Turn(copying_turn(), ns=(3,))], dict(prompt_included=True), Args)
        for word in ('key n = 3', 'dflash', 'lookup>=4', 'max', 'by bucket', 'by region', 'per-turn'):
            self.assertIn(word, text)


class BuilderTests(unittest.TestCase):
    def test_tapes_from_a_lab_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            turns = [dict(status='ok', arm='A1', id='t0', set='swe', cluster='c', weight=2.0, request_id='cmpl-1', prompt_tokens=3, thinking=True)]
            outputs = [dict(arm='A1', id='t0', output_ids=[10, 11, 12, build_tapes.THINK_END, 13, build_tapes.TOOL_CALL, 14])]
            log = ['[PACKED-PHASE] round=1 users=4 bind_ms=1 live=4',
                   '[PACKED] request=cmpl-1-0 segment=0 position=3 prefix=1 emitted=3 predictions=[11,12,%d]' % build_tapes.THINK_END,
                   '[PACKED] request=cmpl-1-0 segment=0 position=6 prefix=1 emitted=2 predictions=[13,%d]' % build_tapes.TOOL_CALL,
                   '[PACKED] request=cmpl-1-0 segment=0 position=8 prefix=1 emitted=1 predictions=[14]']
            paths = {}
            for name, rows in (('turns.jsonl', turns), ('outputs.jsonl', outputs)):
                paths[name] = root / name
                paths[name].write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
            paths['log'] = root / 'server.log'
            paths['log'].write_text('\n'.join(log) + '\n', encoding='utf-8')
            ids = root / 'a.ids.jsonl.gz'
            with gzip.open(str(ids), 'wt', encoding='utf-8') as handle:
                handle.write(json.dumps(dict(turn_id='t0', ids=[1, 2, 3])) + '\n')
            out = root / 'tapes.jsonl.gz'
            build_tapes.main(['--turns', str(paths['turns.jsonl']), '--outputs', str(paths['outputs.jsonl']), '--log', str(paths['log']),
                              '--ids', str(ids), '--out', str(out)])
            header, tapes = lookup_sim.load(str(out))
        self.assertTrue(header['prompt_included'])
        self.assertEqual((header['alignment_checked'], header['alignment_matched']), (3, 3))
        tape_, = tapes
        self.assertEqual(tape_['prompt'], [1, 2, 3])
        self.assertEqual(tape_['rounds'], [[1, 3, 1, None], [4, 2, 1, None], [6, 1, 0, None]])      # the last round is not counted
        self.assertEqual((tape_['think_end'], tape_['tool_at']), (4, 5))
        self.assertTrue(lookup_sim.Turn(tape_, ns=(2,)).counted_set())

    def test_no_prompt_is_flagged_in_the_header(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 't.jsonl').write_text(json.dumps(dict(status='ok', arm='A1', id='t0', request_id='r', prompt_tokens=3)) + '\n', encoding='utf-8')
            (root / 'o.jsonl').write_text(json.dumps(dict(arm='A1', id='t0', output_ids=[1, 2])) + '\n', encoding='utf-8')
            (root / 'l.log').write_text('', encoding='utf-8')
            build_tapes.main(['--turns', str(root / 't.jsonl'), '--outputs', str(root / 'o.jsonl'), '--log', str(root / 'l.log'), '--out', str(root / 'x.jsonl')])
            header, _ = lookup_sim.load(str(root / 'x.jsonl'))
        self.assertFalse(header['prompt_included'])


if __name__ == '__main__':
    unittest.main()
