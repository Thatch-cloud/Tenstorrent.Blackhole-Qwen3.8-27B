"""CPU checks for k64j_card_b.py at ONE KV head per chip (--kv-heads 1: the four-card S2 port, CB1-TP4 and CB2a-TP4); no
device, no ttnn.

A four-card chip holds 6 query heads on one KV head: a token folds to 6 rows (token-major), a group of R tokens is (1, 1, 6R,
256), the narrow mask (B, 1, 6R, 256), and K64j serves 0x21 / 0x23 (the q-slice needs a second KV head and is refused).

  - the geometry: card.set_kv_heads switches the helpers (masks, folds, queries) and is restored; the one-head masks and
    folds are the four-card pinned siblings' (attention_mask_replay_tp, attention_head_fold_tp at QWEN_FAST_TP=4); the
    combos, trace combos and served flags; the pair's queries are the bytes they were;
  - the arguments and the report: --kv-heads, the defaults it selects, kv_heads an int in the report and a word on the verdict
    line (and neither at the pair), the native decode's local-head and KV-head counts, the q-slice refusal's literal;
  - the runner: the watcher pass at one head runs the one-head combos, the pair's argv is unchanged, and the three evidence
    jobs of scripts/ci/references/tp4-s2-serve-jobs (EV-W1, EV-F1, EV-F2) pass the flags this harness implements;
  - the flow on the fake ttnn (test_k64j_card_b.FakeExtentTtnn, which refuses the q-slice at one KV head as F15 does): CB1 and
    CB2a pass end to end at one head with the shapes the four-card chip runs, and the variants that must be caught are.

    py -3.11 -B -m unittest test_k64j_one_head      (from this directory; the runner test needs Git Bash on Windows)
"""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import test_k64j_card_b as base  # noqa: E402 - sys.path (k64j, k64j_probe, sdpa_decode_qwen, scripts/ci), the fake
import test_k64j_cb2a as cb2a  # noqa: E402 - CB2a's dry-run setup and its small arguments

import apply_factory_slice  # noqa: E402 - sdpa_decode_slice (F15's literals)
import attention_head_fold_tp  # noqa: E402 - scripts/ci
import attention_mask_replay_tp  # noqa: E402 - scripts/ci

card_b, probe, model, card = base.card_b, base.probe, base.model, base.card
FakeExtentTtnn = base.FakeExtentTtnn
ROOT = base.ROOT
JOBS = ROOT / 'scripts' / 'ci' / 'references' / 'tp4-s2-serve-jobs'
sha = base.sha


def one_head():
    """The geometry of a four-card chip in force (and restored): card.set_kv_heads(1)."""
    return _Restore(card.set_kv_heads(1))


class _Restore:
    def __init__(self, previous):
        self.previous = previous

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        card.set_kv_heads(self.previous)
        return False


def four_cards():
    return mock.patch.dict(os.environ, {'QWEN_FAST_TP': '4'})


class GeometryTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def test_the_switch_restores_and_refuses(self):
        self.assertEqual((card.KV_HEADS, card.local_heads(), card_b.local_heads()), (2, 12, 12))
        with one_head():
            self.assertEqual((card.KV_HEADS, card.local_heads(), card_b.local_heads()), (1, 6, 6))
        self.assertEqual((card.KV_HEADS, card.local_heads()), (2, 12))
        for bad in (0, 3, '1', None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                card.set_kv_heads(bad)
        self.assertEqual(card.KV_HEADS, 2)

    def test_the_masks_are_the_four_card_pinned_siblings(self):
        torch = self.torch
        with one_head(), four_cards():
            for word in (0, 7, 32, 127, 240, 255, 131079):
                self.assertEqual(card_b.mask_row_positions(word),
                                 [[attention_mask_replay_tp.mask_position(word, 8, batch, head) for head in range(48)]
                                  for batch in range(2)])
                self.assertEqual(card_b.mask_row_positions(word),
                                 card.mask_positions(word, card_b.SERVED_OFFSETS, rows=8))
            self.assertEqual(card_b.mask_row_positions(9, rows=4, batches=3, offset=4),
                             [[attention_mask_replay_tp.mask_position(9, 4, batch, head, 4) for head in range(24)]
                              for batch in range(3)])
            # Token-major: folded row h of a group sits at the position of token h // 6.
            self.assertEqual(card_b.mask_row_positions(100)[1][:14], [108] * 6 + [109] * 6 + [110] * 2)
            narrow = card_b.narrow_mask(torch, 255)
            self.assertEqual(tuple(narrow.shape), (2, 1, 48, 256))
            self.assertEqual(tuple(card_b.served_mask(torch, 2304 + 7, 2304 + 256, width=256).shape), (2, 1, 48, 256))
            self.assertEqual(tuple(card_b.wide_mask(torch, 2304 + 7, 2560).shape), (2, 1, 48, 2560))
            for extent in (2304, 4352, 131328):
                for offset in (0, 7, 127, 240, 255):
                    start = extent - 256 + offset
                    narrow = card_b.narrow_mask(torch, start)
                    self.assertTrue(torch.equal(card.int16_view(torch, narrow),
                                                card.int16_view(torch, card_b.served_mask(torch, start, extent,
                                                                                          width=256))))
                    # card M's own mask builder, at the geometry in force, is the wide and the narrow mask
                    wide = card_b.wide_mask(torch, start, extent)
                    self.assertTrue(torch.equal(card.int16_view(torch, wide), card.int16_view(torch, card.build_mask(
                        torch, extent, start, card_b.SERVED_OFFSETS, rows=8))))
                    self.assertTrue(torch.equal(card.int16_view(torch, card_b.narrow_mask(torch, start)),
                                                card.int16_view(torch, card.build_mask(
                                                    torch, 256, start & 255, card_b.SERVED_OFFSETS, rows=8))))
                    bits = set(card.int16_view(torch, wide).unique().tolist())
                    self.assertTrue(bits <= {0, -128}, bits)                     # 0x0000 and 0xff80 only
                    if offset + 16 <= 256:       # the four-card host oracle's causal mask (token-major rows)
                        for entry, group in enumerate(card_b.SERVED_OFFSETS):
                            truth = attention_head_fold_tp.causal_mask(8, start + group, extent)[0, 0]
                            self.assertTrue(torch.equal(card.int16_view(torch, wide[entry, 0]),
                                                        card.int16_view(torch, truth)), (extent, offset))
        # At the pair the same call gives 96 folded rows.
        self.assertEqual(tuple(card_b.narrow_mask(torch, 255).shape), (2, 1, 96, 256))

    def test_the_fold_is_the_four_card_siblings(self):
        torch = self.torch
        tokens = torch.arange(16 * 6 * 256, dtype=torch.float32).reshape(1, 16, 6, 256)
        with one_head(), four_cards():
            folded = card.fold_entries(torch, tokens, card_b.SERVED_OFFSETS, card_b.SERVED_ROWS)
            self.assertEqual(tuple(folded.shape), (1, 2, 48, 256))
            for entry, offset in enumerate(card_b.SERVED_OFFSETS):
                self.assertTrue(torch.equal(folded[:, entry:entry + 1],
                                            attention_head_fold_tp.fold_query(tokens[:, offset:offset + 8])))
                # one KV head: the fold is a reshape, token-major
                self.assertTrue(torch.equal(folded[0, entry], tokens[0, offset:offset + 8].reshape(48, 256)))
            self.assertTrue(torch.equal(card.unfold_entries(torch, folded, 8), tokens))
            self.assertTrue(torch.equal(card.unfold_rows(folded[:, :1], 8),
                                        attention_head_fold_tp.unfold_output(folded[:, :1], 8)))
            relative = card_b.mask_row_positions(0)
            for entry in range(2):
                for head in range(48):
                    self.assertTrue(torch.equal(folded[0, entry, head], tokens[0, relative[entry][head], head % 6]))
            self.assertTrue(torch.equal(card_b.ticket_query(torch, lambda p: tokens[0, p - 500], 500), folded))
            query = card.build_query(torch, 2, 3, 'normal', rows=8)
            self.assertEqual(tuple(query.shape), (1, 2, 48, 256))
            self.assertEqual(tuple(card.build_mask(torch, 2304, 7, card_b.SERVED_OFFSETS, width=256, rows=8).shape),
                             (2, 1, 48, 256))

    def test_the_token_queries(self):
        torch = self.torch
        table = torch.arange(64, dtype=torch.int32)
        keys = torch.randn(64, 1, 64, 256).to(torch.bfloat16)
        with one_head():
            normal = card_b.token_query(torch, 0, 'normal', 300)
            peaky = card_b.token_query(torch, 0, 'peaky', 300, keys, table)
            self.assertEqual((tuple(normal.shape), tuple(peaky.shape)), ((6, 256), (6, 256)))
            self.assertFalse(torch.equal(normal, peaky))
            self.assertTrue(torch.equal(normal, card_b.token_query(torch, 0, 'normal', 300)))
        # The pair's query is exactly the 12-head draw it always was.
        generator = torch.Generator().manual_seed((0 * 1000003 + 300) * 2)
        self.assertTrue(torch.equal(card_b.token_query(torch, 0, 'normal', 300),
                                    torch.randn(12, 256, generator=generator).to(torch.bfloat16)))

    def test_the_pair_defaults_do_not_move(self):
        self.assertEqual(card_b.default_combos(), [('G4B3', 0x21), ('G4B3', 0x23), ('G8B2', 0x21), ('G8B2', 0x23),
                                                   ('G8B2', 0x27), ('G8B2', 0x2F)])
        self.assertEqual(card_b.default_trace_combos(), card_b.TRACE_COMBOS)
        self.assertEqual(card_b.served_flags_for(), (0x27, 0x7))
        self.assertEqual((card_b.SERVED_FLAGS, card_b.COMPILE_FLAGS), (0x27, 0x7))
        self.assertTrue(probe.q_slice_saves(8) and not probe.q_slice_saves(4))
        self.assertTrue(probe.q_slice_saves(8, 2) and not probe.q_slice_saves(8, 1))
        with one_head():
            self.assertFalse(probe.q_slice_saves(8))                         # the geometry in force: no second KV head
        self.assertTrue(probe.q_slice_saves(8))

    def test_the_one_head_combos_and_flags(self):
        one = card_b.ONE_KV_HEAD
        self.assertEqual(card_b.default_combos(one), [('G4B3', 0x21), ('G4B3', 0x23), ('G8B2', 0x21), ('G8B2', 0x23)])
        self.assertEqual(card_b.default_trace_combos(one), (('G4B3', 0x21), ('G8B2', 0x23)))
        self.assertEqual(card_b.served_flags_for(one), (0x23, 0x3))
        self.assertEqual(card_b.parse_combos('G4B3:0x21,G8B2:0x23', one), [('G4B3', 0x21), ('G8B2', 0x23)])
        for slice_combo in ('G8B2:0x27', 'G8B2:0x2f', 'G4B3:0x27'):
            with self.subTest(combo=slice_combo), self.assertRaises(ValueError):
                card_b.parse_combos(slice_combo, one)
        self.assertEqual(card_b.parse_combos('G8B2:0x27'), [('G8B2', 0x27)])        # still runnable at the pair
        self.assertFalse(card_b.valid_combo('G8B2', 0x27, one))
        self.assertTrue(card_b.valid_combo('G8B2', 0x23, one))
        # 0x23 is 0x27 without the slice: the same entries, the extent and the tail, share on
        self.assertEqual(card_b.SERVED_FLAGS_ONE_HEAD, card_b.SERVED_FLAGS & ~card_b.SLICE)
        self.assertEqual(card_b.COMPILE_FLAGS_ONE_HEAD, card_b.COMPILE_FLAGS & ~card_b.SLICE)
        self.assertEqual(card_b.reference_flags(0x23), 0x3)

    def test_the_split_predictions_at_one_head(self):
        pair = probe.split_predictions((2304, 131328), 131328, [3, 2, 1])
        self.assertEqual(pair, probe.split_predictions((2304, 131328), 131328, [3, 2, 1], 2))
        one = probe.split_predictions((2304, 131328), 131328, [3, 2, 1], 1)
        self.assertEqual({(row['batch'], row['cores_per_head']) for row in one}, {(3, 16), (2, 16), (1, 16)})


class ContractTests(unittest.TestCase):
    def test_the_refusal_literal_is_f15s(self):
        self.assertEqual(card_b.SLICE_HEADS_REFUSAL, apply_factory_slice.HEADS_REFUSAL)
        self.assertIn('needs num_q_heads ({}) a multiple of num_kv_heads ({}) > 1', ''.join(apply_factory_slice.F15_NEW))

    def test_the_one_head_unverified_list(self):
        pair = card_b.unverified_items()
        self.assertEqual(pair, card_b.UNVERIFIED)
        one = card_b.unverified_items(1)
        self.assertEqual(len(one), len(pair))
        changed = [(a, b) for a, b in zip(pair, one) if a != b]
        self.assertEqual(len(changed), 1)
        self.assertIn('12 local heads', changed[0][0])
        self.assertIn('6 local heads on 1 KV head', changed[0][1])
        self.assertNotIn('UNVERIFIED K2: the query shape (1, 1, 12', '\n'.join(card_b.native_decode_lines(1)))
        self.assertIn('the query shape (1, 1, 12, 256)', '\n'.join(card_b.native_decode_lines()))

    def test_the_arguments(self):
        args = card_b.parse_args(['--out', 'x.json'])
        self.assertEqual((args.kv_heads, args.served_flags, args.compile_flags, args.combos, args.trace_combos),
                         (2, 0x27, 0x7, card_b.default_combos(), list(card_b.TRACE_COMBOS)))
        args = card_b.parse_args(['--out', 'x.json', '--kv-heads', '1'])
        self.assertEqual((args.kv_heads, args.served_flags, args.compile_flags), (1, 0x23, 0x3))
        self.assertEqual(args.combos, [('G4B3', 0x21), ('G4B3', 0x23), ('G8B2', 0x21), ('G8B2', 0x23)])
        self.assertEqual(args.trace_combos, [('G4B3', 0x21), ('G8B2', 0x23)])
        self.assertEqual(card.KV_HEADS, 2)                        # parsing never switches the geometry: main does
        args = card_b.parse_args(['--out', 'x.json', '--kv-heads', '1', '--combos', 'G8B2:0x23'])
        self.assertEqual(args.combos, [('G8B2', 0x23)])
        for bad in (['--kv-heads', '3'], ['--kv-heads', '0'], ['--kv-heads', 'x'],
                    ['--kv-heads', '1', '--combos', 'G8B2:0x27'], ['--kv-heads', '1', '--combos', 'G8B2:0x2f'],
                    ['--kv-heads', '1', '--trace-combos', 'G8B2:0x27'], ['--kv-heads', '1', '--combos', '']):
            with self.subTest(bad=bad), self.assertRaises(SystemExit), mock.patch('sys.stderr'):
                card_b.parse_args(['--out', 'x.json'] + bad)
        card_b.parse_args(['--out', 'x.json', '--combos', 'G8B2:0x27', '--trace-combos', 'G8B2:0x27'])   # the pair's

    def test_the_verdict_line_carries_kv_heads_only_at_one_head(self):
        decision = dict(verdict='PASS', reasons=[], first_differing=[], k2='PASS', k2_coverage=dict(covered=1980,
                                                                                                design=1980))
        report = dict(decision=decision, comparisons=[], liveness=[])
        pair = card_b.verdict_line(report)
        self.assertNotIn('kv_heads', pair)
        one = card_b.verdict_line(dict(report, kv_heads=1))
        self.assertIn(' kv_heads=1 k2_verdict=PASS k2_coverage=1980/1980 ', one)
        self.assertEqual(one.replace(' kv_heads=1', ''), pair)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.graft = base.make_graft(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def launch(self, **env):
        environ = {key: value for key, value in os.environ.items() if key not in base.SCRUB}
        environ['QUAL_CARD'] = base.CARD_X
        environ.update(HOME=self.dir.as_posix(), RESULTS=(self.dir / 'results').as_posix(), K64J_CARD_DRY_RUN='1',
                       KOPGRAFT64=self.graft.as_posix(), EXPECT_TTNNCPP_SHA256=sha(base.BINARY))
        environ.update(env)
        result = subprocess.run([base.BASH, base.RUNNER.as_posix()], env=environ, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        line = [line for line in result.stdout.splitlines() if line.startswith('### argv: ')]
        return shlex.split(line[0][len('### argv: '):])

    def harness_args(self, **env):
        argv = self.launch(**env)
        return argv[argv.index('card') + 1:]

    def parsed(self, **env):
        return card_b.parse_args(self.harness_args(**env))

    def test_the_watcher_pass_runs_the_one_head_combos(self):
        pair = self.parsed(WATCHER='1')
        self.assertEqual((pair.kv_heads, pair.combos, pair.trace_combos),
                         (2, [('G4B3', 0x21), ('G4B3', 0x23), ('G8B2', 0x27)], [('G4B3', 0x21), ('G8B2', 0x27)]))
        for spelling in ('--kv-heads 1', '--kv-heads=1', '--sections N,X --kv-heads 1 --seeds 0'):
            one = self.parsed(WATCHER='1', CARD_B_ARGS=spelling)
            self.assertEqual((one.kv_heads, one.combos, one.trace_combos, one.extents, one.seeds, one.no_timing,
                              one.watchdog),
                             (1, [('G4B3', 0x21), ('G4B3', 0x23), ('G8B2', 0x23)], [('G4B3', 0x21), ('G8B2', 0x23)],
                              [2304, 33024], one.seeds, True, 120.0), spelling)
        back = self.parsed(WATCHER='1', CARD_B_ARGS='--kv-heads 1 --kv-heads 2')          # the last one wins
        self.assertEqual((back.kv_heads, back.combos), (2, pair.combos))
        self.assertEqual(self.harness_args(WATCHER='1').count('--kv-heads'), 0)            # the pair's argv is unchanged

    def test_the_full_run_defaults_at_one_head(self):
        one = self.parsed(CARD_B_ARGS='--kv-heads 1 --seeds 0,1,2,3,4')
        self.assertEqual((one.kv_heads, one.seeds, one.sections, one.extents, one.starts, one.combos, one.trace_combos,
                          one.no_timing, one.deadline_s),
                         (1, [0, 1, 2, 3, 4], list(card_b.DEFAULT_SECTIONS), list(probe.EXTENTS), list(probe.STARTS),
                          card_b.default_combos(1), list(card_b.default_trace_combos(1)), False,
                          5400 - card_b.DEADLINE_MARGIN_S))
        pair = self.parsed(CARD_B_ARGS='--seeds 0,1,2,3,4')
        self.assertEqual((pair.kv_heads, pair.combos, pair.trace_combos),
                         (2, card_b.default_combos(), list(card_b.TRACE_COMBOS)))

    def job(self, name):
        """(C2_CARDM_HARNESS, C2_CARDM_ARGS, the env words) of a committed evidence-job template."""
        values = {}
        for line in (JOBS / name).read_text(encoding='utf-8').splitlines():
            if line and not line.startswith('#'):
                key, _, value = line.partition('=')
                values[key] = value
        return values

    def test_the_evidence_jobs_call_the_harness_with_the_flags_implemented(self):
        placeholders = {'@K64J_GRAFT_DIR@': self.graft.as_posix(), '@K64J_TTNNCPP_SHA256@': sha(base.BINARY)}
        expected = {
            'EV-W1-cb1-cb2a-watcher.env': ('--kv-heads 1 --sections N,X,M,K,L,T,K2,X7,Z --seeds 0 --extents 2304,131328 '
                                           '--variants normal,peaky --no-timing', True),
            'EV-F1-cb1.env': ('--kv-heads 1 --seeds 0,1,2,3,4', False),
            'EV-F2-cb2a.env': ('--kv-heads 1 --sections K2,X7,Z --seeds 0,1,2,3,4 --variants normal,peaky --no-timing',
                               False),
        }
        for name, (arguments, watcher) in expected.items():
            values = self.job(name)
            self.assertEqual((values['C2_CARDM_HARNESS'], values['C2_CARDM_ARGS']),
                             ('optimisation/ttnn-op/k64j/run_card_b.sh', arguments), name)
            words = dict(word.split('=', 1) for word in values['C2_CARDM_ENV'].split())
            self.assertEqual(words.get('K64J_HARNESS'), 'card', name)
            self.assertEqual(words.get('WATCHER') == '1', watcher, name)
            env = {key: placeholders.get(value, value) for key, value in words.items()
                   if key in ('WATCHER', 'KOPGRAFT64', 'EXPECT_TTNNCPP_SHA256', 'K64J_HARNESS')}
            args = self.parsed(CARD_B_ARGS=values['C2_CARDM_ARGS'], **env)
            with self.subTest(job=name):
                self.assertEqual((args.kv_heads, args.served_flags, args.compile_flags), (1, 0x23, 0x3))
                self.assertFalse({0x27, 0x2F} & {flags for _shape, flags in args.combos + args.trace_combos})
                if name.startswith('EV-W1'):
                    self.assertEqual((args.sections, args.seeds, args.extents, args.variants, args.no_timing),
                                     (['N', 'X', 'M', 'K', 'L', 'T', 'K2', 'X7', 'Z'], [0], [2304, 131328],
                                      ['normal', 'peaky'], True))
                    self.assertEqual((args.combos, args.trace_combos),
                                     ([('G4B3', 0x21), ('G4B3', 0x23), ('G8B2', 0x23)],
                                      [('G4B3', 0x21), ('G8B2', 0x23)]))
                elif name.startswith('EV-F1'):
                    self.assertEqual((args.sections, args.seeds, args.extents, args.combos, args.no_timing),
                                     (list(card_b.DEFAULT_SECTIONS), [0, 1, 2, 3, 4], list(probe.EXTENTS),
                                      card_b.default_combos(1), False))
                else:
                    self.assertEqual((args.sections, args.seeds, args.variants, args.no_timing),
                                     (['K2', 'X7', 'Z'], [0, 1, 2, 3, 4], ['normal', 'peaky'], True))
                    self.assertEqual((args.k2_sweep, args.cb2_extents, args.cb2_starts, args.z_families),
                                     (card_b.K2_SWEEP, list(card_b.CB2_EXTENTS), list(card_b.CB2_STARTS),
                                      list(card_b.Z_FAMILIES)))


class OneHeadFlow(unittest.TestCase):
    """The harness end to end on the fake at --kv-heads 1 (test_k64j_card_b.DryRunTests' and test_k64j_cb2a's setup)."""

    BASE = ['--capacity', '4352', '--extents', '512,2304,4352', '--starts', '7,255', '--seeds', '0',
            '--trace-families', '4', '--trace-references', '2', '--no-timing', '--kv-heads', '1']

    def setUp(self):
        import torch
        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        graft = base.make_graft(self.dir)
        self.kernels = graft / 'sdpa_decode' / 'device' / 'kernels'
        self.binary = graft / '_ttnncpp.so'

    def tearDown(self):
        self.tmp.cleanup()

    run_card = base.DryRunTests.run_card
    kinds = base.DryRunTests.kinds

    def test_cb1_passes_end_to_end_at_one_kv_head(self):
        fake = FakeExtentTtnn(self.torch, record_calls=True)
        status, report = self.run_card(fake)
        self.assertEqual((report.get('error'), report['failures'], report['warnings']), (None, [], []))
        self.assertEqual((status, report['passed'], report['decision']['verdict']), (0, True, 'PASS'))
        # What the recorder reads: kv_heads an int, the local heads, the combos without the slice, the sections.
        self.assertIs(type(report['kv_heads']), int)
        self.assertEqual((report['kv_heads'], report['local_heads'], report['fold_rows_per_token']), (1, 6, 6))
        self.assertEqual(report['combos'], ['G4B3:0x21', 'G4B3:0x23', 'G8B2:0x21', 'G8B2:0x23'])
        self.assertEqual(report['trace_combos'], ['G4B3:0x21', 'G8B2:0x23'])
        self.assertEqual(report['sections'], list(card_b.DEFAULT_SECTIONS))
        self.assertEqual(report['sections_done'], ['N/seed0', 'X/seed0', 'M/seed0', 'K/seed0', 'L/seed0', 'T/seed0'])
        kinds = self.kinds(report)
        self.assertEqual(kinds['refusal'], (8, 8))                       # the pair's seven and the q-slice's
        refusal = report['refusals']['0x27 q-slice at one KV head']
        self.assertTrue(refusal['refused'] and refusal['matched'], refusal)
        self.assertEqual(refusal['needle'], apply_factory_slice.HEADS_REFUSAL)
        # X: 10 entries over the four combos x 3 extents x 2 starts.
        self.assertEqual(kinds['extent_vs_reference'], (60, 60))
        self.assertEqual(kinds['mixed_vs_reference'], (6, 6))
        self.assertEqual(kinds['share_slot0'], (2, 2))                    # G4B3 0x23 and G8B2 0x23
        self.assertEqual((kinds['skip_live'], kinds['share_skip_slot0_wins']), ((4, 4), (1, 1)))
        self.assertEqual((kinds['trace_vs_eager'], kinds['trace_vs_reference']), ((8, 8), (4, 4)))
        self.assertEqual(kinds['trace_skip_live'], (4, 4))               # the non-share trace combo only
        self.assertEqual(kinds['trace_fence_vs_clean'], (12, 12))
        self.assertEqual(report['fence_extents'], {'G4B3:0x21': [512, 2304, 512], 'G8B2:0x23': [512, 512]})
        self.assertTrue(report['liveness'] and all(entry['live'] for entry in report['liveness']))
        # 6 folded rows per token: a four-token group is 24 rows, an eight-token group 48 (the pair's 48 and 96).
        self.assertEqual(sorted({(call['rows'], call['batches']) for call in fake.recorded}),
                         [(24, 1), (24, 3), (48, 2)])
        # No program carries the q-slice (0x4): every requested flag set is 0x1, 0x3, 0x21 or 0x23.
        flags = {program[0] for program in report['requested_programs']}
        self.assertEqual(flags, {0x1, 0x3, 0x21, 0x23})
        self.assertEqual({line['q_slice'] for line in report['extent_lines']}, {'false'})
        self.assertEqual({line['writer'] for line in report['extent_lines']}, {'writer_decode_qwen_slice.cpp'})
        self.assertIn(' kv_heads=1 k2_verdict=not_run ', report['verdict_line'])
        self.assertTrue(report['verdict_line'].startswith(
            'K64J_CARD verdict=PASS extent=60/60 mixed=6/6 share_slot0=3/3 trace=12/12 fence=12/12 skip=8/8 '
            'refusals=8/8 live=25/25 skipped_rows=unwritten'), report['verdict_line'])
        self.assertEqual((fake.closed, fake.live_traces_at_close), (True, 0))
        self.assertEqual(card.KV_HEADS, 2)                               # main restored the pair's geometry

    def test_the_pair_report_has_no_one_head_words(self):
        pair = [word for word in self.BASE if word not in ('--kv-heads', '1')]
        with mock.patch.object(type(self), 'BASE', pair):
            status, report = self.run_card(FakeExtentTtnn(self.torch), ['--sections', 'N'])
        self.assertEqual((status, report['failures'], report['decision']['verdict']), (0, [], 'PASS'))
        for key in ('kv_heads', 'local_heads', 'fold_rows_per_token'):
            self.assertNotIn(key, report)
        self.assertNotIn('kv_heads', report['verdict_line'])
        self.assertEqual(self.kinds(report)['refusal'], (7, 7))
        self.assertNotIn('0x27 q-slice at one KV head', report['refusals'])
        self.assertEqual(card.KV_HEADS, 2)

    def test_a_q_slice_the_binary_accepts_at_one_head_fails_the_refusal(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'accept_slice_one_head'}),
                                       ['--sections', 'N'])
        self.assertEqual((status, report['failures'], report['decision']['verdict']), (1, [], 'FAIL'))
        self.assertEqual(self.kinds(report)['refusal'], (7, 8))
        self.assertFalse(report['refusals']['0x27 q-slice at one KV head']['refused'])
        self.assertIn('refusals=7/8', report['verdict_line'])

    def test_a_program_that_ignores_the_word_fails_at_one_head(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'ignore_word'}),
                                       ['--sections', 'X', '--combos', 'G8B2:0x23', '--starts', '7'])
        self.assertEqual((status, report['failures'], report['decision']['verdict']), (1, [], 'FAIL'))
        self.assertEqual(self.kinds(report)['extent_vs_reference'], (2, 6))         # only E = C is right

    def test_share_entries_on_their_own_slot_fail_at_one_head(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'own_slot'}),
                                       ['--sections', 'M,K', '--combos', 'G4B3:0x21,G4B3:0x23', '--starts', '7'])
        self.assertEqual((report['failures'], report['decision']['verdict']), ([], 'FAIL'))
        self.assertEqual(self.kinds(report)['share_slot0'], (0, 1))
        self.assertEqual(self.kinds(report)['mixed_vs_reference'], (3, 3))

    def test_a_dead_liveness_control_decides_nothing_at_one_head(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'clamp'}),
                                       ['--sections', 'L', '--combos', 'G8B2:0x23'])
        self.assertEqual((status, report['decision']['verdict']), (1, 'NO-DECISION'))

    def test_a_binary_that_never_logs_f22_was_not_executed_at_one_head(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'no_f22'}),
                                       ['--sections', 'X', '--combos', 'G4B3:0x21', '--extents', '2304',
                                        '--starts', '7'])
        self.assertEqual((status, report['decision']['verdict']), (1, 'NO-DECISION'))
        self.assertEqual(len(report['failures']), 1)
        self.assertTrue(report['failures'][0].startswith('extent log: flags=0x21 B=3'), report['failures'])


class OneHeadCB2aFlow(unittest.TestCase):
    """CB2a at one head on the fake: K2's native decode at 6 local heads on 1 KV head against the G8B2 0x23 bundle."""

    BASE = ['--capacity', '4352', '--extents', '2304', '--seeds', '0', '--no-timing', '--sections', 'K2,X7,Z',
            '--cb2-extents', '2304,4352', '--kv-heads', '1']
    SMALL = cb2a.SMALL

    setUp = cb2a.DryRunTests.setUp
    tearDown = cb2a.DryRunTests.tearDown
    run_card = cb2a.DryRunTests.run_card
    kinds = cb2a.DryRunTests.kinds

    def test_cb2a_passes_end_to_end_at_one_kv_head(self):
        fake = FakeExtentTtnn(self.torch, record_calls=True)
        status, report = self.run_card(fake, ['--variants', 'peaky', '--z-starts', '0,32,240'])
        self.assertEqual((report.get('error'), report['failures'], report['warnings']), (None, [], []))
        self.assertEqual((status, report['passed'], report['decision']['verdict'], report['decision']['k2']),
                         (0, True, 'PASS', 'REDUCED-PASS'))
        self.assertEqual((report['kv_heads'], report['local_heads']), (1, 6))
        # K2's coverage is the pair's: the same tickets (the design's set), at 6 local heads.
        self.assertEqual(report['decision']['k2_coverage'], dict(full=False, covered=183, design=1980,
                                                                 short=['seeds', 'variants', 'family']))
        kinds = self.kinds(report)
        self.assertEqual(kinds['k2_native_vs_extent'], (183, 183))
        self.assertEqual((kinds['x7_narrow_vs_wide'], kinds['x7_extent_vs_wide']), ((10, 10), (10, 10)))
        self.assertEqual((kinds['z_trace_vs_eager'], kinds['z_trace_vs_reference']), ((45, 45), (45, 45)))
        self.assertEqual(report['z_families'], list(card_b.Z_FAMILIES))
        self.assertTrue(all(entry['live'] for entry in report['liveness']))
        self.assertEqual({entry['rows'] for entry in report['liveness'] if entry['section'] == 'K2'}, {6})
        self.assertEqual({entry['rows'] for entry in report['liveness'] if entry['section'] == 'X7'}, {48})
        # The report records the native decode's local-head and KV-head counts, and what is unverified at one head.
        native = report['native_decode']
        self.assertEqual((native['local_heads'], native['kv_heads'], native['query_shape']), (6, 1, [1, 1, 6, 256]))
        self.assertEqual({key: value for key, value in native.items()
                          if key not in ('local_heads', 'kv_heads', 'query_shape')},
                         json.loads(json.dumps(card_b.NATIVE_DECODE)))
        self.assertEqual(report['unverified'], list(card_b.unverified_items(1)))
        self.assertEqual(report['cb2a']['served'], dict(flags='0x23', compile_flags='0x3', rows=8, batch=2,
                                                        offsets=[0, 8], k_chunk_size=256))
        self.assertIn(' kv_heads=1 k2_verdict=REDUCED-PASS k2_coverage=183/1980 x7=20/20 z=90/90 z_families=15 k4=not_run',
                      report['verdict_line'])
        # The calls: the native one as the model issues it, now one row of 6 heads; the subject and its compile-time twin on
        # the 48 folded rows of a G8B2 bundle, at 0x23 and 0x3 (no q-slice, never 0x27 or 0x7).
        native_calls = [call for call in fake.recorded if call['program_config']['q_chunk_size'] == card.LEGACY]
        subject = [call for call in fake.recorded
                   if call['program_config']['q_chunk_size'] == card.MAGIC | card_b.SERVED_FLAGS_ONE_HEAD]
        compile_time = [call for call in fake.recorded
                        if call['program_config']['q_chunk_size'] == card.MAGIC | card_b.COMPILE_FLAGS_ONE_HEAD]
        self.assertEqual(len(fake.recorded), len(native_calls) + len(subject) + len(compile_time))
        self.assertTrue(native_calls and subject and compile_time)
        for call in native_calls:
            self.assertEqual((call['options'], call['is_causal'], call['memory_config'], call['rows'], call['batches']),
                             (sorted(card_b.NATIVE_CALL_KWARGS), None, 'l1', 6, 1))
        for group, expected in ((subject, cb2a.SERVED_KWARGS), (compile_time, cb2a.SERVED_KWARGS - {'cur_pos_tensor'})):
            for call in group:
                self.assertEqual((call['options'], call['is_causal'], call['memory_config'], call['rows'],
                                  call['batches']), (sorted(expected), False, 'l1', 48, 2))
        self.assertEqual(card.KV_HEADS, 2)

    def test_a_half_tile_q_that_moves_fails_k2_at_one_head(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'half_tile'}),
                                       ['--sections', 'K2,X7'] + self.SMALL)
        self.assertEqual((report['failures'], report['decision']['verdict'], report['decision']['k2']),
                         ([], 'FAIL', 'FAIL'))
        self.assertEqual(self.kinds(report)['k2_native_vs_extent'][0], 0)
        self.assertEqual(self.kinds(report)['x7_narrow_vs_wide'], (2, 2))

    def test_a_mask_read_at_the_capacity_fails_x7_at_one_head(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'tail_at_capacity'}),
                                       ['--sections', 'X7', '--cb2-extents', '2304,4352', '--cb2-starts', '0,240'])
        self.assertEqual((report['failures'], report['decision']['verdict']), ([], 'FAIL'))
        kinds = self.kinds(report)
        self.assertEqual((kinds['x7_narrow_vs_wide'], kinds['x7_extent_vs_wide']), ((0, 4), (0, 4)))

    def test_a_program_that_ignores_the_word_fails_z_at_one_head(self):
        status, report = self.run_card(FakeExtentTtnn(self.torch, broken={'ignore_word'}),
                                       ['--sections', 'Z'] + cb2a.Z_PAIR)
        self.assertEqual((report['failures'], report['decision']['verdict']), ([], 'FAIL'))
        self.assertLess(self.kinds(report)['z_trace_vs_reference'][0], self.kinds(report)['z_trace_vs_reference'][1])


if __name__ == '__main__':
    unittest.main()
