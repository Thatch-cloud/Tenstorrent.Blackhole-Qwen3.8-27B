"""Prefill lever S1 (oneq): the factory patch, its host-executed C++, the work-split planner, the time model and the
protocol model on the oneq topology (optimisation/ttnn-op/sdpa_prefill_oneq).

  patch        apply_factory_ps.py: PS0-PS5 over the PF factory (and over the served one), inversion, anchors, markers,
               the protected block, the added lines, the word decoder
  stubs        g++ -fsyntax-only of the cut-out blocks (skipped without g++), and the harness's teeth
  host exec    the REAL work-split / F1 / F4 / reader-range C++ of the patched factory compiled and run for many shapes
               and program words (skipped without g++), compared per core with oneq_planner.plan()
  planner      coverage, chains, the paired baseline against the existing protocol model, TP2, odd and over-grid shapes
  time model   the paired SDPA of profile run 38051000905 (chunks 1, 32, 63) and the estimate table (0.47 / 7.88 / 29.80 s)
  protocol     pf_protocol_model.simulate on the oneq round lists (single unit, Q CB of one buffer): terminates, content
               exact, the checker's teeth, and an unequal-length group always hangs

Run from scripts/ci:  python -B -m unittest test_sdpa_oneq_factory
"""

import difflib
import hashlib
import itertools
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
ONEQ = ROOT / 'optimisation' / 'ttnn-op' / 'sdpa_prefill_oneq'
CHAIN = ROOT / 'optimisation' / 'ttnn-op' / 'sdpa_prefill_chain'
for path in (ONEQ, CHAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import apply_factory_pf as pf  # noqa: E402
import apply_factory_ps as ps  # noqa: E402
import host_exec_oneq as hx  # noqa: E402
import oneq_planner as op  # noqa: E402
import pf_protocol_model as model  # noqa: E402
import stub_compile_oneq as stubs  # noqa: E402

SERVED = (CHAIN / 'fixtures' / 'sdpa_program_factory.fd8c0676.cpp').read_bytes()
HAVE_GXX = stubs.find_gxx() is not None
NL = chr(10)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class PatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pf_bytes = pf.patch(SERVED)
        cls.ps_bytes = ps.patch(SERVED)
        cls.text = cls.ps_bytes.decode('utf-8')

    def test_bases_are_the_recorded_ones(self):
        self.assertEqual(sha(SERVED), ps.SERVED_FACTORY)
        self.assertEqual(sha(self.pf_bytes), ps.PF_FACTORY)
        self.assertEqual(ps.PF_FACTORY, 'bfab8558d889ad215f0e9ee732c75a4142be1a1e7f37810f4ca8c5e3c73bdf65')

    def test_output_is_the_recorded_sha(self):
        self.assertEqual(sha(self.ps_bytes), ps.PS_FACTORY)
        self.assertNotEqual(ps.PS_FACTORY, ps.PF_FACTORY)

    def test_served_and_pf_inputs_give_the_same_bytes(self):
        self.assertEqual(ps.patch(self.pf_bytes), self.ps_bytes)

    def test_edits_invert(self):
        self.assertEqual(ps.unpatch(self.ps_bytes), self.pf_bytes)
        self.assertEqual(pf.unpatch(ps.unpatch(self.ps_bytes)), SERVED)

    def test_only_the_six_edits_differ_from_pf(self):
        old = self.pf_bytes.decode('utf-8').splitlines()
        new = self.text.splitlines()
        removed, added, hunks = [], [], 0
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
            if tag == 'equal':
                continue
            hunks += 1
            removed += old[i1:i2]
            added += new[j1:j2]
        self.assertEqual(hunks, len(ps.EDITS))
        # Exactly the three modified served-PF lines are removed (PS0's pair switch, PS2's accepted set, PS3's predicate).
        self.assertEqual([line.strip() for line in removed],
                         [ps.PS0_OLD.strip(), ps.PS2_OLD.strip(), ps.PS3_OLD.strip()])
        self.assertTrue(all(line.startswith('    ') for line in added), 'a new line outside the function body')

    def test_every_new_line_is_inside_create_descriptor(self):
        body = self.text.index(ps.FUNCTION_LINE)
        for label, _line, _old, new in ps.EDITS:
            self.assertGreater(self.text.index(new), body, label)

    def test_no_cr_and_no_namespace_scope_addition(self):
        self.assertNotIn(chr(13).encode(), self.ps_bytes)
        for label, _line, _old, new in ps.EDITS:
            for line in new.splitlines():
                self.assertTrue(line.startswith('    '), '%s: %r' % (label, line))

    def test_the_draft_fp32_block_is_untouched(self):
        self.assertEqual(pf.protected_block(self.text), pf.protected_block(SERVED.decode('utf-8')))

    def test_anchors_start_at_their_recorded_pf_lines(self):
        pf_text = self.pf_bytes.decode('utf-8')
        for label, line, old, _new in ps.EDITS:
            self.assertEqual(pf_text.count(old), 1, label)
            self.assertEqual(pf_text[:pf_text.index(old)].count(NL) + 1, line, label)

    def test_drift_and_foreign_input_are_refused(self):
        with self.assertRaises(ps.Refusal):
            ps.patch(b'not a factory')
        drifted = self.pf_bytes.decode('utf-8').replace('    // Update mcast_enabled', '    // extra line' + NL + '    // Update mcast_enabled')
        with self.assertRaises(ps.Refusal):
            ps.patch(drifted.encode('utf-8'))      # sha is not PF any more
        # an anchor that moved by one line, with the sha check bypassed
        moved = self.pf_bytes.decode('utf-8').replace(ps.PS0_OLD, NL + ps.PS0_OLD)
        with self.assertRaises(ps.Refusal):
            ps.check_anchors(moved)

    def test_cli_writes_the_recorded_factory_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'served.cpp'
            source.write_bytes(SERVED)
            out = Path(directory) / 'out.cpp'
            self.assertEqual(ps.main([str(source), '--out', str(out)]), 0)
            self.assertEqual(sha(out.read_bytes()), ps.PS_FACTORY)
            again = Path(directory) / 'again.cpp'
            self.assertEqual(ps.main([str(out), '--out', str(again)]), 0)
            self.assertEqual(again.read_bytes(), out.read_bytes())

    def test_new_qwen_strings_are_the_two_the_binary_check_expects(self):
        self.assertEqual(ps.NEW_QWEN_STRINGS, (ps.FATAL_TEXT, ps.LOG_TEXT))
        for text in ps.NEW_QWEN_STRINGS:
            self.assertEqual(self.text.count('"%s"' % text), 1, text)
            self.assertNotIn(text, self.pf_bytes.decode('utf-8'))
        self.assertEqual(ps.FATAL_TEXT, op.FATAL_TEXT)

    def test_constants_cross_the_files_unchanged(self):
        self.assertEqual(ps.PF_TAG, 0x5EFA0000)
        self.assertEqual(ps.FLAG_ONEQ, 0x8)
        self.assertEqual(op.FLAG_ONEQ, ps.FLAG_ONEQ)
        self.assertEqual(op.PF_TAG, ps.PF_TAG)
        self.assertIn('kQwenOqTag = 0x5EFA0000u', self.text)
        self.assertIn('kQwenOqBits = 0x9u', self.text)
        self.assertIn('kQwenPfOneQ = 0x8u', self.text)
        self.assertEqual(ps.ONEQ_BITS, 0x9)
        self.assertEqual(ps.ONEQ_FLAGS, (0x9, 0xB, 0xD, 0xF))

    def test_decode_word(self):
        self.assertEqual(ps.decode_word(16), (False, 0, False))
        self.assertEqual(ps.decode_word(0x5EFA0003), (True, 3, False))
        self.assertEqual(ps.decode_word(0x5EFA000B), (True, 0xB, True))
        self.assertEqual(ps.decode_word(0x5EFA000F), (True, 0xF, True))
        for word in (0x5EFA0008, 0x5EFA0000, 0x5EFA0010, 0x5EFA1003):
            with self.assertRaises(ValueError):
                ps.decode_word(word)
        with self.assertRaises(ValueError):
            ps.decode_word(0x5EFA0103)                       # test flag without the env
        self.assertEqual(ps.decode_word(0x5EFA010B, test_env='1'), (True, 0x10B, True))

    def test_the_early_decode_agrees_with_the_planner_for_every_word(self):
        for word in range(0x5EFA0000, 0x5EFA0400):
            self.assertEqual(op.one_q_word(word), word & 0x9 == 0x9, hex(word))
        self.assertFalse(op.one_q_word(16))
        self.assertFalse(op.one_q_word(0x00000009))
        self.assertFalse(op.one_q_word(0x5EFB0009))


@unittest.skipUnless(HAVE_GXX, 'no g++')
class StubTests(unittest.TestCase):
    def test_blocks_compile_against_the_stubs(self):
        code, output = stubs.compile_harness(stubs.find_gxx(), stubs.harness())
        self.assertEqual(code, 0, output)

    def test_the_harness_has_teeth(self):
        text = stubs.patched_text()
        gxx = stubs.find_gxx()
        for old, new in (('(global_q_pair_distribute || (qwen_one_q', '(global_q_pair_distribute || (qwen_one_x'),
                         ('max_global_q_chunks_per_core == 1', 'max_global_q_chunks_per_cor == 1'),
                         ('kQwenPfKvChain | kQwenPfInjBatch | kQwenPfNocOrder | kQwenPfOneQ |',
                          'kQwenPfKvChain | kQwenPfInjBatch | kQwenPfNocOrder | kQwenPfOneX |'),
                         ('total_q_chunks, num_cores, qwen_chains, qwen_members);', 'total_q_chunks, num_cores, qwen_chainz, qwen_members);')):
            self.assertEqual(text.count(old), 1, old)
            code, output = stubs.compile_harness(gxx, stubs.harness(text.replace(old, new)))
            self.assertNotEqual(code, 0, old)


CASES = []
for _word in (0x5EFA0003, 0x5EFA0001, 0x5EFA0007, 0x5EFA0009, 0x5EFA000B, 0x5EFA000D, 0x5EFA000F):
    for _rows in (2048, 1024, 512):
        for _grid in ((13, 10), (11, 10)):
            CASES.append((6, 1, _rows, _grid, _word))                        # TP4
for _word in (0x5EFA0003, 0x5EFA000B):
    for _rows in (2048, 1024, 512):
        for _grid in ((11, 10), (13, 10)):
            CASES.append((12, 2, _rows, _grid, _word))                       # TP2
CASES += [(8, 1, 2048, (13, 10), 0x5EFA000B), (8, 1, 2048, (11, 10), 0x5EFA000B), (4, 1, 2048, (11, 10), 0x5EFA000B),
          (4, 1, 2048, (11, 10), 0x5EFA0003), (6, 1, 1920, (13, 10), 0x5EFA000B), (6, 1, 1920, (13, 10), 0x5EFA0003),
          (6, 1, 2048, (13, 10), 16), (6, 1, 2048, (11, 10), 16), (12, 2, 2048, (11, 10), 16),
          (6, 1, 2048, (13, 10), 0x5EFA0008), (6, 1, 2048, (13, 10), 0x5EFA0004), (6, 1, 2048, (13, 10), 0x5EFA0013),
          (6, 2, 2048, (13, 10), 0x5EFA000B), (6, 1, 2048, (8, 8), 0x5EFA000B), (6, 1, 2048, (8, 12), 0x5EFA000B),
          (6, 1, 2048, (8, 12), 0x5EFA0003)]


@unittest.skipUnless(HAVE_GXX, 'no g++')
class HostExecTests(unittest.TestCase):
    """The patched factory's own C++ lines, run: per core range and chain words, the logs, the refusals."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp(prefix='oneq-exec-')
        cls.exe = hx.build(cls.directory)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def expected(self, nqh, nkh, rows, grid, word):
        """('fatal', text) or ('ok', plan or None) for a call, by the Python reading of the factory."""
        try:
            chain, flags, oneq = ps.decode_word(word)
        except ValueError as error:
            return 'fatal', str(error)
        the_plan = op.plan(nqh, nkh, rows, 128, grid, oneq=oneq, noc_order=bool(flags & 4))
        if not chain:
            return 'ok', the_plan
        q_num_chunks = rows // 128
        envelope = (nkh == nkh and nqh % nkh == 0 and (the_plan['pair'] or (oneq and q_num_chunks % 2 == 0)))
        if not envelope:
            return 'fatal', '[QWEN-SDPA-PF] kv_chain outside its qualified envelope'
        if oneq and the_plan['refusal']:
            return 'fatal', the_plan['refusal']
        return 'ok', the_plan

    def test_every_case_matches_the_planner(self):
        seen = {'fatal': 0, 'chain': 0, 'plain': 0}
        for nqh, nkh, rows, grid, word in CASES:
            label = 'nqh=%d nkh=%d rows=%d grid=%r word=%#x' % (nqh, nkh, rows, grid, word)
            outcome = hx.run(self.exe, nqh, nkh, rows, 128, grid, word)
            kind, expected = self.expected(nqh, nkh, rows, grid, word)
            if kind == 'fatal':
                seen['fatal'] += 1
                self.assertEqual(outcome['fatal'], expected, label)
                continue
            self.assertIsNone(outcome['fatal'], label)
            self.assertEqual(len(outcome['cores']), expected['num_cores'], label)
            for core in outcome['cores']:
                start, count = expected['ranges'][core['core']]
                self.assertEqual((core['start'], core['count']), (start, count), '%s core %d' % (label, core['core']))
            chain = (word & 0xFFFF0000) == ps.PF_TAG
            if not chain:
                seen['plain'] += 1
                self.assertEqual(outcome['logs'], [], label)
                self.assertFalse(any(core['participates'] for core in outcome['cores']), label)
                continue
            seen['chain'] += 1
            self.assertEqual(self.chain_words(expected, outcome, label), None)
            flags = word & 0xFFFF
            line = '[QWEN-SDPA-PF] flags=%#x kv_chain=1 chains=%d members=%d order=%s' % (
                flags, expected['chain_count'], expected['member_count'], 'noc' if flags & 4 else 'raster')
            self.assertEqual(outcome['logs'][0], line, label)
            if flags & ps.FLAG_ONEQ:
                self.assertEqual(outcome['logs'][1], '[QWEN-SDPA-PF] oneq=1 q_chunks=%d cores=%d chunks_per_core=1 '
                                 'chains=%d members=%d' % (expected['total'], expected['num_cores'],
                                                           expected['chain_count'], expected['member_count']), label)
                self.assertEqual(len(outcome['logs']), 2, label)
            else:
                self.assertEqual(len(outcome['logs']), 1, label)
        self.assertGreater(seen['fatal'], 5)
        self.assertGreater(seen['chain'], 40)
        self.assertGreater(seen['plain'], 2)

    def chain_words(self, the_plan, outcome, label):
        """The 14-word chain block the factory hands each core, against the planner's chains."""
        by_core = {core['core']: core for core in outcome['cores']}
        role = {}
        for key, members in the_plan['chains'].items():
            for position, core in enumerate(members):
                role[core] = (key, position, members)
        for core, words in by_core.items():
            if core not in role:
                self.assertEqual((words['participates'], words['injector'], words['sink']), (0, 0, 0), '%s %d' % (label, core))
                continue
            key, position, members = role[core]
            last = position + 1 == len(members)
            units = len(key) // 3
            self.assertEqual(words['participates'], 1, label)
            self.assertEqual((words['injector'], words['sink']), (int(position == 0), int(last)), label)
            self.assertEqual((words['batch'], words['head'], words['qstart'], words['qcount']), (key[0], key[1], key[2], units),
                             label)
            grid = the_plan['grid']
            prev = op.physical(members[position - 1], grid) if position else (0, 0)
            nxt = (0, 0) if last else op.physical(members[position + 1], grid)
            self.assertEqual((words['prev'], words['next']), (prev, nxt), '%s core %d' % (label, core))
            self.assertEqual(words['nextq'], 0 if last else units, label)
        return None

    def test_tp4_2048_is_96_cores_in_16_chains_of_6_and_the_paired_one_is_the_profiles(self):
        paired = hx.run(self.exe, 6, 1, 2048, 128, (13, 10), 0x5EFA0003)
        self.assertEqual(paired['logs'], ['[QWEN-SDPA-PF] flags=0x3 kv_chain=1 chains=8 members=48 order=raster'])
        self.assertEqual(sum(1 for core in paired['cores'] if core['count']), 48)
        self.assertEqual({core['count'] for core in paired['cores'] if core['count']}, {2})
        oneq = hx.run(self.exe, 6, 1, 2048, 128, (13, 10), 0x5EFA000B)
        self.assertEqual(oneq['logs'][0], '[QWEN-SDPA-PF] flags=0xb kv_chain=1 chains=16 members=96 order=raster')
        busy = [core for core in oneq['cores'] if core['count']]
        self.assertEqual(len(busy), 96)
        self.assertEqual({core['count'] for core in busy}, {1})
        self.assertEqual(sum(core['injector'] for core in busy), 16)
        self.assertEqual(sum(core['sink'] for core in busy), 16)
        self.assertEqual(oneq['split'], {'total': '96', 'max_per_core': '1', 'q_buffer_factor': '1'})
        self.assertEqual(paired['split'], {'total': '96', 'max_per_core': '2', 'q_buffer_factor': '2'})

    def test_tp2_2048_is_refused_by_name_and_tp2_1024_runs(self):
        for grid in ((11, 10), (13, 10)):
            outcome = hx.run(self.exe, 12, 2, 2048, 128, grid, 0x5EFA000B)
            self.assertEqual(outcome['fatal'], '[QWEN-SDPA-PF] oneq needs one q chunk per core: 192 q chunks on %d cores'
                             % (grid[0] * grid[1]))
            self.assertEqual(outcome['cores'], [])
        outcome = hx.run(self.exe, 12, 2, 1024, 128, (11, 10), 0x5EFA000B)
        self.assertIsNone(outcome['fatal'])
        self.assertEqual(outcome['logs'][0], '[QWEN-SDPA-PF] flags=0xb kv_chain=1 chains=16 members=96 order=raster')

    def test_the_off_path_is_the_pf_factory(self):
        # For every word without the oneq bits the patched blocks behave as the PF ones: same ranges, same chains, same
        # log (the planner with oneq=False is the PF reading, and the harness above showed the factory equals it).
        for word in (16, 0, 0x5EFA0001, 0x5EFA0003, 0x5EFA0007):
            for nqh, nkh in ((6, 1), (12, 2)):
                outcome = hx.run(self.exe, nqh, nkh, 2048, 128, (13, 10), word)
                the_plan = op.plan(nqh, nkh, 2048, 128, (13, 10), oneq=False)
                self.assertEqual([(c['start'], c['count']) for c in outcome['cores']], the_plan['ranges'])

    def test_a_bad_word_is_refused_before_anything_else(self):
        for word in (0x5EFA0008, 0x5EFA0010, 0x5EFA0013, 0x5EFA0100):
            outcome = hx.run(self.exe, 6, 1, 2048, 128, (13, 10), word)
            self.assertIsNotNone(outcome['fatal'], hex(word))
            self.assertTrue(outcome['fatal'].startswith('[QWEN-SDPA-PF] '), outcome['fatal'])


class PlannerTests(unittest.TestCase):
    def test_zigzag_is_a_bijection_for_every_head_length(self):
        for n in range(1, 40):
            self.assertEqual(sorted(op.zigzag(pos, n) for pos in range(n)), list(range(n)), n)

    def test_oneq_covers_every_head_and_q_chunk_exactly_once(self):
        for nqh, nkh, rows, grid in ((6, 1, 2048, (13, 10)), (6, 1, 2048, (11, 10)), (6, 1, 1024, (13, 10)),
                                     (6, 1, 512, (11, 10)), (12, 2, 1024, (11, 10)), (8, 1, 2048, (13, 10)),
                                     (4, 1, 2048, (8, 8))):
            the_plan = op.plan(nqh, nkh, rows, 128, grid, oneq=True)
            self.assertIsNone(the_plan['refusal'], (nqh, rows, grid))
            seen = [unit for units in the_plan['units'] for unit in units]
            self.assertEqual(sorted(seen), [(0, h, q) for h in range(nqh) for q in range(rows // 128)])
            self.assertTrue(all(len(units) <= 1 for units in the_plan['units']))
            self.assertEqual(the_plan['busy'], nqh * rows // 128)

    def test_paired_covers_every_unit_too(self):
        the_plan = op.plan(6, 1, 2048, 128, (13, 10))
        seen = [unit for units in the_plan['units'] for unit in units]
        self.assertEqual(sorted(seen), [(0, h, q) for h in range(6) for q in range(16)])
        self.assertEqual(the_plan['busy'], 48)
        self.assertEqual({len(units) for units in the_plan['units'] if units}, {2})

    def test_chain_members_share_one_unit_and_cover_all_heads(self):
        the_plan = op.plan(6, 1, 2048, 128, (13, 10), oneq=True)
        self.assertEqual(the_plan['chain_count'], 16)
        self.assertEqual(the_plan['member_count'], 96)
        for key, members in the_plan['chains'].items():
            self.assertEqual(len(members), 6)
            heads = []
            for core in members:
                (unit,) = the_plan['units'][core]
                self.assertEqual((unit[0], unit[1] // 6, unit[2]), key)
                heads.append(unit[1])
            self.assertEqual(sorted(heads), list(range(6)))

    def test_chains_have_one_round_length_so_a_mixed_group_always_hangs(self):
        # With ONE unit per core, two members of a group have the same unit => the same rounds; any two different q chunks
        # have different round counts (C + q + 1), unlike the paired scheme's equal-length divergence (m and m').
        for context in (0, 128, 1920, 2048, 65536, 126976):
            counts = {q: op.blocks(q, context) for q in range(16)}
            self.assertEqual(len(set(counts.values())), 16, context)

    def test_oneq_over_the_grid_is_refused_with_the_factorys_text(self):
        the_plan = op.plan(12, 2, 2048, 128, (13, 10), oneq=True)
        self.assertEqual(the_plan['refusal'], '[QWEN-SDPA-PF] oneq needs one q chunk per core: 192 q chunks on 130 cores')
        self.assertIsNone(op.plan(12, 2, 2048, 128, (13, 10))['refusal'])
        self.assertIsNone(op.plan(6, 1, 2048, 128, (13, 10), oneq=True)['refusal'])

    def test_the_paired_split_is_the_existing_protocol_models(self):
        for nqh, nkh, rows, grid in ((12, 2, 2048, (11, 10)), (12, 2, 1024, (11, 10)), (6, 1, 2048, (13, 10)),
                                     (6, 1, 2048, (11, 10)), (6, 1, 1024, (13, 10))):
            geo = model.geometry(NQH=nqh, NKH=nkh, rows=rows, grid=grid)
            self.assertEqual(op.plan(nqh, nkh, rows, 128, grid)['ranges'],
                             model.core_ranges(geo['num_cores'], geo['total'], geo['pair']))
            self.assertEqual(op.plan(nqh, nkh, rows, 128, grid)['units'], model.unit_lists(geo))
            topo = model.g6_topology(geo)
            self.assertEqual(op.plan(nqh, nkh, rows, 128, grid)['chain_count'], topo['chains'])
            self.assertEqual(op.plan(nqh, nkh, rows, 128, grid)['member_count'], topo['members'])

    def test_the_oneq_split_is_the_existing_models_non_pair_branch(self):
        for nqh, nkh, rows, grid in ((6, 1, 2048, (13, 10)), (6, 1, 2048, (11, 10)), (12, 2, 1024, (11, 10))):
            geo = dict(model.geometry(NQH=nqh, NKH=nkh, rows=rows, grid=grid), pair=False)
            the_plan = op.plan(nqh, nkh, rows, 128, grid, oneq=True)
            self.assertEqual(the_plan['ranges'], model.core_ranges(geo['num_cores'], geo['total'], False))
            self.assertEqual(the_plan['units'], model.unit_lists(geo))
            topo = model.g6_topology(geo)
            self.assertEqual((the_plan['chain_count'], the_plan['member_count']), (topo['chains'], topo['members']))

    def test_noc_order_is_a_permutation_with_no_greater_cost(self):
        raster = op.plan(6, 1, 2048, 128, (13, 10), oneq=True)
        noc = op.plan(6, 1, 2048, 128, (13, 10), oneq=True, noc_order=True)
        for key, members in noc['chains'].items():
            self.assertEqual(sorted(members), sorted(raster['chains'][key]))
            phys = [op.physical(core, (13, 10)) for core in members]
            base = [op.physical(core, (13, 10)) for core in raster['chains'][key]]
            self.assertLessEqual(op.order_cost(phys, range(6)), op.order_cost(base, range(6)))

    def test_q_buffer_factor_is_one_only_for_one_chunk_per_core(self):
        self.assertEqual(op.plan(6, 1, 2048, 128, (13, 10))['q_buffer_factor'], 2)
        self.assertEqual(op.plan(6, 1, 2048, 128, (13, 10), oneq=True)['q_buffer_factor'], 1)


class TimeModelTests(unittest.TestCase):
    paired = op.plan()
    oneq = op.plan(oneq=True)

    def test_paired_model_matches_the_profile_chunks(self):
        # chunk 1, 32, 63 of run 38051000905: measured per attention layer (us) vs the 2-chunks-per-core block model.
        for chunk, measured in ((1, 750.4), (32, 14177.3), (63, 27604.4)):
            modelled = op.layer_us(self.paired, chunk * 2048)
            self.assertLess(abs(modelled - measured) / measured, 0.003, (chunk, modelled, measured))

    def test_critical_blocks_are_the_closed_forms(self):
        for context in (0, 2048, 65536, 251904):
            k = context // 128
            self.assertEqual(op.critical_blocks(self.paired, context), 2 * k + 17)
            self.assertEqual(op.critical_blocks(self.oneq, context), k + 16)

    def test_busiest_oneq_core_is_the_last_q_chunk(self):
        blocks = op.core_blocks(self.oneq, 4096)
        busiest = max(range(len(blocks)), key=blocks.__getitem__)
        self.assertEqual(self.oneq['units'][busiest][0][2], 15)

    def test_estimates_are_the_analysis_table(self):
        table = {tokens: (floor, pessimistic) for tokens, _chunks, floor, pessimistic in op.estimate_table()}
        self.assertAlmostEqual(table[32768][0], 0.47, places=2)
        self.assertAlmostEqual(table[131072][0], 7.88, places=2)
        self.assertAlmostEqual(table[253952][0], 29.80, places=2)
        # At the 16.5 us step Q2 measured for a 16-chain geometry the saving shrinks but stays well above half of the best case.
        self.assertGreater(table[131072][1], 0.7 * table[131072][0])
        self.assertLess(table[131072][1], table[131072][0])
        self.assertAlmostEqual(table[131072][1], 6.09, places=2)
        self.assertAlmostEqual(table[253952][1], 23.10, places=2)

    def test_oneq_is_never_slower_in_the_model(self):
        for context in range(0, 252000, 2048):
            self.assertLess(op.layer_us(self.oneq, context), op.layer_us(self.paired, context))

    def test_tp2_geometry_has_no_oneq_gain_because_it_is_refused(self):
        tp2 = op.plan(12, 2, 2048, 128, (11, 10))
        self.assertEqual(tp2['busy'], 96)                    # paired TP2 already uses 96 of 110 cores
        self.assertEqual(op.critical_blocks(tp2, 0), 2 * 0 + 17)


class ProtocolTests(unittest.TestCase):
    """pf_protocol_model.simulate on the oneq group: one unit of C + q + 1 rounds per member, a one-buffer Q CB."""

    @staticmethod
    def rounds(context_chunks, q):
        ext = model.extents(context_chunks)
        return model.expand(model.round_list(context_chunks, [(0, 0, q)], 6, ext))

    def simulate(self, context_chunks, q, seed, variant='ok', mutation=None):
        with mock.patch.object(model, 'served_group_rounds', lambda c, m, g=0, geo=None: self.rounds(c, m)), \
                mock.patch.object(model, 'Q_CB_TILES', 32):
            return model.simulate(context_chunks, q, seed, variant, mutation=mutation)

    @staticmethod
    def check(result):
        with mock.patch.object(model, 'Q_CB_TILES', 32):          # check_ok compares the drained Q CB with the constant
            return model.check_ok(result)

    def test_round_list_is_context_plus_q_plus_one(self):
        for context in (0, 1, 15, 16, 33, 64, 992):
            for q in (0, 7, 15):
                rounds = self.rounds(context, q)
                self.assertEqual(len(rounds), context + q + 1)
                self.assertEqual([content[2] for _u, _k, _f, content in rounds], list(range(context + q + 1)))
                self.assertTrue(rounds[0][2])

    def test_ok_terminates_with_exact_content(self):
        for context, q in itertools.product((0, 1, 15, 16, 33, 64), (0, 3, 7, 15)):
            for seed in range(6):
                result = self.simulate(context, q, seed)
                self.assertEqual(self.check(result), [], (context, q, seed))

    def test_hang_variant_blocks_exactly_the_planted_waits(self):
        for context, q in ((1, 0), (16, 15), (33, 7)):
            for seed in range(4):
                result = self.simulate(context, q, seed, 'hang')
                self.assertEqual(model.check_hang(result), [], (context, q, seed))

    def test_unequal_round_counts_deadlock_instead_of_completing(self):
        for context, q in ((1, 3), (15, 0), (33, 15)):
            result = self.simulate(context, q, 1, 'wrong_c')
            self.assertFalse(result.terminated, (context, q))

    def test_the_checker_still_has_teeth_on_this_topology(self):
        for mutation in ('relay_before_ack', 'credit_before_reserve', 'no_reset'):
            caught = 0
            for seed in range(20):
                result = self.simulate(16, 7, seed, mutation=mutation)
                caught += bool(self.check(result))
            self.assertGreater(caught, 10, mutation)


if __name__ == '__main__':
    unittest.main()
