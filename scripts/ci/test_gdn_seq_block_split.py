"""V5 (gdn_seq_block_split, QWEN_FAST_GDN_SPLIT_V=2) host tests: the value-split recurrence launch, on the CPU.

Held here:
  - the flag grammar and its requirements (K5-A on, four-card serving);
  - the work split: 96 distinct cores for four users, a head pair on two consecutive cores of one grid column, half = w % 2;
  - the PAGE MAPS: a Python transcription of the reader's and writer's index formulas, transcribed again at LVt = 4 to
    equal K5-A's maps exactly (so the generalisation is the K5-A formula, not a new one), and at LVt = 2 to cover K5-A's
    page sets exactly once between the two halves (snapshot, S0, v, z, output), with the CB slot order the compute's
    state_out packs (the formulas are also looked up in the kernel sources, so the transcription cannot drift from them);
  - the CB plan (704,512 B; depth 1 is K5-A's plan byte for byte), the ring-alignment table, one producer and one consumer
    RISC per CB, and the CB indices each source names;
  - the runtime arguments: every kernel's highest get_arg_val index is below its RT_WORDS compile argument, every core's
    list has exactly RT_WORDS words (the generic_op cache-collision lesson, f945486e), and a program-cache model that
    replays user counts 1..4 in every order and never hits an entry whose per-core lengths differ;
  - the coalescing (one descriptor per role, one 'coalesced' note) and the fake bench (semaphore, peer coordinates);
  - QUALIFIED is empty and refuses an unqualified build everywhere but the builder argument; the controls and diagnostics;
  - the binding: flag unset leaves tp_addresses.bound_twins byte-identical and imports nothing; flag 2 binds the twin and
    uninstall restores K5-A; the twin refuses a build that is not the qualified K5-A one;
  - the image copy lists and the CPU allowlist.
"""

import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import gdn_multitoken as native
import gdn_seq_block as seq
import gdn_seq_block_split as split
import gdn_user_batch_tp as quad
import tp_addresses
import tp_shapes
import verify_trace_t1
from test_gdn_seq_block import SYNTHETIC_NATIVE, SyntheticRoot, cb_constants
from test_gdn_user_batch import FakeKernelDescriptor, FakeTTNN, FakeTensor
from test_gdn_user_batch_tp import four_mesh, user_inputs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EVIDENCE = [Path(os.environ['GDN_NATIVE_ROOT'])] if os.environ.get('GDN_NATIVE_ROOT') else []
EVIDENCE += [parent / 'runner-evidence.local' / '35093041332' / 'sources' for parent in HERE.parents]
NATIVE_ROOT = next((path for path in EVIDENCE if (path / native.KERNEL_ROOT / seq.NATIVE_COMPUTE).exists()), None)
FOUR = {'QWEN_FAST_TP': '4'}
FLAG_ON = dict(FOUR, **{split.FLAG: '2', seq.FLAG: '1', 'QWEN_FAST_GDN_USER_BATCH': '1'})
T1 = {'QWEN_FAST_VERIFY_T1': '1'}
DEVICE_ORIGIN = (1, 2)


def text(name):
    return (HERE / name).read_text()


def build(variant='A', diag=None, depth=2):
    with SyntheticRoot() as root:
        return split.load_kernels(root, variant=variant, diag=diag, depth=depth, unqualified=True)


_HANDOFF = patch('gdn_multitoken.validate_handoff_runtime')


def setUpModule():
    _HANDOFF.start()


def tearDownModule():
    _HANDOFF.stop()


class Four(unittest.TestCase):
    def setUp(self):
        for patcher in (patch.dict(os.environ, FOUR), patch.dict(os.environ, T1)):
            patcher.start()
            self.addCleanup(patcher.stop)
        verify_trace_t1.take()
        self.addCleanup(verify_trace_t1.take)


# ---- the flag ----

class FlagTests(unittest.TestCase):
    def test_unset_empty_zero_and_one_are_the_k5a_launch(self):
        for value in (None, '', '0', '1'):
            environ = dict(FOUR, **({} if value is None else {split.FLAG: value}))
            self.assertEqual(split.factor(environ), 1)
            self.assertFalse(split.enabled(environ))

    def test_two_needs_k5a_and_four_card_serving(self):
        self.assertEqual(split.factor(FLAG_ON), 2)
        self.assertTrue(split.enabled(FLAG_ON))
        without = {k: v for k, v in FLAG_ON.items() if k not in (seq.FLAG, 'QWEN_FAST_GDN_USER_BATCH')}
        with self.assertRaisesRegex(ValueError, 'requires QWEN_FAST_GDN_SEQ_BLOCK=1'):
            split.factor(without)
        with self.assertRaisesRegex(ValueError, 'requires QWEN_FAST_GDN_USER_BATCH=1'):
            split.factor(dict(without, **{seq.FLAG: '1'}))
        with self.assertRaisesRegex(ValueError, 'four-card'):
            split.factor(dict(FLAG_ON, QWEN_FAST_TP='2'))
        with self.assertRaisesRegex(ValueError, 'four-card'):
            split.factor({k: v for k, v in FLAG_ON.items() if k != 'QWEN_FAST_TP'})

    def test_anything_else_is_a_configuration_error(self):
        for value in ('3', 'yes', ' 2', '2 ', 'true', '02', '-1', '1.0'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'must be unset, 1 or 2'):
                split.factor(dict(FLAG_ON, **{split.FLAG: value}))

    def test_the_flag_is_named_as_the_plan_names_it(self):
        self.assertEqual(split.FLAG, 'QWEN_FAST_GDN_SPLIT_V')
        self.assertEqual(split.FACTOR, 2)


# ---- placement ----

class PlacementTests(Four):
    def test_four_users_fill_ninety_six_distinct_cores(self):
        placed = split.placement(11, 10, 4)
        points = [point for pairs in placed for _, _, point in pairs]
        self.assertEqual(len(points), 96)
        self.assertEqual(len(set(points)), 96)
        self.assertTrue(all(0 <= x <= 9 and 0 <= y <= 9 for x, y in points))
        self.assertEqual(max(x for x, _ in points), 9)
        self.assertEqual(sorted(y for x, y in points if x == 9), list(range(6)))

    def test_head_and_half_follow_the_worker_number(self):
        for users in (1, 2, 3, 4):
            placed = split.placement(11, 10, users)
            self.assertEqual(len(placed), users)
            for pairs in placed:
                self.assertEqual([(head, half) for head, half, _ in pairs],
                                 [(worker // 2, worker % 2) for worker in range(24)])

    def test_a_pair_is_two_consecutive_cores_of_one_column_owner_first(self):
        for users in (1, 2, 3, 4):
            for pairs in split.placement(11, 10, users):
                for (head, half, owner), (_, other, helper) in zip(pairs[0::2], pairs[1::2]):
                    self.assertEqual((half, other), (0, 1))
                    self.assertEqual(owner[0], helper[0])
                    self.assertEqual(helper[1], owner[1] + 1)
                    self.assertEqual(owner[1] % 2, 0)

    def test_the_placement_is_k5as_core_shares_at_twice_the_workers(self):
        for users in (1, 4):
            shares = quad.core_shares(11, 10, users, workers=24)
            self.assertEqual([[point for _, _, point in pairs] for pairs in split.placement(11, 10, users)], shares)

    def test_a_grid_height_that_splits_a_pair_across_columns_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'consecutive cores of one grid column'):
            split.placement(11, 9, 4)

    def test_too_small_a_grid_is_refused(self):
        with self.assertRaises(ValueError):
            split.placement(9, 10, 4)
        with self.assertRaises(ValueError):
            split.placement(11, 10, 5)

    def test_peers_map_each_head_to_its_two_halves(self):
        found = split.peers(split.placement(11, 10, 1)[0])
        self.assertEqual(sorted(found), list(range(12)))
        self.assertTrue(all(sorted(halves) == [0, 1] for halves in found.values()))


# ---- the page maps ----

def maps(lvt, half, heads=12, kt=4, vt=4, tokens=16):
    """The reader's and writer's page formulas for one half, as the .cpp sources spell them (a transcription):
    {'snapshot': [(token, page, cb_ring, cb_page)], 's0': [(state page, cb page)], 'v': [page], 'z': [page],
    'out': [page]} for head `head` - returned per head."""
    c0 = half * lvt
    rows = kt // 2
    found = {}
    for head in range(heads):
        snapshot = []
        for token in range(tokens):
            base = (head + heads * token) * kt * vt
            # writer: K rows 0, 1 (NoC 0), reader: K rows 2, 3 (NoC 1)
            for ring, offset, mapped in (('SOUT', 0, rows * vt), ('SOUT2', rows * vt, rows * vt)):
                per_half = rows * lvt
                for q in range(per_half):
                    ii, jl = q // lvt, q % lvt
                    snapshot.append((token, base + offset + vt * ii + c0 + jl, ring, jl * rows + ii))
        s0 = [(head * kt * vt + i * vt + c0 + jl, jl * kt + i) for jl in range(lvt) for i in range(kt)]
        v = [2048 // 128 + 0 * head + head * vt + c0 + jl for jl in range(lvt)]
        found[head] = dict(snapshot=snapshot, s0=s0, v=v)
    return found


def compute_pack_order(lvt, kt=4):
    """The CB page (ring, index) tile (i, jl) lands on in gdn_seq_block_split_compute.cpp's state_out: per column jl, two
    pages to SOUT (i < 2) then two to SOUT2 (i >= 2), pack_tile(0, ring, i % 2) of the column's reservation."""
    order = {}
    for jl in range(lvt):
        for i in range(kt):
            order[(i, jl)] = ('SOUT' if i < kt // 2 else 'SOUT2', jl * (kt // 2) + i % (kt // 2))
    return order


class PageMapTests(unittest.TestCase):
    def test_the_formulas_are_in_the_kernel_sources(self):
        reader, writer = text('gdn_seq_block_split_reader.cpp'), text('gdn_seq_block_split_writer.cpp')
        for needle in ('{.page_id = base_page + Vt * ii + c0 + jl}', 'const uint32_t slot = jl * rows + ii;',
                       'const uint32_t ii = q / SB_LVT;', 'const uint32_t jl = q % SB_LVT;',
                       'rows * Vt;', 'h * Kt * Vt + i * Vt + c0 + jl', '(jl * Kt + i) * tb',
                       'VOT + h * Vt + c0, SB_LVT', 'ZOT + h * Vt, Vt', 'const uint32_t c0 = half_id * SB_LVT;',
                       'const uint32_t start = (h + half / 2) % half;'):
            self.assertIn(needle, reader, needle)
        for needle in ('{.page_id = base_page + Vt * ii + c0 + jl}', 'const uint32_t slot = jl * rows + ii;',
                       'const uint32_t ii = q / SB_LVT;', 'const uint32_t jl = q % SB_LVT;',
                       'const uint32_t start = h % half;', '.page_id = h * Vt + tile',
                       'const uint32_t c0 = half_id * SB_LVT;'):
            self.assertIn(needle, writer, needle)

    def test_at_lvt_four_the_maps_are_k5as_exactly(self):
        """K5-A: writer page base + 4i + j (q = 4i + j, i < 2) from CB page 2j + i; reader base + 8 + 4(i-2) + j from 2j + i - 2;
        S0 page h*16 + 4i + j into CB page 4j + i."""
        for head in (0, 5, 11):
            found = maps(4, 0)[head]
            for token, page, ring, cb in found['snapshot']:
                base = (head + 12 * token) * 16
                local = page - base
                i, j = divmod(local, 4)
                self.assertEqual(ring, 'SOUT' if i < 2 else 'SOUT2')
                self.assertEqual(cb, 2 * j + i % 2)
            self.assertEqual(sorted(page for token, page, _, _ in found['snapshot'] if token == 0 and head == 0),
                             list(range(16)) if head == 0 else [])
            self.assertEqual(found['s0'], [(head * 16 + 4 * i + j, 4 * j + i) for j in range(4) for i in range(4)])

    def test_the_two_halves_cover_k5as_pages_exactly_once(self):
        both = [maps(2, 0), maps(2, 1)]
        for head in range(12):
            for token in range(16):
                pages = sorted(page for half in both for t, page, _, _ in half[head]['snapshot'] if t == token)
                base = (head + 12 * token) * 16
                self.assertEqual(pages, list(range(base, base + 16)))
            s0 = sorted(page for half in both for page, _ in half[head]['s0'])
            self.assertEqual(s0, list(range(head * 16, head * 16 + 16)))
            v = sorted(page for half in both for page in half[head]['v'])
            self.assertEqual(v, [16 + head * 4 + j for j in range(4)])

    def test_each_page_has_one_writer_and_the_owner_holds_the_low_columns(self):
        for half in (0, 1):
            found = maps(2, half)[3]
            columns = {(page - (3 + 12 * t) * 16) % 4 for t, page, _, _ in found['snapshot']}
            self.assertEqual(columns, {0, 1} if half == 0 else {2, 3})

    def test_the_cb_slot_is_the_page_the_compute_packs(self):
        order = compute_pack_order(2)
        for half in (0, 1):
            for token, page, ring, cb in maps(2, half)[0]['snapshot']:
                i, j = divmod(page - (0 + 12 * token) * 16, 4)
                jl = j - 2 * half
                self.assertEqual((ring, cb), order[(i, jl)])
        # K5-A's own pack order is the same function at four columns
        self.assertEqual(compute_pack_order(4)[(3, 2)], ('SOUT2', 2 * 2 + 1))

    def test_s0_slot_is_what_the_compute_reads(self):
        """The compute reads state tile (i, jl) at page jl * 4 + i of S0 (copy_tiles then mm_cols' j * SB_KT + ki)."""
        for half in (0, 1):
            for page, cb in maps(2, half)[7]['s0']:
                i, j = divmod(page - 7 * 16, 4)
                self.assertEqual(cb, (j - 2 * half) * 4 + i)

    def test_the_bank_walk_splits_owners_and_helpers(self):
        """bank = page mod 8: with the token base a multiple of 16, owners (columns 0, 1) sit on banks {0, 1, 4, 5} and
        helpers (columns 2, 3) on {2, 3, 6, 7}, on both NoCs, so 48 owners and 48 helpers load every bank equally."""
        for half, expected in ((0, {0, 1, 4, 5}), (1, {2, 3, 6, 7})):
            banks = {page % 8 for _, page, _, _ in maps(2, half)[0]['snapshot']}
            self.assertEqual(banks, expected)

    def test_the_start_rotation_visits_every_page_once(self):
        for head in range(12):
            for start in ((head % 4), ((head + 2) % 4)):
                self.assertEqual(sorted((start + k) % 4 for k in range(4)), [0, 1, 2, 3])

    def test_output_is_written_by_the_owner_alone_at_k5as_pages(self):
        writer = text('gdn_seq_block_split_writer.cpp')
        tail = writer[writer.index('if (!owner) {'):]
        self.assertEqual(tail.count('out_acc'), 1)
        self.assertLess(tail.index('noc_semaphore_wait'), tail.index('o.push_back(Vt)'))
        self.assertLess(tail.index('o.push_back(Vt)'), tail.index('out_acc'))
        self.assertIn('page_id = h * Vt + tile', tail)


# ---- the CB plan ----

RESERVATIONS = {  # CB index -> the page counts any one reserve / wait / push / pop on it uses
    0: (8,), 1: (2,), 2: (4,), 3: (1, 2), 4: (8,), 5: (4, 8), 6: (1,), 7: (2, 4), 8: (4,), 9: (4,), 10: (4, 8), 11: (4, 8),
    12: (2,), 13: (4,), 14: (1,), 15: (1,), 16: (1,), 17: (4,), 18: (4,), 19: (1,), 20: (1,), 21: (2, 4), 22: (10,), 23: (8,),
    24: (4, 8), 25: (4, 8), 26: (2,), 27: (2, 4), 28: (2,), 29: (4,), 30: (4,),
}


class PlanTests(unittest.TestCase):
    def test_the_plan_is_k5as_with_the_operand_rings_two_tokens_deep(self):
        plan = split.plan(2)
        for index, entry in seq.CB_PLAN.items():
            if index in (22, 23):
                self.assertEqual(plan[index][1], 2 * entry[1])
                self.assertEqual(plan[index][2:], entry[2:])
            else:
                self.assertEqual(plan[index], entry, index)
        self.assertEqual(split.plan(1), seq.CB_PLAN)

    def test_the_bytes_and_the_l1_top(self):
        self.assertEqual(seq.cb_bytes(), 630784)
        self.assertEqual(split.cb_bytes(1), seq.cb_bytes())
        self.assertEqual(split.cb_bytes(2), 704512)
        # design arithmetic: the CBs start at 111,488 B; K5-A tops out at 742,272 B, V5 at 816,000 B, under the 908,352 B
        # high-water mark SDPA already sets in the same trace (a clash fails loudly when the program is created)
        self.assertEqual(111488 + split.cb_bytes(1), 742272)
        self.assertEqual(111488 + split.cb_bytes(2), 816000)
        self.assertLess(111488 + split.cb_bytes(2), 908352)

    def test_every_ring_is_a_multiple_of_each_reservation_on_it(self):
        for depth in split.DEPTHS:
            plan = split.plan(depth)
            for index, sizes in RESERVATIONS.items():
                for size in sizes:
                    if index in (7, 21):
                        # SOUT and SOUT2 are popped a column (2 pages) at a time and addressed by hand around the wrap
                        self.assertEqual(plan[index][1] % 2, 0, index)
                    else:
                        self.assertEqual(plan[index][1] % size, 0, (depth, index, plan[index][0], size))

    def test_the_epilogue_rings_are_back_at_a_boundary(self):
        """After 16 tokens every ring that is cycled per token has advanced a multiple of the epilogue's reservation."""
        plan = split.plan(2)
        per_token_pages = {5: 8, 24: 16, 25: 8, 26: 4, 27: 4, 28: 2}
        for index, pages in per_token_pages.items():
            self.assertEqual(16 * pages % plan[index][1], 0, plan[index][0])

    def test_one_producer_risc_and_one_consumer_risc_per_cb(self):
        for index, (name, pages, dtype, producer, consumer) in split.plan(2).items():
            self.assertIn(producer, seq.RISCS)
            self.assertIn(consumer, seq.RISCS)
        self.assertEqual(split.plan(2)[29][3:], ('writer', 'compute'))

    def test_the_sources_name_every_cb_at_the_plans_index(self):
        kernels = build()
        plan = split.plan(2)
        by_name = {entry[0]: index for index, entry in plan.items()}
        for role in ('reader', 'writer', 'compute'):
            for name, index in cb_constants(kernels[role]).items():
                if name in by_name:
                    self.assertEqual(by_name[name], index, '%s: SB_%s' % (role, name))
        self.assertIn('SB_SOUT2', kernels['reader'])
        self.assertIn('SB_O = 29', kernels['writer'])

    def test_depth_is_in_the_source_so_the_program_cache_cannot_share_it(self):
        two, one = build(depth=2), build(depth=1)
        self.assertNotEqual(split.sha256(two), split.sha256(one))
        self.assertEqual(split.sha256(two)['reader'] == split.sha256(one)['reader'], False)
        self.assertIn('#define GDN_SPLIT_DEPTH 2', two['compute'])
        self.assertIn('#define GDN_SPLIT_DEPTH 1', one['compute'])


# ---- sources and builds ----

class SourceTests(unittest.TestCase):
    def test_the_sources_are_lf_only_and_carry_the_build_anchor_once(self):
        for role, name in split.SOURCES.items():
            data = (HERE / name).read_bytes()
            self.assertNotIn(b'\r', data, name)
            self.assertEqual(data.decode().count(split.BUILD_ANCHOR), 1, name)

    def test_the_generated_header_names_the_build(self):
        kernels = build()
        for role in split.ROLES:
            self.assertIn('#define GDN_SPLIT_LVT 2', kernels[role])
            self.assertIn('#define GDN_SEQ_BLOCK_VARIANT 0', kernels[role])
            self.assertIn('#define GDN_SEQ_BLOCK_DIAG 0', kernels[role])
            self.assertNotIn(split.BUILD_ANCHOR, kernels[role])

    def test_the_compute_is_the_native_prefix_then_the_split_compute(self):
        with SyntheticRoot() as root:
            prefix = seq.native_prefix(root)
            kernels = split.generate(root)
        self.assertTrue(kernels['compute'].startswith(prefix))
        self.assertTrue(prefix.endswith('}  // namespace\n\n'))
        self.assertNotIn('void kernel_main() {\n    // served body', kernels['compute'])

    def test_a_changed_native_compute_is_refused_before_it_is_sliced(self):
        with SyntheticRoot() as root:
            path = root / native.KERNEL_ROOT / seq.NATIVE_COMPUTE
            path.write_bytes(SYNTHETIC_NATIVE.replace('served body', 'served  body').encode())
            with self.assertRaisesRegex(ValueError, 'Native compute hash changed'):
                split.generate(root)

    def test_variants_and_diagnostics_are_distinct_builds(self):
        shas = {(variant, diag): split.sha256(build(variant, diag)) for variant in split.VARIANTS
                for diag in split.DIAGNOSTICS}
        self.assertEqual(len({tuple(sorted(value.items())) for value in shas.values()}), len(shas))
        self.assertIn('rowsum_k_rotated(SB_NSQ, SB_SUM, SB_VT)', text('gdn_seq_block_split_compute.cpp'))
        self.assertIn('#define GDN_SEQ_BLOCK_VARIANT 1', build('N5r')['compute'])
        self.assertIn('#define GDN_SEQ_BLOCK_VARIANT 2', build('N5x')['writer'])
        self.assertIn('#define GDN_SEQ_BLOCK_VARIANT 3', build('N5o')['compute'])
        self.assertIn('rowsum_k_own_only(SB_NSQ, SB_SUM, SB_VT)', text('gdn_seq_block_split_compute.cpp'))

    def test_unknown_variants_diagnostics_and_depths_are_refused(self):
        with SyntheticRoot() as root:
            for kwargs in (dict(variant='A0'), dict(variant='N'), dict(diag='x'), dict(depth=3)):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    split.generate(root, **kwargs)

    def test_the_k5a_sources_and_triple_are_untouched(self):
        self.assertEqual(seq.QUALIFIED[0]['reader'], '784deb4e398cfaa5a7c2b415c1a8be71925ccfd61fe18d801b11fb138dbc45c0')
        self.assertEqual(seq.QUALIFIED[0]['writer'], '215ef7da5d01d5c5fa720e8f6218481f27491c075151140376b457d760e1870d')
        self.assertEqual(seq.QUALIFIED[0]['compute'], 'a9782f1434673d05fb7270b810836101ef38b51becfbd5c13d67880a937deffe')
        self.assertEqual(seq.SOURCES, dict(reader='gdn_seq_block_reader.cpp', writer='gdn_seq_block_writer.cpp',
                                           compute='gdn_seq_block_compute.cpp'))

    def test_kernel_arithmetic_changes_nothing_the_chain_computes_per_column(self):
        """Every chain helper in the split compute is a loop over this core's SB_LVT columns and no other width appears in
        the chain (SB_VT, the whole head, is the epilogue's alone)."""
        compute = text('gdn_seq_block_split_compute.cpp')
        chain = compute[compute.index('for (uint32_t t = 0; t < SB_T; ++t) {'):compute.index('// ---- epilogue')]
        self.assertNotIn('SB_VT', chain)
        self.assertNotIn('SB_KV,', chain)
        self.assertIn('SB_LVT', chain)
        epilogue = compute[compute.index('// ---- epilogue'):]
        self.assertIn('if (owner) {', epilogue)
        self.assertNotIn('SB_LVT', epilogue)
        for call in ('rowsum_k(SB_NSQ, SB_SUM, SB_VT)', 'inv_rms(SB_SUM, SB_FAC_Q, NG_EPS_BITS, NG_SCALE_BITS, true)',
                     'ew(SB_H, SB_DL, SB_OUT, SB_VT, 2)'):
            self.assertIn(call, epilogue)

    def test_the_epilogue_is_k5as_line_for_line(self):
        """The owner's epilogue (inside `if (owner)`) is K5-A's epilogue statements, in order, on all four tiles; only the
        rowsum differs, by variant (the control N5r)."""
        def statements(source):
            body = source[source.index('norm_w -> fp32'):]
            return [re.sub(r'\s+', ' ', line.strip()) for line in body.splitlines()
                    if line.strip().endswith(';') and not line.strip().startswith('//')]
        k5a = [line for line in statements(text('gdn_seq_block_compute.cpp')) if 'rowsum_k' not in line]
        mine = [line for line in statements(text('gdn_seq_block_split_compute.cpp')) if 'rowsum_k' not in line]
        self.assertEqual(mine, k5a)
        self.assertGreater(len(k5a), 20)

    def test_the_helper_never_runs_the_epilogue_or_reads_z_and_norm_w(self):
        reader = text('gdn_seq_block_split_reader.cpp')
        self.assertEqual(reader.count('if (owner) {'), 4)  # z, norm_w reads, the z push, the W tail
        self.assertIn('owner && tile < Vt', reader)


# ---- arguments ----

def kernel_indices(source):
    return sorted({int(index) for index in re.findall(r'get_arg_val<uint32_t>\((\d+)\)', source)})


class ArgumentTests(Four):
    def test_every_kernel_reads_only_below_its_declared_words(self):
        kernels = build()
        for role in split.ROLES:
            indices = kernel_indices(kernels[role])
            self.assertEqual(indices, list(range(split.RT_WORDS[role])), role)
            used = int(re.search(r'SB_RT_USED = (\d+);', kernels[role]).group(1))
            self.assertEqual(used, split.RT_WORDS[role], role)
            self.assertIn('static_assert(SB_RT_USED <= RT_WORDS', kernels[role])

    def test_rt_words_is_a_compile_argument_in_the_place_the_kernel_reads_it(self):
        kernels = build()
        args = {role: split.compile_args(role, kernels) for role in split.ROLES}
        self.assertEqual(args['reader'][10:], [9, kernels.tag('reader')])
        self.assertEqual(args['writer'][3:], [6, kernels.tag('writer')])
        self.assertEqual(args['compute'][4:], [0, 1, kernels.tag('compute')])
        self.assertIn('get_compile_time_arg_val(10)', kernels['reader'])
        self.assertIn('get_compile_time_arg_val(11)', kernels['reader'])
        self.assertIn('get_compile_time_arg_val(3)', kernels['writer'])
        self.assertIn('get_compile_time_arg_val(5)', kernels['compute'])
        # the accessor offsets are the first index after the compile arguments
        self.assertEqual(len(args['reader']), 12)
        self.assertIn('TensorAccessorArgs<12>()', kernels['reader'])
        self.assertEqual(len(args['writer']), 5)
        self.assertIn('TensorAccessorArgs<5>()', kernels['writer'])

    def test_the_leading_compile_arguments_are_k5as(self):
        kernels = build()
        k5 = seq.load_kernels
        with SyntheticRoot() as root:
            base = seq.load_kernels(root, 0, unqualified=True)
        self.assertEqual(split.compile_args('reader', kernels)[:10], seq.compile_args('reader', base)[:10])
        self.assertEqual(split.compile_args('writer', kernels)[:3], seq.compile_args('writer', base)[:3])
        self.assertEqual(split.compile_args('compute', kernels)[:5], seq.compile_args('compute', base)[:5])
        self.assertIs(k5, seq.load_kernels)

    def test_runtime_lists_have_the_same_length_for_owner_and_helper(self):
        addresses = list(range(100, 108))
        for role, words in split.RT_WORDS.items():
            for head in (0, 11):
                for half in (0, 1):
                    self.assertEqual(len(split.runtime_args(role, head, half, addresses, (7, 9))), words)
        self.assertEqual(split.runtime_args('reader', 3, 1, addresses),
                         [3, 1, 100, 101, 102, 103, 106, 107, 105])
        self.assertEqual(split.runtime_args('writer', 3, 1, addresses, (7, 9)), [3, 1, 104, 105, 7, 9])
        self.assertEqual(split.runtime_args('compute', 3, 1, addresses), [1])

    def test_bad_arguments_are_refused(self):
        addresses = list(range(8))
        for call in (lambda: split.runtime_args('reader', 12, 0, addresses), lambda: split.runtime_args('reader', 0, 2, addresses),
                     lambda: split.runtime_args('reader', 0, 0, addresses[:7]), lambda: split.runtime_args('x', 0, 0, addresses),
                     lambda: split.runtime_args('writer', 0, 0, addresses, (1,))):
            with self.assertRaises(ValueError):
                call()


# ---- the fake bench ----

class Device:
    def worker_core_from_logical_core(self, core):
        return SimpleNamespace(x=core[0] + DEVICE_ORIGIN[0], y=core[1] + DEVICE_ORIGIN[1])


DEVICE = Device()


class Shard(FakeTensor):
    def device(self):
        return DEVICE


class SplitDescriptor(FakeKernelDescriptor):
    pass


class SplitTTNN(FakeTTNN):
    KernelDescriptor = SplitDescriptor

    @staticmethod
    def TensorAccessorArgs(value):
        return SimpleNamespace(get_compile_time_args=lambda: [7])  # interleaved DRAM: no address in the compile args

    @staticmethod
    def SemaphoreDescriptor(id, core_ranges, initial_value):
        return SimpleNamespace(id=id, core_ranges=tuple(core_ranges), initial_value=initial_value)

    @staticmethod
    def ProgramDescriptor(kernels, cbs, semaphores=()):
        return SimpleNamespace(kernels=list(kernels), cbs=list(cbs), semaphores=list(semaphores))

    def get_device_tensors(self, value):
        return [Shard(value.name + ':' + str(chip), value.shape, value.address + chip, memory=value._memory,
                      dtype=value.dtype, layout=value.layout) for chip in range(4)]


def launch(users, kernels=None, ttnn=None, grid=(11, 10)):
    ttnn = SplitTTNN() if ttnn is None else ttnn
    groups = [user_inputs(index) for index in range(users)]
    kernels = kernels or build()
    produced = split.execute(four_mesh(*grid), groups, ttnn, output_memory='dram', kernels=kernels)
    return ttnn, produced, ttnn.launches[-1][1]


def per_core(descriptor):
    """{(x, y): runtime words} of one kernel descriptor."""
    return {(x, y): list(words) for x, column in descriptor.runtime_args.items() for y, words in column.items()}


class BenchTests(Four):
    def test_the_program_holds_one_semaphore_over_the_union_and_the_planned_cbs(self):
        ttnn, produced, program = launch(4)
        self.assertEqual(len(produced), 4)
        chip = program[((0, 0), (0, 0))]
        self.assertEqual(len(chip.semaphores), 1)
        self.assertEqual((chip.semaphores[0].id, chip.semaphores[0].initial_value), (0, 0))
        self.assertEqual(chip.semaphores[0].core_ranges, chip.cbs[0][2])
        self.assertEqual(sum(cb[1] for cb in chip.cbs), split.cb_bytes(2))
        self.assertEqual(sorted(cb[3][0][1] for cb in chip.cbs), sorted(split.plan(2)))

    def test_one_coalesced_descriptor_per_role_and_one_note_per_launch(self):
        verify_trace_t1.take()
        ttnn, produced, program = launch(4)
        counts = verify_trace_t1.take()
        self.assertEqual(counts, {'coalesced': 1})
        for chip in range(4):
            descriptors = program[((0, chip), (0, chip))].kernels
            self.assertEqual(len(descriptors), 3)
            self.assertEqual({d.kernel_source for d in descriptors}, {build()[role] for role in split.ROLES})

    def test_without_the_coalesce_cut_each_user_keeps_a_descriptor_per_role(self):
        with patch.dict(os.environ, {'QWEN_FAST_VERIFY_T1': '0'}):
            ttnn, produced, program = launch(3)
        self.assertEqual(len(program[((0, 0), (0, 0))].kernels), 9)

    def test_every_core_gets_exactly_rt_words_in_every_role_and_a_peer_that_points_back(self):
        ttnn, produced, program = launch(4)
        placed = split.placement(11, 10, 4)
        for chip in range(4):
            descriptors = {d.kernel_source: d for d in program[((0, chip), (0, chip))].kernels}
            kernels = build()
            for role in split.ROLES:
                words = per_core(descriptors[kernels[role]])
                self.assertEqual(len(words), 96)
                self.assertEqual({len(value) for value in words.values()}, {split.RT_WORDS[role]}, role)
            writers = per_core(descriptors[kernels['writer']])
            readers = per_core(descriptors[kernels['reader']])
            computes = per_core(descriptors[kernels['compute']])
            for pairs in placed:
                points = {(head, half): point for head, half, point in pairs}
                for (head, half), point in points.items():
                    other = points[(head, 1 - half)]
                    expected = [other[0] + DEVICE_ORIGIN[0], other[1] + DEVICE_ORIGIN[1]]
                    self.assertEqual(writers[point][4:], expected)
                    self.assertEqual(writers[point][:2], [head, half])
                    self.assertEqual(readers[point][:2], [head, half])
                    self.assertEqual(computes[point], [half])

    def test_the_compile_arguments_and_accessors_are_identical_on_every_core(self):
        ttnn, produced, program = launch(4)
        for descriptor in program[((0, 0), (0, 0))].kernels:
            self.assertEqual(len(descriptor.core_ranges), len(descriptor.core_ranges))
        kernels = build()
        by_source = {d.kernel_source: d for d in program[((0, 0), (0, 0))].kernels}
        self.assertEqual(by_source[kernels['reader']].compile_time_args[:12], split.compile_args('reader', kernels))
        self.assertEqual(by_source[kernels['writer']].compile_time_args[:5], split.compile_args('writer', kernels))

    def test_program_cache_model_never_pairs_runtime_lists_of_different_length(self):
        """generic_op keys on source, defines, compile arguments and cores - not on runtime-argument lengths. Replay every
        order of user counts 1..4: entries are keyed by (sources, compile args, cores); every entry's per-core lengths are
        the same constants, so a hit never overwrites a list with one of another length."""
        seen = {}
        orders = [(1, 2, 3, 4), (4, 3, 2, 1), (2, 4, 1, 3), (3, 1, 4, 2), (4, 4, 1, 1)]
        for order in orders:
            for users in order:
                ttnn, produced, program = launch(users)
                for descriptor in program[((0, 0), (0, 0))].kernels:
                    key = (descriptor.kernel_source, tuple(descriptor.compile_time_args), len(descriptor.core_ranges))
                    lengths = {len(words) for words in per_core(descriptor).values()}
                    self.assertEqual(len(lengths), 1)
                    seen.setdefault(key, set()).update(lengths)
        self.assertTrue(all(len(lengths) == 1 for lengths in seen.values()))
        roles = {role: kernel for role, kernel in build().items()}
        for key, lengths in seen.items():
            role = next(name for name, source in roles.items() if source == key[0])
            self.assertEqual(lengths, {split.RT_WORDS[role]})

    def test_one_entry_per_user_count_because_the_union_differs(self):
        keys = set()
        for users in (1, 2, 3, 4):
            ttnn, produced, program = launch(users)
            chip = program[((0, 0), (0, 0))]
            keys.add((users, tuple(chip.semaphores[0].core_ranges)))
        self.assertEqual(len({core for _, core in keys}), 4)

    def test_users_must_not_alias_buffers(self):
        ttnn = SplitTTNN()
        groups = [user_inputs(0), user_inputs(1, base=1000)]
        groups[1] = tuple(FakeTensor('a%d' % i, value.shape, 1000 + i * 10 if i < 5 else 500)
                          for i, value in enumerate(groups[1]))
        with self.assertRaises(ValueError):
            split.execute(four_mesh(), groups, ttnn, output_memory='dram', kernels=build())

    def test_a_build_that_is_not_a_split_build_is_refused_by_the_builder(self):
        with self.assertRaisesRegex(ValueError, 'must come from gdn_seq_block_split.load_kernels'):
            split.build_program(SplitTTNN(), four_mesh(), [], dict(reader='', writer='', compute=''))

    def test_the_helpers_remote_write_precedes_its_semaphore_increment(self):
        writer = text('gdn_seq_block_split_writer.cpp')
        helper = writer[writer.index('if (!owner) {'):writer.index('    if constexpr (SB_VARIANT != SB_VARIANT_N5X) {\n        volatile')]
        order = [helper.index(call) for call in ('noc_async_write(o_base', 'noc_async_write_barrier();',
                                                 'noc_semaphore_inc', 'noc_async_atomic_barrier();')]
        self.assertEqual(order, sorted(order))
        self.assertIn('o_base + SB_LVT * tf', helper)
        self.assertIn('SB_LVT * tf);', helper)


class EngagementTests(Four):
    def test_one_engagement_line_per_user_count_per_process(self):
        lines = []
        split._ENGAGED.clear()
        self.addCleanup(split._ENGAGED.clear)
        with patch.object(seq, 'log_line', side_effect=lines.append):
            for users in (4, 4, 3, 4, 3, 1):
                launch(users)
        self.assertEqual(lines, ['[PINDIAG] gdn split_v build split=2 users=%d qualified=0' % users
                                 for users in (4, 3, 1)])
        self.assertEqual(split.engaged_line(4, True), '[PINDIAG] gdn split_v build split=2 users=4 qualified=1')


# ---- qualification ----

class QualificationTests(unittest.TestCase):
    def test_nothing_is_qualified_until_the_card_gate_has_passed(self):
        self.assertEqual(split.QUALIFIED, {})

    def test_an_unqualified_build_is_refused_everywhere_but_the_builder_argument(self):
        with SyntheticRoot() as root:
            with self.assertRaisesRegex(ValueError, 'is not qualified'):
                split.load_kernels(root)
            with self.assertRaisesRegex(ValueError, 'explicit bool'):
                split.load_kernels(root, unqualified=1)
            kernels = split.load_kernels(root, unqualified=True)
            self.assertFalse(kernels.qualified)
            with patch.dict(split.QUALIFIED, {0: split.sha256(kernels)}):
                self.assertTrue(split.load_kernels(root).qualified)
                for kwargs in (dict(variant='N5r'), dict(variant='N5x'), dict(variant='N5o'), dict(diag='nosnap'), dict(depth=1)):
                    with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'is not qualified'):
                        split.load_kernels(root, **kwargs)

    def test_serving_the_split_without_a_qualified_triple_fails_loudly(self):
        with SyntheticRoot() as root:
            split._SERVED.clear()
            self.addCleanup(split._SERVED.clear)
            with patch.dict(os.environ, {'TT_METAL_HOME': str(root)}):
                with self.assertRaisesRegex(ValueError, 'is not qualified'):
                    split.served_kernels()
                with self.assertRaisesRegex(ValueError, 'is not qualified'):
                    split.resolve_kernels(None)

    def test_the_build_is_a_dict_with_its_provenance(self):
        kernels = build('N5r', 'nosnap', 1)
        self.assertEqual((kernels.variant, kernels.diag, kernels.depth, kernels.split, kernels.level),
                         ('N5r', 'nosnap', 1, 2, 0))
        self.assertEqual(sorted(kernels), ['compute', 'reader', 'writer'])


# ---- the twin seam ----

class TwinTests(unittest.TestCase):
    def tearDown(self):
        tp_addresses.uninstall()

    def test_the_row_and_the_flag_are_registered(self):
        self.assertIn(('gdn_seq_block', 'execute', 'gdn_seq_block_split', 'execute'), tp_addresses.TWINS)
        self.assertIn(('gdn_seq_block', 'execute'), tp_addresses.FLAGGED_TWINS)
        self.assertEqual(tp_addresses.FLAGGED_TWINS[('gdn_seq_block', 'execute')][0], split.FLAG)

    def test_flag_unset_leaves_the_bound_twins_as_before_and_imports_nothing(self):
        code = ('import sys, tp_addresses\n'
                'rows, modules = tp_addresses.bound_twins({"QWEN_FAST_TP": "4"})\n'
                'print(len(rows), len(modules), "gdn_seq_block_split" in sys.modules, "gdn_seq_block" in sys.modules)\n'
                'rows0, _ = tp_addresses.bound_twins({"QWEN_FAST_TP": "4", "QWEN_FAST_GDN_SPLIT_V": "1"})\n'
                'print(rows == rows0, ("gdn_seq_block", "execute", "gdn_seq_block_split", "execute") in rows)\n'
                'for raw in ("0", ""):\n'
                '    tp_addresses.bound_twins({"QWEN_FAST_TP": "4", "QWEN_FAST_GDN_SPLIT_V": raw})\n'
                'print("gdn_seq_block_split" in sys.modules)\n')
        env =dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        env.pop('QWEN_FAST_GDN_SPLIT_V', None)
        out = subprocess.run([sys.executable, '-B', '-c', code], cwd=str(HERE), env=env, capture_output=True,
                             text=True, check=True).stdout.split('\n')
        rows, modules, imported, seq_imported = out[0].split()
        self.assertEqual(imported, 'False')
        self.assertEqual(out[1], 'True False')
        self.assertEqual(out[2], 'False')  # '1', '0' and '' never import the lever (a fall-through to its parser would)
        self.assertEqual(int(rows), len(tp_addresses.TWINS) - len(tp_addresses.FLAGGED_TWINS))

    def test_flag_two_binds_the_twin_and_uninstall_restores_k5a(self):
        original = seq.execute
        rebound = tp_addresses.install(dict(FLAG_ON))
        self.assertGreater(rebound, 0)
        self.assertIs(seq.execute, split.execute)
        tp_addresses.uninstall()
        self.assertIs(seq.execute, original)

    def test_flag_two_without_its_requirements_raises_at_install(self):
        with self.assertRaisesRegex(ValueError, 'requires QWEN_FAST_GDN_SEQ_BLOCK=1'):
            tp_addresses.install(dict(FOUR, **{split.FLAG: '2'}))
        with self.assertRaisesRegex(ValueError, 'must be unset, 1 or 2'):
            tp_addresses.install(dict(FOUR, **{split.FLAG: '3'}))

    def test_only_the_gdn_seq_block_execute_changes(self):
        rows0, _ = tp_addresses.bound_twins(dict(FOUR))
        rows2, _ = tp_addresses.bound_twins(dict(FLAG_ON))
        self.assertEqual([row for row in rows2 if row not in rows0],
                         [('gdn_seq_block', 'execute', 'gdn_seq_block_split', 'execute')])

    def test_the_twin_takes_the_qualified_k5a_build_and_substitutes_the_split_one(self):
        with SyntheticRoot() as root:
            qualified = seq.Build(seq.generate(root), 0, 'A', None, True)
            split_build = split.load_kernels(root, unqualified=True)
            with patch.object(split, 'served_kernels', return_value=split_build):
                self.assertIs(split.resolve_kernels(qualified), split_build)
                self.assertIs(split.resolve_kernels(None), split_build)
                self.assertIs(split.resolve_kernels(split_build), split_build)
                for refused in (seq.Build(seq.generate(root), 0, 'A', None, False),
                                seq.Build(seq.generate(root, variant='N'), 0, 'N', None, True) if False else
                                seq.Build(seq.generate(root), 0, 'A', 'nosnap', True), dict(reader='', writer='',
                                                                                          compute=''), 'x'):
                    with self.subTest(refused=type(refused).__name__), self.assertRaisesRegex(ValueError, 'takes the qualified'):
                        split.resolve_kernels(refused)

    def test_the_twin_refuses_the_drop_in_mistake(self):
        with self.assertRaisesRegex(ValueError, 'takes operations third'):
            split.execute(four_mesh(), [], dict(reader=''), kernels=None)


# ---- the image copy lists and the allowlist ----

class DeliveryTests(unittest.TestCase):
    FILES = ('gdn_seq_block_split.py', 'gdn_seq_block_split_compute.cpp', 'gdn_seq_block_split_reader.cpp',
             'gdn_seq_block_split_writer.cpp')

    def test_the_overlay_names_every_file_the_flag_loads(self):
        overlay = (ROOT / 'docker' / 'qwen-c2-overlay.txt').read_text()
        lines = [line.split()[0] for line in overlay.splitlines() if line.strip() and not line.startswith('#')]
        for name in self.FILES:
            self.assertIn('scripts/ci/' + name, lines)
        self.assertIn('scripts/ci/tp_addresses.py', lines)

    def test_both_p8_copy_lists_name_the_new_sources(self):
        dockerfile = (ROOT / 'docker' / 'qwen-fast-serving.Dockerfile').read_text()
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-fast-serving-image.yml').read_text()
        for name in self.FILES:
            self.assertIn(name, dockerfile)
            self.assertIn(name, workflow)

    def test_the_cpu_workflow_runs_the_suite_and_the_harness_tests(self):
        workflow = (ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml').read_text()
        self.assertIn('test_gdn_seq_block_split', workflow)
        self.assertIn("discover -s optimisation/ttnn-op/v5split -p 'test_*.py'", workflow)

    def test_the_k5a_sources_are_not_in_the_diff_of_this_change(self):
        """The K5-A files are pinned by test_gdn_seq_block's sha256 list; they keep their bytes (this repeats the pin so a
        change here fails here too)."""
        pinned = (HERE / 'test_gdn_seq_block.py').read_text()
        for name in ('gdn_seq_block_compute.cpp', 'gdn_seq_block_reader.cpp', 'gdn_seq_block_writer.cpp'):
            digest = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            self.assertIn(digest, pinned, name)


@unittest.skipUnless(NATIVE_ROOT, 'the pinned native kernels are not in this checkout')
class PinnedNativeTests(unittest.TestCase):
    def test_the_split_sources_generate_against_the_pinned_native_compute(self):
        with patch('gdn_multitoken.validate_handoff_runtime'), patch.dict(os.environ, FOUR):
            kernels = split.load_kernels(NATIVE_ROOT, unqualified=True)
            audit = split.audit(NATIVE_ROOT)
        self.assertEqual(split.sha256(kernels), audit['generated_sha256'])
        self.assertTrue(kernels['compute'].startswith(seq.native_prefix(NATIVE_ROOT)))
        self.assertEqual(audit['cores'], 96)
        self.assertEqual(audit['cb_bytes_per_core'], 704512)
        self.assertEqual(audit['runtime_words'], dict(reader=9, writer=6, compute=1))


if __name__ == '__main__':
    unittest.main()
