"""QWEN_FAST_OCTO_DRAFT (octo_draft_tp.py): ONE eight-seat, eight-row, 64-row draft pass for the octo rounds, in place of the two four-seat quad passes. Default off, gate only.

What is held here, on the CPU (nothing runs on a card; the card jobs OD1/OD2 of scripts/ci/references/tp4-octo2-jobs are the hardware half):

  - the flag is strict (0, 1, unset), the admission names every reason and is part of the octo admission (the six [OCTO] UNQUALIFIED lines stay six);
  - the geometry: the 24-piece K/V plan gives every user a 2080-key segment with its 8 live keys at 2048..2055 (the single-user T8 layout), the mask is the single-user T8
    mask, the RoPE tables are the four-user halves' packed tables at eight rows;
  - the fold: folded head (8 * group) * h + group * u + j is head h * group + j with user u's rows first, the GQA group sends it to KV head 8h + u, and every user's rows
    of the folded SDPA are its single-user T8 SDPA bit for bit - on two CPU stand-ins (a per-head float32 SDPA and a key-chunk-ordered online-softmax SDPA, whose firing
    controls show that a changed chunk order, a mis-mapped segment or a missing rotation changes the bits), in three regimes and with a hundredfold partner;
  - the branches at eight users: the attention and MLP branches run the quad's op sequence at 64 rows (per_core_M = 2), the SDPA at 64 query / 16 KV heads over 2080 keys,
    the conv with eight seams;
  - the readback: every user's features, candidates and scores are its own eight rows' selection, merged per 32-row half and stitched;
  - the trace on eight-user, four-chip fakes (placeholders or the pool's live banks, host inputs per round, collect/adopt/finish, close, a wider ticket refused);
  - the coordinator: eight steady users run one octo pass and no quad or pair; six and seven live, a refusal, a short DRAM headroom and a failure leave the quads; two failures
    in a row block it; the pooled mask and output shapes carry slots 0-7 with the flag and only with it;
  - OFF IS TODAY: the pair process never loads the module, the coordinator and the hook make exactly today's calls without the flag, execute_proposal's guard and seams are the
    quad's for any pass without `users` / `block`;
  - shipping: both copy lists, the overlay, the CPU suite.

    python3 -B -m unittest test_octo_draft      (from scripts/ci, over a python with torch)
"""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import draft_singles_audit  # noqa: E402
import octo_draft_tp as octo  # noqa: E402
import pair_row_exact  # noqa: E402
import quad_draft as pinned  # noqa: E402
import quad_draft_tp as quad  # noqa: E402
import tp_shapes  # noqa: E402
import test_quad_draft as base  # noqa: E402
from test_pair_row_exact import Device, TorchOps, coded, keep, same_bits  # noqa: E402
from test_dflash_proposal_trace import FakeShard, FakeTensor, RecordingOps  # noqa: E402
from test_quad_draft_tp4 import (COLLECTIVES, ChunkedOps, FakeTensor4, HostOps4, RecordingOps4, ShapeOps4, RecordingQuad,  # noqa: E402
                                 clean_environment, fresh_banks)
from tp_test_support import four_cards, pair  # noqa: E402

CPU_WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml'
FLAGS = {'QWEN_FAST_QUAD_DRAFT': '1', 'QWEN_FAST_PACKED_PROPOSAL': '1', 'QWEN_FAST_PIPELINED_PROPOSALS': '1', 'QWEN_FAST_PAIR_ROW_EXACT': '1',
         'QWEN_FAST_ROUND_B1': '1', 'QWEN_FAST_TP': '4', 'QWEN_FAST_OCTO': 'alternate', 'QWEN_FAST_OCTO_DRAFT': '1'}
MESH4 = SimpleNamespace(shape=[1, 4], compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))


def t8_mask():
    """The single-user T8 mask: what every folded head reads."""
    return octo.octo_host_mask()


# ---------------------------------------------------------------------------------------------
# The flag and the geometry.
# ---------------------------------------------------------------------------------------------

class FlagTests(unittest.TestCase):
    def test_zero_or_one_only_and_off_by_default(self):
        self.assertFalse(octo.enabled({}))
        self.assertFalse(octo.enabled({octo.FLAG: '0'}))
        self.assertTrue(octo.enabled({octo.FLAG: '1'}))
        for value in ('', 'yes', 'true', '2', ' 1', 'on'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                octo.enabled({octo.FLAG: value})
        with clean_environment():
            self.assertFalse(octo.enabled())

    def test_the_conv_mode_is_the_quads_three_and_defaults_to_110(self):
        self.assertEqual(octo.conv_mode({}), '110')
        for value in ('110', '80', 'halves'):
            self.assertEqual(octo.conv_mode({octo.CONV_FLAG: value}), value)
        with self.assertRaises(ValueError):
            octo.conv_mode({octo.CONV_FLAG: '64'})

    def test_the_geometry_is_eight_users_of_eight_rows_in_a_64_row_block(self):
        self.assertEqual((octo.USERS, octo.BLOCK, octo.ROWS, octo.PROPOSALS), (8, 8, 64, 7))
        self.assertEqual(octo.USERS * octo.BLOCK, octo.ROWS)
        self.assertEqual(octo.HALF_USERS * octo.BLOCK, 32, 'four users fill one 32-row tile half')
        self.assertEqual(octo.CONTEXT + octo.BLOCK + octo.PAD, octo.SPAN)
        self.assertEqual(octo.PREPARED_BLOCK, 16, 'the prepared drafter layers are the T16 ones')
        self.assertEqual(octo.PAIRS, ((0, 1), (2, 3), (4, 5), (6, 7)))
        self.assertEqual(octo.SLOTS, tuple(range(8)))

    def test_the_heads_at_four_cards_are_64_over_16(self):
        with four_cards():
            self.assertEqual(octo.heads(), (8, 2, 4, 64, 16))
        with pair():
            self.assertEqual(octo.heads(), (16, 4, 4, 128, 32))


class KeyValueLayoutTests(unittest.TestCase):
    def test_the_plan_is_24_pieces_covering_8_segments_of_2080_keys(self):
        plan = octo.octo_key_value_plan()
        self.assertEqual(len(plan), 24)
        self.assertEqual(sum(piece['rows'] for piece in plan), 8 * 2080)
        for user in range(8):
            cached, live, pad = plan[3 * user:3 * user + 3]
            self.assertEqual((cached['kind'], cached['user'], cached['rows']), ('cached', user, 2048))
            self.assertEqual((live['kind'], live['rows'], live['source']), ('live', 8, slice(8 * user, 8 * user + 8)))
            half = user // 4
            self.assertEqual((pad['kind'], pad['rows'], pad['source']), ('pad', 24, slice(32 * half, 32 * half + 24)))
            self.assertLessEqual(pad['source'].stop, 64)

    def test_key_value_plan_refuses_other_geometries(self):
        plan, spans, keys = octo.key_value_plan([2048] * 8, 16)
        self.assertEqual(keys, 8 * 2080)
        self.assertEqual([(span['rows'].start, span['rows'].stop) for span in spans], [(8 * user, 8 * user + 8) for user in range(8)])
        self.assertEqual([span['keys'] for span in spans], [slice(2080 * user, 2080 * user + 2080) for user in range(8)])
        for contexts, rows in (([2048] * 4, 16), ([2048] * 8, 8), ([2048] * 7 + [1024], 16), ([2048] * 8, True)):
            with self.subTest(contexts=len(contexts), rows=rows), self.assertRaises(ValueError):
                octo.key_value_plan(contexts, rows)

    def test_every_users_segment_is_the_single_user_t8_layout(self):
        fixture = Fixture8(2, 2, 3)
        for user in range(8):
            keys = fixture.octo_keys['k'][:, :, 2080 * user:2080 * (user + 1)]
            self.assertTrue(torch.equal(keys[:, :, :2048], fixture.cache[user]['k']))
            self.assertTrue(torch.equal(keys[:, :, 2048:2056], fixture.live[user]['k']), 'the 8 live keys follow the history')
            self.assertTrue(torch.isfinite(keys[:, :, 2056:].float()).all(), 'the pad is masked, and finite')

    def test_the_host_mask_is_the_single_user_t8_mask(self):
        mask = t8_mask()
        self.assertEqual(tuple(mask.shape), (1, 1, 32, 2080))
        visible = mask[0, 0] == 0
        self.assertTrue(visible[:8, 2048:2056].all(), 'every live row sees all 8 live keys (the block is bidirectional)')
        for row in range(8):
            # the sliding window of 2048 positions: the row at position 2048 + row sees the history keys after row
            self.assertEqual(int(visible[row, :2048].sum()), 2047 - row)
        self.assertFalse(visible[:8, 2056:].any(), 'the 24 pad keys are masked for the live rows')
        self.assertTrue((visible[8:].sum(dim=-1) == 1).all(), 'every other row sees exactly one key')
        self.assertTrue(visible[8:, 2048].all())
        from dflash_t16_native_attention import validate_mask

        validate_mask(mask)

    def test_the_rope_tables_are_each_halfs_packed_tables_at_eight_rows(self):
        from dflash_batched_mask import live_key_rope_from, packed_rope_tables

        users = [dict(position=4096 + 1000 * user + 7, history_rows=2048) for user in range(8)]
        query, live = octo.octo_rope(users)
        for tables in (query, live):
            self.assertEqual([tuple(table.shape) for table in tables], [(1, 1, 64, 128)] * 2)
        for half in range(2):
            members = users[4 * half:4 * half + 4]
            expected = packed_rope_tables(members, 8)
            rows = slice(32 * half, 32 * half + 32)
            for table, theirs in zip(query, expected['q']):
                self.assertTrue(same_bits(table[:, :, rows], theirs))
            for table, theirs in zip(live, live_key_rope_from(expected['k'], members, 8)):
                self.assertTrue(same_bits(table[:, :, rows], theirs))
        # user u's query row r is at position_u + r: the table of a user at another position differs from the one at its own
        from draft_head_preparation import rope_tables

        own = rope_tables(users[5]['position'], 8)
        for table, theirs in zip(query, own):
            self.assertTrue(same_bits(table[:, :, 40:48], theirs))
        with self.assertRaises(ValueError):
            octo.octo_rope(users[:4])


# ---------------------------------------------------------------------------------------------
# The fold.
# ---------------------------------------------------------------------------------------------

class Fixture8:
    """Eight users' operands at `heads` query and `kv` KV heads per chip, built two ways: as the octo pass assembles them (one 64-row live block, the 24-piece plan) and as
    each user's single-user T8 trace does (its own 8 live rows, 24 pad keys and 24 pad query rows)."""

    def __init__(self, heads, kv, seed=0, *, scale=1.0):
        generator = torch.Generator().manual_seed(seed)

        def normal(*shape, factor=1.0):
            return (torch.randn(*shape, generator=generator) * factor).bfloat16()

        self.heads, self.kv = heads, kv
        self.cache = [{name: normal(1, kv, 2048, 128) for name in 'kv'} for _ in range(8)]
        self.live = [{name: normal(1, kv, 8, 128) for name in 'kv'} for _ in range(8)]
        self.live_pad = [{name: normal(1, kv, 24, 128) for name in 'kv'} for _ in range(8)]
        self.query = [normal(1, heads, 8, 128, factor=scale) for _ in range(8)]
        self.query_pad = [normal(1, heads, 24, 128, factor=scale) for _ in range(8)]
        self.rebuild()

    def rebuild(self):
        self.block = {name: torch.cat([self.live[user][name] for user in range(8)], dim=2) for name in 'kv'}
        self.octo_query = torch.cat(self.query, dim=2)
        self.octo_keys = self.assemble(octo.octo_key_value_plan(), self.block, self.cache)

    @staticmethod
    def assemble(plan, block, caches):
        """The branch's assembly loop on the host."""
        out = {}
        for name in 'kv':
            pieces = []
            for part in plan:
                if part['kind'] == 'cached':
                    pieces.append(caches[part['user']][name])
                    continue
                start = part['source'].start if 'source' in part else 0
                pieces.append(block[name][:, :, start:start + part['rows']])
            out[name] = torch.cat(pieces, dim=2)
        return out

    def single(self, user):
        """The single-user T8 trace's operands: query (1, heads, 32, 128) with its 24 pad rows, keys and values (1, kv, 2080, 128)."""
        keys = {name: torch.cat([self.cache[user][name], self.live[user][name], self.live_pad[user][name]], dim=2) for name in 'kv'}
        return torch.cat([self.query[user], self.query_pad[user]], dim=2), keys['k'], keys['v']


def single_rows(operations, fixture, user, mask):
    query, keys, values = fixture.single(user)
    return operations.sdpa(Device(query), Device(keys), Device(values), attn_mask=Device(mask), is_causal=False, scale=128 ** -0.5,
                           program_config=None, compute_kernel_config=None, memory_config='dram').value[:, :, :8]


class FoldMappingTests(unittest.TestCase):
    def test_folded_query_head_is_head_h_group_j_with_user_u_first(self):
        with four_cards():
            operations, owned = TorchOps(), []
            query = coded(8, 64)
            folded = octo.octo_fold_query(operations, Device(query), keep(owned)).value
            self.assertEqual(tuple(folded.shape), (1, 64, 32, 128))
            for folded_head, (h, user, j, kv_head) in octo.octo_head_map().items():
                half, k = divmod(user, 4)
                source = query[0, 4 * h + j, 32 * half:32 * half + 32]
                expected = torch.roll(source, -8 * k, dims=0)
                self.assertTrue(torch.equal(folded[0, folded_head], expected), 'head %d (h=%d u=%d j=%d)' % (folded_head, h, user, j))
                self.assertTrue(torch.equal(folded[0, folded_head, :8], query[0, 4 * h + j, 8 * user:8 * user + 8]), 'user %d rows come first' % user)
                self.assertEqual(kv_head, 8 * h + user)

    def test_the_gqa_group_sends_a_folded_head_to_its_users_kv_head(self):
        with four_cards():
            query_heads, key_heads, group, folded, kv = octo.heads()
            for folded_head, (h, user, j, kv_head) in octo.octo_head_map().items():
                self.assertEqual(folded_head // group, kv_head, 'the SDPA maps query head i to KV head i // group')
            self.assertEqual(len(octo.octo_head_map()), folded)
            self.assertEqual(folded, 8 * query_heads)
            self.assertEqual(kv, 8 * key_heads)

    def test_the_folded_kv_head_is_the_users_segment_of_head_h(self):
        with four_cards():
            fixture = Fixture8(8, 2, 1)
            operations, owned = TorchOps(), []
            keys = octo.octo_fold_keys(operations, Device(fixture.octo_keys['k']), keep(owned)).value
            self.assertEqual(tuple(keys.shape), (1, 16, 2080, 128))
            for h in range(2):
                for user in range(8):
                    expected = fixture.octo_keys['k'][0, h, 2080 * user:2080 * (user + 1)]
                    self.assertTrue(torch.equal(keys[0, 8 * h + user], expected))
                    self.assertTrue(torch.equal(keys[0, 8 * h + user, :2048], fixture.cache[user]['k'][0, h]))

    def test_the_unfold_returns_every_users_rows_to_its_block_rows(self):
        with four_cards():
            operations, owned = TorchOps(), []
            query = coded(8, 64)
            folded = Device(octo.octo_fold_query(operations, Device(query), keep(owned)).value)
            out = octo.octo_unfold_output(operations, folded, keep(owned)).value
            self.assertEqual(tuple(out.shape), (1, 8, 64, 128))
            self.assertTrue(torch.equal(out, query), 'fold then unfold of an identity attention is the packed query')

    def test_the_operands_are_validated_against_the_width(self):
        with four_cards():
            fixture = Fixture8(8, 2, 2)
            args = (Device(fixture.octo_query), Device(fixture.octo_keys['k']), Device(fixture.octo_keys['v']), Device(t8_mask()), keep([]))
            operations = TorchOps()
            octo.fold_attention(operations, *args, mask_validated=True)
            with self.assertRaises(ValueError):
                octo.fold_attention(operations, *args)
            for index, bad in ((0, Device(fixture.octo_query[:, :, :32])), (1, Device(fixture.octo_keys['k'][:, :, :8320])),
                               (3, Device(t8_mask()[:, :, :, :2048])), (3, Device(t8_mask()[:, :, :16]))):
                changed = list(args)
                changed[index] = bad
                with self.subTest(index=index), self.assertRaises(ValueError):
                    octo.fold_attention(operations, *changed, mask_validated=True)
            wrong = Device(fixture.octo_query)
            wrong.dtype = 'fp32'
            with self.assertRaises(ValueError):
                octo.fold_attention(operations, wrong, *args[1:4], keep([]), mask_validated=True)

    def test_the_op_count_per_layer(self):
        """Documented, not tuned: 2 halves x (1 half slice + 4 quarter slices + 3 rotation concats + 1 group concat) + 1 half concat for the query; 8 slices and 3 concats for the output."""
        with four_cards():
            fixture = Fixture8(8, 2, 3)
            operations = TorchOps()
            octo.fold_attention(operations, Device(fixture.octo_query), Device(fixture.octo_keys['k']), Device(fixture.octo_keys['v']), Device(t8_mask()),
                                keep([]), mask_validated=True)
        kinds = [call[0] for call in operations.calls]
        self.assertEqual(kinds.count('slice'), 2 * 5 + 8)
        self.assertEqual(kinds.count('concat'), 2 * 4 + 1 + 2 + 1)
        self.assertEqual(kinds.count('sdpa'), 1)
        self.assertEqual([call for call in operations.calls if call[0] == 'sdpa'],
                         [('sdpa', (1, 64, 32, 128), (1, 16, 2080, 128), (1, 1, 32, 2080))], '64 query heads over 16 KV heads of 2080 keys, the one mask')


class FoldEqualsSingleTests(unittest.TestCase):
    """Every octo work unit is the user's single-user T8 unit: at 8 / 2 heads, bit for bit."""

    OPS = (TorchOps, ChunkedOps)

    def octo_rows(self, ops, fixture):
        return octo.fold_attention(ops, Device(fixture.octo_query), Device(fixture.octo_keys['k']), Device(fixture.octo_keys['v']), Device(t8_mask()),
                                   keep([]), mask_validated=True).value

    def test_every_users_rows_equal_the_single_user_t8_trace_bit_for_bit(self):
        for factory in self.OPS:
            for seed, scale in ((0, 1.0), (1, 4.0), (2, 0.25)):
                with self.subTest(ops=factory.__name__, seed=seed, scale=scale), four_cards():
                    fixture = Fixture8(8, 2, seed, scale=scale)
                    rows = self.octo_rows(factory(), fixture)
                    self.assertEqual(tuple(rows.shape), (1, 8, 64, 128))
                    for user in range(8):
                        mine = rows[:, :, 8 * user:8 * user + 8]
                        alone = single_rows(factory(), fixture, user, t8_mask())
                        self.assertTrue(same_bits(mine, alone), 'user %d against drafting alone' % user)

    def test_a_hundredfold_partner_changes_nothing(self):
        with four_cards():
            fixture, loud = Fixture8(8, 2, 5), Fixture8(8, 2, 5)
            for user in (0, 2, 5):
                for name in 'kv':
                    for part in ('cache', 'live'):
                        getattr(loud, part)[user][name] = (getattr(loud, part)[user][name].float() * 100).bfloat16()
            loud.rebuild()
            for factory in self.OPS:
                quiet, louder = self.octo_rows(factory(), fixture), self.octo_rows(factory(), loud)
                for user in (1, 3, 4, 6, 7):
                    rows = slice(8 * user, 8 * user + 8)
                    self.assertTrue(same_bits(louder[:, :, rows], quiet[:, :, rows]), '%s user %d' % (factory.__name__, user))
                self.assertTrue(same_bits(louder[:, :, 8:16], single_rows(factory(), fixture, 1, t8_mask())))

    def test_the_pad_keys_and_pad_query_rows_are_irrelevant(self):
        with four_cards():
            fixture, other = Fixture8(8, 2, 8), Fixture8(8, 2, 8)
            generator = torch.Generator().manual_seed(99)
            for user in range(8):
                for name in 'kv':
                    other.live_pad[user][name] = (torch.randn(1, 2, 24, 128, generator=generator) * 3).bfloat16()
                other.query_pad[user] = (torch.randn(1, 8, 24, 128, generator=generator) * 3).bfloat16()
            for factory in self.OPS:
                for user in range(8):
                    self.assertTrue(same_bits(single_rows(factory(), fixture, user, t8_mask()), single_rows(factory(), other, user, t8_mask())))

    def test_firing_controls_the_emulation_notices_a_changed_chunk_order_a_mis_mapped_segment_and_a_missing_rotation(self):
        """The equalities above mean something only if the stand-in can tell a wrong computation from a right one."""
        with four_cards():
            fixture = Fixture8(8, 2, 6, scale=4.0)
            good = self.octo_rows(ChunkedOps(), fixture)
            wide = ChunkedOps()
            wide.chunk = 64
            self.assertFalse(same_bits(self.octo_rows(wide, fixture), good), 'a different key-chunk size changes the bits')
            operations, owned = ChunkedOps(), []
            folded = octo.octo_fold_query(operations, Device(fixture.octo_query), keep(owned))
            keys = octo.octo_fold_keys(operations, Device(fixture.octo_keys['k']), keep(owned)).value
            values = octo.octo_fold_keys(operations, Device(fixture.octo_keys['v']), keep(owned)).value

            def unfolded(folded_query, keys_, values_):
                attention = operations.sdpa(folded_query, Device(keys_), Device(values_), attn_mask=Device(t8_mask()), is_causal=False, scale=128 ** -0.5,
                                            program_config=None, compute_kernel_config=None, memory_config=None)
                return octo.octo_unfold_output(operations, attention, keep(owned)).value

            alone = single_rows(ChunkedOps(), fixture, 3, t8_mask())
            self.assertTrue(same_bits(unfolded(folded, keys, values)[:, :, 24:32], alone), 'the right fold is the single')
            shifted_keys = torch.roll(keys.reshape(1, 2, 8, 2080, 128), 1, dims=2).reshape(1, 16, 2080, 128)
            shifted_values = torch.roll(values.reshape(1, 2, 8, 2080, 128), 1, dims=2).reshape(1, 16, 2080, 128)
            self.assertFalse(same_bits(unfolded(folded, shifted_keys, shifted_values)[:, :, 24:32], alone), 'a mis-mapped segment is caught')
            # a fold that does not rotate: every user keeps the half's own row order, so users 1-3 of a half put their rows at 8k, not at 0
            plain = Device(self.unrotated(fixture))
            self.assertFalse(same_bits(unfolded(plain, keys, values)[:, :, 24:32], alone), 'a missing rotation is caught')

    @staticmethod
    def unrotated(fixture):
        heads, group = 8, 4
        half = [fixture.octo_query[:, :, 32 * p:32 * p + 32].reshape(2, group, 32, 128) for p in range(2)]
        parts = [half[p] for p in range(2) for _ in range(4)]
        return torch.cat(parts, dim=1).reshape(1, 64, 32, 128)


# ---------------------------------------------------------------------------------------------
# The branches at eight users.
# ---------------------------------------------------------------------------------------------

class RecordingOcto(octo.OctoPass):
    """The octo pass with the conv recorded (the E1b program is the quad's and has its own tests)."""

    def __init__(self, ops, **options):
        super().__init__(**options)
        self.ops = ops

    def convolution(self, operations, mesh, hidden, dynamic, base_kernels, **options):
        return base.recording_convolution(self.ops)(operations, mesh, hidden, dynamic, base_kernels, **options)


def octo_attention_run(*, chips=4):
    """execute_attention_branch at four cards on a shape fake, over the octo pass. Returns the event log."""
    import draft_attention_branch

    found = tp_shapes.active()
    ops = ShapeOps4(chips)
    rows = 64
    rope = {'q': (base.Tensor((1, 1, rows, 128)), base.Tensor((1, 1, rows, 128))), 'live_k': (base.Tensor((1, 1, rows, 128)), base.Tensor((1, 1, rows, 128)))}
    caches = [{name: base.Tensor((1, found.draft_kv_heads, 2048, 128)) for name in 'kv'} for _ in range(8)]
    pack = [dict(position=4096 + 1000 * user, history_rows=2048) for user in range(8)]
    mask = base.Tensor((1, 1, 32, 2080))
    mesh = SimpleNamespace(shape=[1, chips])
    parameters = dict(operations=ops, mesh=mesh, block_rows=16, native_head_layout=True, native_proposal_attention=True, kernel='kernel',
                      norm=base.Tensor((1, 1, 160, 32)), convolution=base.Tensor((5120, 1280)), bases=[base.Tensor((1, 1, 1, 5120)) for _ in range(4)],
                      projections=dict(q=base.Tensor((5120, found.draft_query)), k=base.Tensor((5120, found.draft_kv_heads * 128)),
                                       v=base.Tensor((5120, found.draft_kv_heads * 128))),
                      head_norms=dict(q=base.Tensor((1, 1, 4, 32)), k=base.Tensor((1, 1, 4, 32))), output_projection=base.Tensor((found.draft_query, 5120)))
    quad_pass = RecordingOcto(ops)
    owned = []
    with patch('dflash_t16_native_scope.require_active', return_value=None), patch('mesh_link_policy.projection_links', return_value=1), \
            patch('feature_collective_tp.projection_links', return_value=1), patch('feature_collective.projection_links', return_value=1):
        output = draft_attention_branch.execute_attention_branch(ops, mesh, COLLECTIVES, base.Tensor((1, 1, rows, 5120)), None, mask, rope,
            lambda value: owned.append(value) or value, parameters=parameters, context=None, pack=pack, convolution_operation=quad_pass.convolution,
            cached_history=caches, native_proposal_mask_validated=True, quad=quad_pass)
    ops.log('output', output.shape)
    return ops.events


def octo_mlp_run(*, chips=4):
    import draft_mlp_branch

    ops = ShapeOps4(chips)
    weights, convolution = object(), object()
    mlp = tp_shapes.geometry(chips).mlp
    parameters = dict(operations=ops, mesh=SimpleNamespace(shape=[1, chips]), source_weights=weights, source_convolution=convolution, kernel='kernel',
                      device_norm=base.Tensor((1, 1, 160, 32)), device_conv=base.Tensor((5120, 1280)), bases=[base.Tensor((1, 1, 1, 5120)) for _ in range(4)],
                      device_projections=[base.Tensor((5120, mlp)), base.Tensor((5120, mlp)), base.Tensor((mlp, 5120))], shards='shards',
                      norm_weight='norm', conv_weight='conv', base_weight='base')
    seams = tuple((start, start + 8) for start in range(0, 64, 8))
    quad_pass = RecordingOcto(ops)
    owned = []
    with patch('mesh_link_policy.projection_links', return_value=1), patch('feature_collective_tp.projection_links', return_value=1), \
            patch('feature_collective.projection_links', return_value=1):
        output = draft_mlp_branch.execute_mlp_branch(ops, parameters['mesh'], COLLECTIVES, base.Tensor((1, 1, 64, 5120)), weights, convolution,
            lambda value: owned.append(value) or value, parameters=parameters, trace_safe=True, convolution_operation=quad_pass.convolution,
            boundaries=seams, quad=quad_pass)
    ops.log('output', output['output'].shape)
    return ops.events


class BranchTests(unittest.TestCase):
    def test_the_attention_branch_is_the_quads_op_sequence_at_64_rows_with_eight_users(self):
        from test_quad_draft_tp4 import attention_run4

        with four_cards():
            quad_events = attention_run4(quad=True)
            events = octo_attention_run()
        names = lambda log: [event[0] for event in log if event[0] in base.CORE_OPS]
        self.assertEqual(names(events), names(quad_events), 'the same core ops (matmuls, norms, rotary, heads, collectives) in the same order')
        programs = [dict(event[1]) for event in events if event[0] == 'program']
        self.assertEqual([item['per_core_M'] for item in programs], [2] * 4)
        self.assertEqual([event for event in events if event[0] == 'create_heads'], [('create_heads', (1, 1, 64, 1024), (1, 1, 64, 512), 8, 2)])
        self.assertEqual([event for event in events if event[0] == 'all_gather'], [('all_gather', (1, 1, 64, 5120), 0)])
        self.assertEqual([event for event in events if event[0] == 'sdpa'], [('sdpa', (1, 64, 32, 128), (1, 16, 2080, 128), (1, 1, 32, 2080))])
        self.assertEqual([event[3] for event in events if event[0] == 'convolve'], [tuple((8 * user, 8 * user + 8) for user in range(8))] * 2)
        self.assertEqual(events[-1], ('output', (1, 1, 64, 5120)))

    def test_the_kv_assembly_is_24_pieces_of_two_heads(self):
        with four_cards():
            events = octo_attention_run()
        assembly = [event for event in events if event[0] == 'concat' and len(event[1]) == 24]
        self.assertEqual(len(assembly), 2, 'one concat each for k and v')
        self.assertEqual(assembly[0][1], ((1, 2, 2048, 128), (1, 2, 8, 128), (1, 2, 24, 128)) * 8)
        live = [event for event in events if event[0] == 'slice' and event[1] == (1, 2, 64, 128)]
        self.assertEqual([event[2][2] for event in live], [0, 0, 8, 0, 16, 0, 24, 0, 32, 32, 40, 32, 48, 32, 56, 32] * 2)

    def test_the_mlp_branch_is_the_quads_at_64_rows_and_eight_seams(self):
        from test_quad_draft_tp4 import mlp_run4

        with four_cards():
            quad_events, events = mlp_run4(quad=True), octo_mlp_run()
        self.assertEqual([event[0] for event in events], [event[0] for event in quad_events], 'op for op')
        self.assertEqual([event[3] for event in events if event[0] == 'convolve'], [tuple((8 * user, 8 * user + 8) for user in range(8))] * 2)
        self.assertEqual([dict(event[1])['per_core_M'] for event in events if event[0] == 'program'], [2] * 4)

    def test_the_conv_seams_are_a_bit_per_user_start(self):
        seams = tuple((8 * user, 8 * user + 8) for user in range(8))
        low, high = pinned.seam_words(seams, 64)
        self.assertEqual((low, high), (0x01010101, 0x01010101), 'seams at rows 0, 8, 16, 24 of each tile row')

    def test_the_pass_is_the_quads_with_this_modules_plan_and_fold(self):
        quad_pass, octo_pass = quad.QuadPass(), octo.OctoPass()
        self.assertIsInstance(octo_pass, quad.QuadPass)
        self.assertEqual((octo_pass.rows, octo_pass.mask_rows, octo_pass.users, octo_pass.block), (64, 2080, 8, 8))
        for name in ('project_key_value', 'concatenate_query_heads', 'head_candidates', 'gather_add_projection'):
            self.assertIs(getattr(type(octo_pass), name), getattr(type(quad_pass), name), name)
        self.assertIs(octo.OctoPass.key_value_plan, octo.key_value_plan)
        self.assertFalse(hasattr(quad_pass, 'users'), 'the quad pass carries neither: execute_proposal keeps its four users of the device\'s rows')
        self.assertFalse(hasattr(quad_pass, 'block'))


class ExecuteProposalGuardTests(unittest.TestCase):
    """execute_proposal's two places that counted four users of 16 rows: the guard and the MLP seams. A pass without `users` / `block` is exactly today's."""

    def source(self):
        return (HERE / 'dflash_device.py').read_text(encoding='utf-8')

    def test_the_guard_and_the_seams_read_the_pass_and_default_to_today(self):
        text = self.source()
        self.assertIn("len(pack) != getattr(quad, 'users', 4)", text)
        self.assertIn("seam_rows = getattr(quad, 'block', self.block_rows) if quad is not None else self.block_rows", text)
        self.assertNotIn('len(pack) != 4 or row_exact', text)

    def run_proposal(self, quad_pass, users, block_rows=16):
        """Drive DFlashDevice.execute_proposal's guard and seam computation on a stub far enough to reach the MLP branch call."""
        import dflash_device

        seen = {}

        def fake_attention(*args, **kwargs):
            return 'hidden'

        def fake_mlp(*args, **kwargs):
            seen['boundaries'] = kwargs.get('boundaries')
            raise StopIteration

        operations = SimpleNamespace(experimental=SimpleNamespace(all_gather_async=lambda *a, **k: 'gathered'), reshape=lambda value, shape: value,
                                     pad=lambda value, *a, **k: value, DRAM_MEMORY_CONFIG='dram')
        device = SimpleNamespace(operations=operations, live_query_qk=False, native_proposal_attention=True, validated_native_proposal_masks={1},
                                 kv_history=SimpleNamespace(), layers=[('attention', 'mlp', 'weights', 'convolution')] * 5, block_rows=block_rows,
                                 fused_convolution=True, model=SimpleNamespace(embd=lambda *a, **k: 'embedded'), collectives=SimpleNamespace(
                                     get_and_cycle_ag_semaphore_handles=lambda: 1, get_and_cycle_barrier_semaphore_handle=lambda: 1),
                                 progress=None, proposal_calls=0, mesh='mesh', position=0)
        pack = [dict(position=4096, history_rows=2048)] * users
        with patch.object(dflash_device, 'addresses', return_value=1), patch.object(dflash_device, 'execute_attention_branch', fake_attention), \
                patch.object(dflash_device, 'execute_mlp_branch', fake_mlp), patch.object(dflash_device, 'projection_links', return_value=1), \
                patch.object(dflash_device, 'fast_ccl_topology', return_value=None), patch.object(dflash_device, 'tp_shapes', tp_shapes):
            try:
                dflash_device.DFlashDevice.execute_proposal(device, 'ids', None, 'mask', {}, context=None, owned=[], retain=lambda value: value,
                                                            stage=lambda *a, **k: None, audit=False, cached_history=[[1] * 5] * users, pack=pack, quad=quad_pass)
            except StopIteration:
                pass
        return seen

    def test_a_pass_without_users_still_demands_four(self):
        with four_cards():
            seen = self.run_proposal(quad.QuadPass(), 4)
            self.assertEqual(seen['boundaries'], ((0, 16), (16, 32), (32, 48), (48, 64)))
            with self.assertRaises(ValueError):
                self.run_proposal(quad.QuadPass(), 8)

    def test_the_octo_pass_takes_eight_users_and_eight_row_seams(self):
        with four_cards():
            seen = self.run_proposal(octo.OctoPass(), 8)
            self.assertEqual(seen['boundaries'], tuple((8 * user, 8 * user + 8) for user in range(8)))
            with self.assertRaises(ValueError):
                self.run_proposal(octo.OctoPass(), 4)


# ---------------------------------------------------------------------------------------------
# The readback.
# ---------------------------------------------------------------------------------------------

def eight_user_outputs(generator, chips=4):
    """The pass's 64-row raw outputs: per candidate chunk and chip (1, 1, 64, 16) values and indices, and the replicated (1, 1, 64, 256) features."""
    from draft_shared_head_tp import candidate_chunks

    chunks = []
    for start, stop in candidate_chunks():
        values, indices = [], []
        for _chip in range(chips):
            values.append(torch.randn(1, 1, 64, 16, generator=generator).bfloat16())
            rows = [torch.randperm(stop - start, generator=generator)[:16] for _ in range(64)]
            indices.append(torch.stack(rows).reshape(1, 1, 64, 16).to(torch.int32))
        chunks.append(dict(start=start, stop=stop, values=SimpleNamespace(chips=values), indices=SimpleNamespace(chips=indices)))
    projected = torch.randn(1, 1, 64, 256, generator=generator).bfloat16()
    return SimpleNamespace(chunks=chunks, projected=SimpleNamespace(chips=[projected.clone() for _ in range(chips)]))


class ReadbackTests(unittest.TestCase):
    def test_every_users_parts_are_its_own_eight_rows_selection(self):
        generator = torch.Generator().manual_seed(31)
        with four_cards():
            outputs = eight_user_outputs(generator)
            device = SimpleNamespace(operations=HostOps4())
            parts = octo.read_octo_outputs(device, outputs)
            self.assertEqual(len(parts), 8)
            raw = draft_singles_audit.raw_outputs(device.operations, outputs)
            for user in range(8):
                expected = draft_singles_audit.read_parts(draft_singles_audit.user_rows(raw, 8 * user, 8), block_rows=8)
                for key in ('hidden', 'candidates', 'unary'):
                    with self.subTest(user=user, key=key):
                        mine = parts[user][key]
                        self.assertEqual(tuple(mine.shape), (1, 7, 256) if key == 'hidden' else (1, 7, 16))
                        self.assertTrue(draft_singles_audit.same_bits(mine, expected[key]))

    def test_the_readback_reads_four_chips_by_two_chunks(self):
        generator = torch.Generator().manual_seed(32)
        with four_cards():
            outputs = eight_user_outputs(generator)
            operations = HostOps4()
            reads = Mock(wraps=operations.to_torch)
            operations.to_torch = reads
            octo.read_octo_outputs(SimpleNamespace(operations=operations), outputs)
        self.assertEqual(reads.call_count, 2 * 4 * 2 + 4, 'the quad\'s 18 reads: the head chunks and the features from every chip')

    def test_replicated_features_that_differ_on_any_chip_are_refused(self):
        generator = torch.Generator().manual_seed(33)
        with four_cards():
            outputs = eight_user_outputs(generator)
            outputs.projected.chips[2][0, 0, 41, 3] += 1
            with self.assertRaises(AssertionError):
                octo.read_octo_outputs(SimpleNamespace(operations=HostOps4()), outputs)

    def test_a_readback_with_the_pairs_two_chips_is_refused_at_four_cards(self):
        generator = torch.Generator().manual_seed(34)
        with four_cards():
            outputs = eight_user_outputs(generator, chips=2)
            with self.assertRaises(AssertionError):
                octo.read_octo_outputs(SimpleNamespace(operations=HostOps4()), outputs)

    def test_select_octo_outputs_selects_each_users_own_slice(self):
        with four_cards():
            parts = [dict(user=user) for user in range(8)]
            with patch.object(octo, 'read_octo_outputs', return_value=parts) as read, \
                    patch('dflash_packed_proposal.select_packed', return_value='tokens') as select:
                device = SimpleNamespace(predecessors='p', successors='s')
                self.assertEqual(octo.select_octo_outputs(device, 'outputs', tuple(range(8)), (7,) * 8), 'tokens')
            self.assertTrue(read.call_args.kwargs['reference'], 'the unbatched selection is today\'s reference path')
            select.assert_called_once_with(parts, tuple(range(8)), (7,) * 8, 'p', 's')


# ---------------------------------------------------------------------------------------------
# The trace on eight-user, four-chip fakes.
# ---------------------------------------------------------------------------------------------

def octo_devices8(ops):
    """Eight devices on one mesh, pooled slots 0-7, each with five layers of active K/V banks that are device tensors."""
    from test_dflash_proposal_trace import pair_devices

    made = []
    for _ in range(4):
        made.extend(pair_devices(ops, context=2048))
    mesh = made[0].mesh
    for index, device in enumerate(made):
        device.mesh = mesh
        device.pool_slot = SimpleNamespace(index=index)
        device.position = 4096 + 1000 * index + 7
        device.kv_history.active = fresh_banks(ops)
    return made


class TraceTests(unittest.TestCase):
    def setUp(self):
        import dflash_proposal_trace

        environment = clean_environment(**FLAGS)
        environment.start()
        self.addCleanup(environment.stop)
        cards = four_cards()
        cards.__enter__()
        self.addCleanup(cards.__exit__, None, None, None)
        for target, value in ((octo, '_NOTED'), (dflash_proposal_trace, '_PAIR_MASK_REFRESH_NOTED')):
            patcher = patch.object(target, value, [])
            patcher.start()
            self.addCleanup(patcher.stop)

    def build(self, **options):
        ops = RecordingOps4()
        devices = octo_devices8(ops)
        return ops, devices, octo.PreparedOctoDFlashProposal(devices, **options)

    def test_the_bucket_owns_eight_users_placeholder_banks_and_never_reads_the_pools(self):
        ops, devices, trace = self.build()
        lines, loguru = base.logged()
        with loguru, patch('memory_ledger.record') as ledger:
            self.assertTrue(trace.prepare_device(tuple(range(11, 19))))
        uploads = [event for event in ops.events if event[0] == 'from_torch' and event[2] is True]
        shapes = [event[1][1] for event in uploads]
        self.assertEqual(shapes[:6], [(1, 64), (1, 1, 32, 2080), (1, 1, 64, 128), (1, 1, 64, 128), (1, 1, 64, 128), (1, 1, 64, 128)])
        self.assertEqual(shapes[6:], [(1, 2, 2048, 128)] * 80, '8 users x 5 layers x k and v, two KV heads')
        bucket = trace.buckets[(2048,) * 8]
        self.assertEqual(set(bucket.rope), {'q', 'live_k'})
        self.assertTrue(same_bits(bucket.host_mask, t8_mask()))
        held = [layer[name] for cache in bucket.cached_history for layer in cache for name in ('k', 'v')]
        self.assertEqual(len(held), 80)
        active = {id(layer[name]) for device in devices for layer in device.kv_history.active for name in ('k', 'v')}
        self.assertFalse(active & {id(value) for value in held}, 'placeholders, never the devices\' own banks')
        call = devices[0].execute_proposal.call_args
        self.assertIsInstance(call.kwargs['quad'], octo.OctoPass)
        self.assertEqual(call.kwargs['pack'], [dict(position=device.position, history_rows=2048) for device in devices])
        self.assertEqual([line for line in lines if line.startswith(octo.MARKER)],
                         ['[PINDIAG] octo draft engaged slots=[0,1,2,3,4,5,6,7] heads=64/16 rows=64 block=8 conv=110'])
        self.assertEqual(ledger.call_args.args[0], 'octo_draft_built')
        self.assertEqual(ledger.call_args.kwargs['point'], 'slots=0,1,2,3,4,5,6,7')
        self.assertTrue(trace.last_built)

    def test_live_banks_bind_the_pools_active_banks_and_own_no_placeholders(self):
        with patch.dict(os.environ, {'QWEN_FAST_FUSED_COMMIT': '1', 'QWEN_FAST_FUSED_COMMIT_INPLACE': '1', 'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS': '1'}):
            ops, devices, trace = self.build()
            for device in devices:
                device.pool_slot.kv = [dict(active=dict(layer)) for layer in device.kv_history.active]
            with base.logged()[1], patch('memory_ledger.record'):
                trace.prepare_device(tuple(range(8)))
        bucket = trace.buckets[(2048,) * 8]
        self.assertTrue(bucket.live_banks)
        uploads = [event for event in ops.events if event[0] == 'from_torch' and event[2] is True]
        self.assertEqual(len(uploads), 6, 'ids, mask and the four rope tables only')
        for device, cache in zip(devices, bucket.cached_history):
            for active, held in zip(device.kv_history.active, cache):
                self.assertIs(held['k'], active['k'])

    def test_every_rounds_host_inputs_are_the_halves_packed_uploads(self):
        ops, devices, trace = self.build()
        seeds = tuple(101 * (user + 1) for user in range(8))
        with base.logged()[1], patch('memory_ledger.record'):
            trace.prepare_device(seeds)
        bucket = trace.buckets[(2048,) * 8]
        ids = bucket.identifiers.value[0].tolist()
        for user, seed in enumerate(seeds):
            self.assertEqual(ids[8 * user:8 * user + 8], [seed] + [248070] * 7)
        query, live = octo.octo_rope(octo.octo_users(devices))
        for name, tables in (('q', query), ('live_k', live)):
            for table, theirs in zip(bucket.rope[name], tables):
                self.assertTrue(same_bits(table.value, theirs), name)

    def test_prepare_collect_adopt_finish_use_this_modules_readback(self):
        ops, devices, trace = self.build()
        parts = [dict(user=user) for user in range(8)]
        with base.logged()[1], patch('memory_ledger.record'), patch.object(octo, 'read_octo_outputs', return_value=parts) as read:
            trace.prepare_device(tuple(range(11, 19)))
            collected = trace.collect()
            read.assert_called_once()
            self.assertEqual(collected, dict(parts=parts, seeds=tuple(range(11, 19)), counts=(7,) * 8))
            with self.assertRaises(ValueError):
                trace.adopt([(1,), (2,)])
            trace.adopt([(user, user + 1, user + 2) for user in range(8)])
            with self.assertRaises(ValueError):
                trace.collect()
            self.assertTrue(trace.has_pending(3, 14))
            self.assertFalse(trace.has_pending(3, 15))
            self.assertEqual(trace.finish(3, 2), (3, 4))
            self.assertFalse(trace.has_pending(3, 14), 'a consumed user is not pending again')
            for user in (0, 1, 2, 4, 5, 6, 7):
                trace.finish(user, 1)
            self.assertIsNone(trace._pending)

    def test_a_ticket_wider_than_seven_proposals_is_an_error_never_a_short_ticket(self):
        ops, devices, trace = self.build()
        with base.logged()[1], patch('memory_ledger.record'), patch.object(octo, 'select_octo_outputs', return_value=tuple((1,) * 7 for _ in range(8))):
            trace.prepare_device(tuple(range(8)))
            for count in (8, 15, 0, True):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    trace.finish(0, count)
            self.assertEqual(trace.finish(0, 7), (1,) * 7)

    def test_finish_without_a_batched_selection_selects_with_this_modules_reader(self):
        ops, devices, trace = self.build()
        with base.logged()[1], patch('memory_ledger.record'), \
                patch.object(octo, 'select_octo_outputs', return_value=tuple((user,) for user in range(8))) as select:
            trace.prepare_device(tuple(range(11, 19)))
            self.assertEqual(trace.finish(1, 1), (1,))
            self.assertEqual(select.call_args.args[2:], (tuple(range(11, 19)), (7,) * 8))
            self.assertEqual(trace.audit_selection(), tuple((user,) for user in range(8)))

    def test_a_failed_build_releases_every_placeholder(self):
        ops, devices, trace = self.build()
        devices[0].execute_proposal.side_effect = RuntimeError('capture failed')
        with base.logged()[1], self.assertRaises(RuntimeError):
            trace.prepare_device(tuple(range(8)))
        self.assertEqual(trace.buckets, {})
        self.assertEqual(trace.owned, [])
        freed = [event for event in ops.events if event[0] == 'deallocate']
        self.assertEqual(len(freed), 6 + 80)
        self.assertEqual(devices[0].validated_native_proposal_masks, set())

    def test_close_releases_the_trace_and_the_placeholders(self):
        ops, devices, trace = self.build()
        with base.logged()[1], patch('memory_ledger.record'):
            trace.prepare_device(tuple(range(8)))
        trace.close()
        self.assertTrue(trace.closed)
        self.assertEqual([event[0] for event in ops.events].count('release_trace'), 1)
        self.assertEqual(trace.owned, [])
        self.assertFalse(trace.prepare_device(tuple(range(8))))

    def test_a_device_mid_publication_or_closed_declines_and_the_constructor_refuses_what_it_cannot_serve(self):
        for attribute, value in (('pending', object()), ('closed', True), ('progress', object())):
            ops, devices, trace = self.build()
            setattr(devices[7], attribute, value)
            with self.subTest(attribute=attribute):
                self.assertFalse(trace.prepare_device(tuple(range(8))))
                self.assertEqual(trace.buckets, {})
        ops = RecordingOps4()
        devices = octo_devices8(ops)
        with self.assertRaises(ValueError):
            octo.PreparedOctoDFlashProposal(devices[:4])
        devices[5].block_rows = 8
        with self.assertRaises(ValueError):
            octo.PreparedOctoDFlashProposal(devices)
        with self.assertRaises(ValueError):
            octo.PreparedOctoDFlashProposal(octo_devices8(RecordingOps4())).prepare_device((1, 2, 3))

    def test_a_pool_that_holds_the_octo_mask_and_outputs_is_borrowed_as_the_quad_does(self):
        import dflash_proposal_trace

        ops, devices, trace = self.build()
        with base.logged()[1], patch('memory_ledger.record'), \
                patch('dflash_proposal_trace.borrow_pooled_mask', wraps=dflash_proposal_trace.borrow_pooled_mask) as mask, \
                patch('dflash_proposal_trace.pool_outputs', wraps=dflash_proposal_trace.pool_outputs) as outputs:
            trace.prepare_device(tuple(range(8)))
        self.assertEqual(mask.call_args.args[1], list(range(8)))
        self.assertEqual(outputs.call_args.args[1], list(range(8)))

    def test_every_round_copies_each_users_active_bank_and_follows_a_swap(self):
        ops, devices, trace = self.build()
        with base.logged()[1], patch('memory_ledger.record'):
            trace.prepare_device(tuple(range(8)))
            del ops.events[:]
            first = devices[2].kv_history.active
            devices[2].kv_history.active = fresh_banks(ops)
            trace.prepare_device(tuple(range(8)))
        copies = [event for event in ops.events if event[0] == 'copy']
        self.assertEqual(len(copies), 80, '8 users x 5 layers x k and v')
        self.assertFalse(trace.last_built)


# ---------------------------------------------------------------------------------------------
# Admission and refusal.
# ---------------------------------------------------------------------------------------------

def refusal_devices(chips=4, count=8, **overrides):
    layers, predecessors, successors = object(), object(), object()
    mesh = SimpleNamespace(shape=[1, chips], compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
    operations = object()
    devices = []
    for _ in range(count):
        fields = dict(operations=operations, mesh=mesh, block_rows=16, fused_convolution=True, layers=layers, predecessors=predecessors, successors=successors)
        fields.update(overrides)
        devices.append(SimpleNamespace(**fields))
    return devices


class RefusalTests(unittest.TestCase):
    def refuse(self, devices=None, batched=(), environ=None):
        return octo.refusal(refusal_devices() if devices is None else devices, [] if batched == () else batched, dict(FLAGS) if environ is None else environ)

    def test_eight_packable_devices_are_served(self):
        self.assertIsNone(self.refuse())

    def test_each_refusal_names_its_reason(self):
        self.assertEqual(self.refuse(environ={}), 'the octo draft serves QWEN_FAST_TP=4 only')
        self.assertEqual(self.refuse(environ=dict(FLAGS, QWEN_FAST_PACKED_PROPOSAL='0', QWEN_FAST_ROUND_B1='0')),
                         'requires QWEN_FAST_PACKED_PROPOSAL=1,QWEN_FAST_ROUND_B1=1')
        live = dict(FLAGS, QWEN_FAST_FUSED_COMMIT_LIVE_BANKS='1')
        self.assertEqual(self.refuse(environ=live), 'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS=1 needs QWEN_FAST_FUSED_COMMIT=1,QWEN_FAST_FUSED_COMMIT_INPLACE=1 (the live bank must never move)')
        self.assertIsNone(self.refuse(environ=dict(live, QWEN_FAST_FUSED_COMMIT='1', QWEN_FAST_FUSED_COMMIT_INPLACE='1')))
        self.assertEqual(self.refuse(batched=None), 'requires the batched selection (QWEN_FAST_ROUND_B1=1)')
        self.assertEqual(self.refuse(refusal_devices(count=4)), 'the octo draft needs 8 devices, not 4')
        self.assertEqual(self.refuse(refusal_devices(chips=2)), 'the eight devices are not on the (1, 4) mesh')
        self.assertEqual(self.refuse(refusal_devices(block_rows=8)), 'the eight devices are not T16 (the prepared layers the pass reuses)')
        self.assertEqual(self.refuse(refusal_devices(fused_convolution=False)), 'requires the fused learned convolution')
        other = refusal_devices()
        other[6].layers = object()
        self.assertEqual(self.refuse(other), 'the eight devices do not share one draft weight set')
        split = refusal_devices()
        split[1].mesh = refusal_devices()[0].mesh
        self.assertEqual(self.refuse(split), 'the eight devices do not share one mesh')
        small = refusal_devices()
        small[0].mesh.compute_with_storage_grid_size = lambda: SimpleNamespace(x=8, y=10)
        self.assertEqual(self.refuse(small), 'the compute grid cannot hold QWEN_FAST_OCTO_DRAFT_CONV=110')
        self.assertIsNone(self.refuse(small, environ=dict(FLAGS, QWEN_FAST_OCTO_DRAFT_CONV='80')))
        with self.assertRaises(ValueError):
            self.refuse(environ=dict(FLAGS, QWEN_FAST_OCTO_DRAFT_CONV='dense'))


class AdmissionTests(unittest.TestCase):
    M3 = (True, 'users=8')

    def environ(self, **changes):
        import test_octo_profiles

        env = test_octo_profiles.container_env('c2-packed-tp4-8x262k-ship-prefix-levern-octo')
        env.update(QWEN_FAST_OCTO_DRAFT='1', QWEN_FAST_PIPELINED_PROPOSALS='1', QWEN_FAST_PACKED_PROPOSAL='1', QWEN_FAST_PAIR_ROW_EXACT='1', QWEN_FAST_ROUND_B1='1')
        env.update(changes)
        return env

    def test_the_flag_off_admission_is_unchanged(self):
        import serving_octo
        import test_octo as host

        env = self.environ()
        env.pop('QWEN_FAST_OCTO_DRAFT')
        log = host.Lines()
        record = serving_octo.octo_admission(self.M3, env, log=log)
        self.assertNotIn('draft', record)
        self.assertEqual(len([line for line in log.lines if line.startswith(serving_octo.UNQUALIFIED_MARKER)]), len(serving_octo.UNQUALIFIED_ITEMS))
        self.assertFalse([line for line in log.lines if 'OCTO-DRAFT' in line])

    def test_the_flag_on_adds_its_own_lines_and_leaves_the_six_unqualified_lines_six(self):
        import serving_octo
        import test_octo as host

        log = host.Lines()
        record = serving_octo.octo_admission(self.M3, self.environ(), log=log)
        self.assertTrue(record['draft'])
        self.assertEqual(len([line for line in log.lines if line.startswith(serving_octo.UNQUALIFIED_MARKER)]), len(serving_octo.UNQUALIFIED_ITEMS))
        draft = [line for line in log.lines if line.startswith('[OCTO-DRAFT]')]
        self.assertEqual(len(draft), len(octo.UNQUALIFIED_ITEMS) + 1)
        self.assertTrue(draft[-1].startswith('[OCTO-DRAFT] admitted rows=64 users=8 block=8 proposals=7 (gate only)'), draft)

    def test_the_draft_reasons_join_the_octo_refusal(self):
        import serving_octo
        import test_octo as host

        for name, value in (('QWEN_FAST_PIPELINED_PROPOSALS', '0'), ('QWEN_FAST_PACKED_PROPOSAL', None), ('QWEN_FAST_QUAD_DRAFT_AUDIT', '1'),
                            ('QWEN_FAST_PARKED_ENGINES', '1'), ('QWEN_FAST_DRAFT_SINGLES_AUDIT', 'all')):
            env = self.environ()
            if value is None:
                env.pop(name)
            else:
                env[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError) as raised:
                serving_octo.octo_admission(self.M3, env, log=host.Lines())
            self.assertIn(name, str(raised.exception))
            self.assertIn('QWEN_FAST_OCTO is refused', str(raised.exception).replace('QWEN_FAST_OCTO=alternate', 'QWEN_FAST_OCTO'))

    def test_a_bad_value_is_a_configuration_error_naming_the_flag(self):
        import serving_octo
        import test_octo as host

        with self.assertRaisesRegex(ValueError, 'QWEN_FAST_OCTO_DRAFT must be 0 or 1'):
            serving_octo.octo_admission(self.M3, self.environ(QWEN_FAST_OCTO_DRAFT='yes'), log=host.Lines())

    def test_the_flag_without_the_octo_block_is_refused_by_name(self):
        lines = []
        with self.assertRaises(ValueError) as raised:
            octo.admission(dict(FLAGS, QWEN_FAST_OCTO='off'), log=lines.append)
        self.assertIn('QWEN_FAST_OCTO_DRAFT=1 needs QWEN_FAST_OCTO=live|alternate', str(raised.exception))
        self.assertTrue(lines and all(line.startswith(octo.REFUSED_LINE) for line in lines))
        self.assertIsNone(octo.admission({}, log=lines.append))
        self.assertIsNone(octo.admission({octo.FLAG: '0'}, log=lines.append))
        record = octo.admission(dict(FLAGS), log=lines.append)
        self.assertEqual(record, dict(rows=64, users=8, block=8, proposals=7))

    def test_the_attach_refuses_the_bundle_flag_beside_no_octo_block_too(self):
        text = (HERE / 'serving_runtime.py').read_text(encoding='utf-8')
        self.assertIn('octo_attn_bundle.admission_problems()', text)
        import octo_attn_bundle

        self.assertTrue(octo_attn_bundle.admission_problems(dict(QWEN_FAST_OCTO_ATTN_BUNDLE='4', QWEN_FAST_TP='4')), 'no octo block: refused by name')

    def test_the_attach_refuses_the_flag_beside_no_octo_block(self):
        text = (HERE / 'serving_runtime.py').read_text(encoding='utf-8')
        self.assertIn("elif os.environ.get('QWEN_FAST_OCTO_DRAFT', '0') != '0':", text)
        self.assertIn('octo_draft_tp.admission(log=pindiag)', text)


# ---------------------------------------------------------------------------------------------
# The coordinator.
# ---------------------------------------------------------------------------------------------

class FakeOctoTrace:
    instances = []
    failures = []

    def __init__(self, devices):
        self.devices = tuple(devices)
        self.prepared, self.discards = [], 0
        self.closed, self.last_built, self.round_number = False, False, None
        self.buckets = {}
        FakeOctoTrace.instances.append(self)

    @property
    def device_a(self):
        return self.devices[0]

    def prepare_device(self, seeds):
        base.LOG.append(('octo', tuple(seeds)))
        if FakeOctoTrace.failures and FakeOctoTrace.failures.pop(0):
            raise RuntimeError('octo capture failed')
        self.last_built = not self.buckets
        self.buckets[(2048,) * 8] = True
        self.prepared.append(tuple(seeds))
        return True

    def has_pending(self, which, seed):
        return bool(self.prepared) and seed == self.prepared[-1][which]

    def finish(self, which, count):
        return ('octo', which, count)

    def collect(self):
        base.LOG.append(('collect', 'octo'))
        return dict(parts=[dict(user=user) for user in range(8)], seeds=self.prepared[-1], counts=(7,) * 8)

    def adopt(self, tokens):
        self.adopted = tuple(tokens)

    def run_audit(self, round_number):
        return None

    def discard_pending(self):
        self.discards += 1

    def close(self):
        self.closed = True


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        flags = dict(FLAGS, QWEN_FAST_PACKED_AUDIT='1')
        environment = clean_environment(**flags)
        environment.start()
        self.addCleanup(environment.stop)
        cards = four_cards()
        cards.__enter__()
        self.addCleanup(cards.__exit__, None, None, None)
        del base.LOG[:]
        FakeOctoTrace.instances, FakeOctoTrace.failures = [], []
        base.FakeQuadTrace.instances, base.FakeQuadTrace.failures = [], []
        self.Pair = base.pair_trace_class()
        self.Pair.instances = []
        from test_dflash_packed_proposal_coordinator import FakeSingleUserCapture

        for target, value in (('octo_draft_tp.PreparedOctoDFlashProposal', FakeOctoTrace),
                              ('quad_draft_tp.PreparedQuadDFlashProposal', base.FakeQuadTrace),
                              ('dflash_proposal_trace.PreparedPackedDFlashProposal', self.Pair),
                              ('dflash_proposal_trace.PreparedDFlashProposal', FakeSingleUserCapture),
                              ('dflash_packed_proposal.select_packed_batched', Mock(side_effect=base.selected_tokens))):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: base.LOG.append(('sync',))))
        self.mesh = MESH4

    def coordinator(self):
        import dflash_packed_proposal_coordinator

        return dflash_packed_proposal_coordinator.PackedProposalCoordinator()

    def prepare(self, coordinator, bridges, **options):
        lines, loguru = base.logged()
        with loguru:
            prepared = coordinator.prepare(bridges, **options)
        return prepared, lines

    def bridges(self, slots=tuple(range(8))):
        return base.quad_bridges(self.operations, self.mesh, slots)

    def test_eight_steady_users_run_one_octo_pass_and_no_quad_or_pair(self):
        coordinator, bridges = self.coordinator(), self.bridges()
        with patch.object(octo, 'refusal', wraps=octo.refusal) as refusal:
            prepared, lines = self.prepare(coordinator, bridges, octo_draft=True)
        refusal.assert_called_once()
        self.assertEqual(len(FakeOctoTrace.instances), 1)
        self.assertEqual(base.FakeQuadTrace.instances, [])
        self.assertEqual(self.Pair.instances, [])
        self.assertEqual(FakeOctoTrace.instances[0].prepared, [tuple(range(100, 108))])
        self.assertEqual(base.LOG, [('octo', tuple(range(100, 108))), ('sync',), ('collect', 'octo')])
        self.assertEqual(len(prepared), 8)
        self.assertIn('[OCTO-DRAFT] round=1 built=1 ms=', lines[0])
        self.assertFalse([line for line in lines if 'fallback' in line or 'disabled' in line])
        self.assertEqual(coordinator.octo_rounds, 1)
        view = prepared[3].proposal_capture
        self.assertTrue(view.has_pending(103))
        self.assertEqual(view.finish(7), ('octo', 3, 7))

    def test_the_next_round_replays_the_same_trace(self):
        coordinator, bridges = self.coordinator(), self.bridges()
        self.prepare(coordinator, bridges, octo_draft=True)
        _, lines = self.prepare(coordinator, bridges, octo_draft=True)
        self.assertEqual(len(FakeOctoTrace.instances), 1)
        self.assertEqual(len(FakeOctoTrace.instances[0].prepared), 2)
        self.assertIn('[OCTO-DRAFT] round=2 built=0 ms=', lines[0])

    def test_without_the_keyword_the_round_is_todays_even_with_the_flag_set(self):
        coordinator, bridges = self.coordinator(), self.bridges()
        with patch.dict(os.environ, {'QWEN_FAST_QUAD_DRAFT_BLOCKS': '2'}):
            self.prepare(coordinator, bridges)
        self.assertEqual(FakeOctoTrace.instances, [])
        self.assertIsNone(coordinator.octo)
        self.assertEqual(coordinator.octo_rounds, 0)

    def test_six_and_seven_live_octo_rounds_keep_the_quads_and_the_pairs(self):
        for slots in ((0, 1, 2, 3, 4, 5, 6), (0, 1, 2, 3, 4, 5), (1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 5, 6, 7)):
            with self.subTest(slots=slots):
                del base.LOG[:]
                FakeOctoTrace.instances = []
                coordinator = self.coordinator()
                with patch.dict(os.environ, {'QWEN_FAST_QUAD_DRAFT_BLOCKS': '2'}):
                    self.prepare(coordinator, self.bridges(slots), octo_draft=True)
                self.assertEqual(FakeOctoTrace.instances, [])
                self.assertIsNone(coordinator.octo_blocked)

    def test_a_refusal_blocks_it_once_with_the_reason_and_the_round_goes_on(self):
        os.environ['QWEN_FAST_FUSED_COMMIT_LIVE_BANKS'] = '1'
        coordinator, bridges = self.coordinator(), self.bridges()
        _, lines = self.prepare(coordinator, bridges, octo_draft=True)
        _, more = self.prepare(coordinator, bridges, octo_draft=True)
        disabled = [line for line in lines + more if line.startswith(octo.DISABLED_MARKER)]
        self.assertEqual(len(disabled), 1)
        self.assertIn('reason=QWEN_FAST_FUSED_COMMIT_LIVE_BANKS=1_needs_QWEN_FAST_FUSED_COMMIT=1', disabled[0])
        self.assertEqual(FakeOctoTrace.instances, [])

    def test_a_short_headroom_falls_back_without_building(self):
        coordinator, bridges = self.coordinator(), self.bridges()
        with patch('dflash_packed_proposal_coordinator.capture_headroom', return_value=(('largest_free',), dict(largest_free=123))):
            _, lines = self.prepare(coordinator, bridges, octo_draft=True)
        self.assertEqual(FakeOctoTrace.instances, [])
        self.assertTrue(any(line.startswith('[OCTO-DRAFT] fallback round=1 reason=dram_reserve:headroom=123') for line in lines), lines)
        self.assertIsNone(coordinator.octo_blocked, 'a short round is not a give-up')

    def test_a_failure_falls_back_and_two_in_a_row_block_it(self):
        coordinator, bridges = self.coordinator(), self.bridges()
        FakeOctoTrace.failures = [True, True]
        _, lines = self.prepare(coordinator, bridges, octo_draft=True)
        self.assertTrue(lines[0].startswith('[OCTO-DRAFT] fallback round=1 reason=RuntimeError:octo_capture_failed'), lines)
        self.assertGreaterEqual(FakeOctoTrace.instances[0].discards, 1, 'the failed attempt\'s pending is dropped (and again, harmlessly, by the round that did not form the pass)')
        _, lines = self.prepare(coordinator, bridges, octo_draft=True)
        self.assertIsNotNone(coordinator.octo_blocked)
        self.assertTrue(any(line.startswith('[PINDIAG] octo draft disabled round=2 failures=2 ') for line in lines), lines)
        _, more = self.prepare(coordinator, bridges, octo_draft=True)
        self.assertEqual(len(FakeOctoTrace.instances), 1, 'blocked: no new trace')
        self.assertFalse([line for line in more if line.startswith(octo.DISABLED_MARKER)], 'the disabled marker is logged once')

    def test_a_success_resets_the_failure_count(self):
        coordinator, bridges = self.coordinator(), self.bridges()
        FakeOctoTrace.failures = [True, False, True]
        for _ in range(3):
            self.prepare(coordinator, bridges, octo_draft=True)
        self.assertIsNone(coordinator.octo_blocked)

    def test_a_closed_member_retires_the_trace_and_the_next_round_builds_a_new_one(self):
        coordinator, bridges = self.coordinator(), self.bridges()
        prepared, _ = self.prepare(coordinator, bridges, octo_draft=True)
        first = FakeOctoTrace.instances[0]
        prepared[5].closed = True
        lines, loguru = base.logged()
        with loguru:
            coordinator.release_closed()
        self.assertTrue(first.closed)
        self.assertIsNone(coordinator.octo)
        self.assertTrue(any(line.startswith('[OCTO-DRAFT] released slots=0,1,2,3,4,5,6,7') for line in lines), lines)
        prepared[5].closed = False
        self.prepare(coordinator, bridges, octo_draft=True)
        self.assertEqual(len(FakeOctoTrace.instances), 2, 'a fresh trace for the next round')

    def test_a_round_that_does_not_form_the_pass_drops_an_earlier_unconsumed_proposal(self):
        coordinator, bridges = self.coordinator(), self.bridges()
        self.prepare(coordinator, bridges, octo_draft=True)
        trace = FakeOctoTrace.instances[0]
        self.assertEqual(trace.discards, 0)
        self.prepare(coordinator, bridges)      # the next round is not an octo round
        self.assertEqual(trace.discards, 1, 'the stale pending is dropped before any device can be answered by it')
        self.prepare(coordinator, bridges[:7], octo_draft=True)    # an octo round with seven live
        self.assertEqual(trace.discards, 2)

    def test_close_closes_the_trace(self):
        coordinator, bridges = self.coordinator(), self.bridges()
        self.prepare(coordinator, bridges, octo_draft=True)
        coordinator.close()
        self.assertTrue(FakeOctoTrace.instances[0].closed)
        self.assertIsNone(coordinator.octo)

    def test_the_pooled_shapes_carry_slots_0_to_7_with_the_flag_and_only_with_it(self):
        import dflash_packed_proposal_coordinator as coordinator

        masks = coordinator.pooled_draft_mask_shapes(8, 16)
        self.assertEqual(masks[tuple(range(8))], (1, 1, 32, 2080))
        outputs = coordinator.pooled_draft_output_shapes(8, 16)
        spec = outputs[tuple(range(8))]
        self.assertEqual(spec['head'], (1, 1, 64, 16))
        self.assertEqual(spec['projected'], (1, 1, 64, 256))
        self.assertEqual(tuple(spec['chunks']), ((0, 32768), (32768, 62080)))
        with patch.dict(os.environ, {'QWEN_FAST_OCTO_DRAFT': '0'}):
            self.assertNotIn(tuple(range(8)), coordinator.pooled_draft_mask_shapes(8, 16))
            self.assertNotIn(tuple(range(8)), coordinator.pooled_draft_output_shapes(8, 16))
        os.environ.pop('QWEN_FAST_OCTO_DRAFT')
        self.assertNotIn(tuple(range(8)), coordinator.pooled_draft_mask_shapes(8, 16))
        self.assertNotIn(tuple(range(8)), coordinator.pooled_draft_output_shapes(4, 16))


# ---------------------------------------------------------------------------------------------
# The hook.
# ---------------------------------------------------------------------------------------------

class HookTests(unittest.TestCase):
    def hook(self, *, octo_block, groups):
        """A FastWorkerHook stub far enough for _drafts_by_block: one bridge per request, a coordinator that records its prepare calls."""
        import serving_worker_hook

        class Bridge:
            def __init__(self, name):
                self.request = SimpleNamespace(session=SimpleNamespace(request_id=name, finished=False, pending=None, phase='idle'), name=name)
                self.calls = []

            def drafts(self, packed_rows=None):
                self.calls.append(packed_rows)
                return None

        bridges = [Bridge('r%d' % index) for index in range(8)]
        hook = serving_worker_hook.FastWorkerHook.__new__(serving_worker_hook.FastWorkerHook)
        hook.packed_step = SimpleNamespace(octo=octo_block, while_waiting_groups=None)
        coordinator = Mock()
        hook._packed_coordinator = coordinator
        return hook, bridges, coordinator

    def drive(self, flag, octo_block, group_block, rows=8):
        hook, bridges, coordinator = self.hook(octo_block=octo_block, groups=None)
        groups = [(group_block, rows, [bridge.request for bridge in bridges])]
        env = {'QWEN_FAST_PIPELINED_PROPOSALS': '1', 'QWEN_FAST_PACKED_PROPOSAL': '1'}
        if flag is not None:
            env['QWEN_FAST_OCTO_DRAFT'] = flag
        with patch.dict(os.environ, env), patch('serving_worker_hook.discard_stale_ticket'), patch('serving_worker_hook.phase', side_effect=lambda name, ids, call: call()), \
                patch('serving_worker_hook.budget_cap_enabled', return_value=False), patch('serving_worker_hook.real_remaining_budget', return_value=None), \
                patch('serving_worker_hook.window_flags_on', return_value=False), patch('dflash_packed_proposal.packed_proposal_enabled', return_value=True), \
                patch('vllm.v1.outputs.DraftTokenIds', create=True, new=Mock()):
            try:
                hook._drafts_by_block(bridges, groups, None)
            except Exception:
                pass
        return coordinator.prepare.call_args

    def test_an_octo_round_with_the_flag_asks_for_the_octo_draft(self):
        block = object()
        call = self.drive('1', block, block)
        self.assertIsNotNone(call)
        self.assertEqual(call.kwargs, {'octo_draft': True})

    def test_every_other_round_makes_todays_call(self):
        block = object()
        for label, flag, octo_block, group_block, rows in (('flag unset', None, block, block, 8), ('flag 0', '0', block, block, 8), ('an m3 group', '1', block, object(), 16),
                                                           ('no octo block', '1', None, object(), 16), ('narrowed', '1', block, block, None)):
            with self.subTest(label):
                call = self.drive(flag, octo_block, group_block, rows)
                self.assertIsNotNone(call)
                self.assertEqual(call.kwargs, {})


# ---------------------------------------------------------------------------------------------
# Off is today.
# ---------------------------------------------------------------------------------------------

class OffIsTodayTests(unittest.TestCase):
    def test_the_pair_process_never_loads_the_module(self):
        code = ('import sys; sys.path.insert(0, %r); import tp_addresses, quad_draft, dflash_packed_proposal_coordinator, serving_octo, serving_worker_hook; '
                "assert 'octo_draft_tp' not in sys.modules, 'the pair loaded the octo draft'; assert 'quad_draft_tp' not in sys.modules" % str(HERE))
        environment = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_')}
        subprocess.run([sys.executable, '-B', '-c', code], env=environment, check=True, timeout=120)

    def test_the_quad_modules_are_the_bytes_they_were(self):
        import hashlib

        from test_quad_draft_tp4 import PINNED_SHA256

        for name, digest in PINNED_SHA256.items():
            data = (HERE / name).read_bytes().replace(b'\r\n', b'\n')
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest, name)

    def test_prepare_without_the_keyword_runs_the_quad_rounds_call_for_call(self):
        """The same eight-live quad round with the coordinator as it stands and with octo_draft=False spelled out: one log, one set of lines."""
        import dflash_packed_proposal_coordinator as coordinator_module

        def run(**options):
            with clean_environment(**dict(FLAGS, QWEN_FAST_OCTO_DRAFT='0', QWEN_FAST_QUAD_DRAFT_BLOCKS='2', QWEN_FAST_PACKED_AUDIT='1')), four_cards():
                del base.LOG[:]
                base.FakeQuadTrace.instances, base.FakeQuadTrace.failures = [], []
                pair = base.pair_trace_class()
                pair.instances = []
                from test_dflash_packed_proposal_coordinator import FakeSingleUserCapture

                operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: base.LOG.append(('sync',))))
                with patch('quad_draft_tp.PreparedQuadDFlashProposal', base.FakeQuadTrace), patch('dflash_proposal_trace.PreparedPackedDFlashProposal', pair), \
                        patch('dflash_proposal_trace.PreparedDFlashProposal', FakeSingleUserCapture), \
                        patch('dflash_packed_proposal.select_packed_batched', Mock(side_effect=base.selected_tokens)):
                    coordinator = coordinator_module.PackedProposalCoordinator()
                    bridges = base.quad_bridges(operations, MESH4, tuple(range(8)))
                    lines, loguru = base.logged()
                    with loguru:
                        coordinator.prepare(bridges, **options)
                return list(base.LOG), [base.strip_ms(line) for line in lines], len(base.FakeQuadTrace.instances)

        self.assertEqual(run(), run(octo_draft=False))

    def test_the_coordinator_carries_no_octo_state_a_round_reads_without_the_keyword(self):
        import dflash_packed_proposal_coordinator as module

        coordinator = module.PackedProposalCoordinator()
        self.assertEqual((coordinator.octo, coordinator.octo_failures, coordinator.octo_blocked, coordinator.octo_rounds), (None, 0, None, 0))
        self.assertFalse(module.octo_draft_requested({}))
        self.assertFalse(module.octo_draft_requested({'QWEN_FAST_OCTO_DRAFT': '0'}))
        self.assertTrue(module.octo_draft_requested({'QWEN_FAST_OCTO_DRAFT': '1'}))

    def test_the_pooled_shapes_without_the_flag_are_todays(self):
        import dflash_packed_proposal_coordinator as module

        with clean_environment(**{name: value for name, value in FLAGS.items() if name != 'QWEN_FAST_OCTO_DRAFT'}), four_cards():
            masks = module.pooled_draft_mask_shapes(8, 16)
            outputs = module.pooled_draft_output_shapes(8, 16)
        self.assertNotIn(tuple(range(8)), masks)
        self.assertNotIn(tuple(range(8)), outputs)


# ---------------------------------------------------------------------------------------------
# The smoke rule (octo_judge.draft_problems), on a log the producers write.
# ---------------------------------------------------------------------------------------------

class DraftJudgeTests(unittest.TestCase):
    def env(self, **extra):
        import test_octo_judge as judged

        return judged.env_of('alternate', 6, QWEN_FAST_OCTO_DRAFT='1', **extra)

    def boot(self, pairs=20, *, draft_rounds=None, admitted=True, engaged=True, fallback=False, disabled=False, refused=False, conv='110'):
        """An alternate octo boot (test_octo_judge.Boot: the real admission and OctoState lines) with the draft pass's lines in the order the coordinator writes them."""
        import test_octo_judge as judged

        boot = judged.Boot(programs=lambda: 5000)
        if admitted:
            octo.log_admitted(boot.lines.append)
        if refused:
            boot.lines.append(octo.REFUSED_LINE + ': because')
        for index in range(pairs):
            boot.round('octo')
            if draft_rounds is None or index < draft_rounds:
                if engaged and index == 0:
                    with four_cards():
                        octo.note(list(range(8)), conv, log=boot.lines.append)
                boot.lines.append(octo.ROUND_LINE.format(round=index + 1, built=int(index == 0), ms='450.0' if index == 0 else '6.1'))
            boot.round('m3')
        if fallback:
            boot.lines.append(octo.FALLBACK_LINE.format(round=3, reason='dram_reserve:headroom=100'))
        if disabled:
            boot.lines.append('%s round=9 failures=2 reason=consecutive_failures=2' % octo.DISABLED_MARKER)
        return boot.text()

    def judge(self, text, **extra):
        import octo_judge

        with patch.object(octo, '_NOTED', []):
            return octo_judge.judge(self.env(**extra), text)

    def test_a_clean_boot_passes_and_reports_the_passs_rounds(self):
        with patch.object(octo, '_NOTED', []):
            text = self.boot()
        problems, facts = self.judge(text)
        self.assertEqual(problems, [])
        self.assertEqual((facts['draft_rounds'], facts['draft_builds'], facts['draft_fallbacks'], facts['draft_octo_rounds_at_8']), (20, 1, 0, 20))

    def test_each_missing_piece_is_named(self):
        cases = (('not admitted', dict(admitted=False), 'admitted lines (one wanted)'), ('refused', dict(refused=True), 'refused line(s)'),
                 ('never engaged', dict(engaged=False), 'octo draft engaged'), ('too few rounds', dict(draft_rounds=5), '5 round(s) were drafted by the pass'),
                 ('a fallback', dict(fallback=True), 'fallback round(s)'), ('given up', dict(disabled=True), 'given up for the process'),
                 ('most octo rounds drafted by the quads', dict(pairs=40, draft_rounds=16), 'most octo rounds were drafted by the quads'))
        for label, options, fragment in cases:
            with patch.object(octo, '_NOTED', []):
                text = self.boot(**options)
            problems, facts = self.judge(text)
            with self.subTest(label):
                self.assertTrue(problems, label)
                if fragment:
                    self.assertTrue(any(fragment in problem for problem in problems), problems)

    def test_the_conv_mode_the_profile_asks_is_the_one_that_engaged(self):
        with patch.object(octo, '_NOTED', []):
            text = self.boot(conv='80')
        problems, facts = self.judge(text)
        self.assertTrue(any('engaged with conv=80 and the profile asks 110' in problem for problem in problems), problems)
        self.assertEqual(self.judge(text, QWEN_FAST_OCTO_DRAFT_CONV='80')[0], [])

    def test_the_flag_off_arm_passes_with_no_line_and_fails_on_any(self):
        import octo_judge
        import test_octo_judge as judged

        boot = judged.Boot(programs=lambda: 5000)
        boot.alternate(20)
        env = judged.env_of('alternate', 6)
        self.assertEqual(octo_judge.judge(env, boot.text())[0], [])
        with patch.object(octo, '_NOTED', []):
            leaky = self.boot()
        problems, facts = octo_judge.judge(env, leaky)
        self.assertTrue(any('octo-draft line(s) on a profile without QWEN_FAST_OCTO_DRAFT' in problem for problem in problems), problems)
        problems, facts = octo_judge.judge(dict(env, QWEN_FAST_OCTO_DRAFT='0'), leaky)
        self.assertTrue(any('octo-draft line(s)' in problem for problem in problems), problems)

    def test_a_bad_value_and_the_flag_beside_no_octo_mode_are_named(self):
        import octo_judge

        self.assertTrue(any('neither 0 nor 1' in problem for problem in octo_judge.judge(dict(self.env(), QWEN_FAST_OCTO_DRAFT='2'), '')[0]))
        env = self.env()
        env['QWEN_FAST_OCTO'] = 'off'
        self.assertTrue(any('needs QWEN_FAST_OCTO=live or alternate' in problem for problem in octo_judge.judge(env, '')[0]))

    def test_the_producers_lines_are_the_markers_templates(self):
        import octo_markers

        lines = []
        octo.log_admitted(lines.append)
        with patch.object(octo, '_NOTED', []), four_cards():
            octo.note(list(range(8)), '110', log=lines.append)
        lines.append(octo.ROUND_LINE.format(round=7, built=1, ms='12.5'))
        lines.append(octo.FALLBACK_LINE.format(round=8, reason='dram_reserve:headroom=1'))
        scan = octo_markers.draft_scan('\n'.join(lines))
        self.assertEqual((scan['admitted'], scan['unqualified'], scan['engaged'], scan['rounds'], scan['fallbacks']),
                         (1, len(octo.UNQUALIFIED_ITEMS), ['110'], [dict(round=7, built=1, ms=12.5)], [dict(round=8, reason='dram_reserve:headroom=1')]))
        import test_octo_judge as judged

        # the coordinator renders the disabled marker through audit_log's brace format
        text = '%s round=9 failures=2 reason=x' % octo.DISABLED_MARKER
        self.assertEqual(octo_markers.draft_scan(text)['disabled'], 1)


# ---------------------------------------------------------------------------------------------
# Shipping.
# ---------------------------------------------------------------------------------------------

class ShippingTests(unittest.TestCase):
    def test_the_module_reaches_the_image_through_both_copy_lists_and_the_overlay(self):
        from test_serving_image_copy_closure import context_modules, dockerfile_modules, dockerfile_text

        overlay = (ROOT / 'docker' / 'qwen-c2-overlay.txt').read_text(encoding='utf-8')
        self.assertIn('octo_draft_tp.py', dockerfile_modules(dockerfile_text()))
        self.assertIn('octo_draft_tp.py', context_modules())
        self.assertIn('scripts/ci/octo_draft_tp.py\n', overlay)

    def test_the_cpu_suite_runs_this_file(self):
        text = CPU_WORKFLOW.read_text(encoding='utf-8')
        self.assertRegex(text, r'python -B -m unittest [^\n]*\btest_octo_draft\b')

    def test_every_new_file_is_lf(self):
        for name in ('octo_draft_tp.py', 'test_octo_draft.py'):
            with self.subTest(name=name):
                self.assertNotIn(b'\r', (HERE / name).read_bytes())

    def test_the_module_is_served_by_the_closure_scan(self):
        import test_tp4_closure_literals as closure

        self.assertIn('octo_draft_tp', closure.SERVED)


if __name__ == '__main__':
    unittest.main()
