"""QWEN_FAST_QUAD_DRAFT: one four-user, 64-row draft pass in place of the two pair passes (quad_draft.py; Q4-core,
quad-draft-plan.md; the plan's Q1 list).

  - the flags: QWEN_FAST_QUAD_DRAFT '0'/'1' only, QWEN_FAST_QUAD_SDPA fold|pairs, QWEN_FAST_QUAD_CONV
    110|80|halves, QWEN_FAST_QUAD_DRAFT_AUDIT all|N, the four required flags, the marker once;
  - the fold: folded query head 16h + 4u + j is head 4h + j with user u's rows at 0-15 of its pair's half, and GQA
    group 4 sends it to KV head 4h + u, user u's segment of head h; the unfold returns every user's rows; on a CPU
    stand-in (a GQA SDPA head by head in float32) the quad fold's rows are BIT-identical to the pair fold's and to
    each user's single-user rows, and QWEN_FAST_QUAD_SDPA=pairs gives the same bits; folded_sdpa's arguments
    stay draft_sdpa's (test_pair_row_exact pins them);
  - the pads: the quad's 12-piece K/V is byte for byte the two pair assemblies concatenated (every pad from rows
    32p of its own pair), where today's rule (every pad from row 0) differs for users 2 and 3;
  - the explicit programs: every matmul program the quad branch builds equals today's but for per_core_M (1 -> 2),
    the attention and MLP branches run today's op sequence at 64 rows, the selector program the same;
  - the readback: each user's features, candidates and scores are its pair's rows, bit for bit, and no user slice
    reads the stitch's dummy row (user_slices(4, 16, 64));
  - the conv: E1b's 110 workers (0-99 three pages, 100-109 two) and E1's 80 cover the 320 pages once, the runtime
    args carry the per-tile-row seam words, the kernel is the promoted card-B file (sha256-pinned, byte-equal to
    the probe's), the halves mode calls the served kernel per half;
  - the trace: the one bucket's placeholders (ids (1, 64), the single-user mask, rope.q / live_k (1, 1, 64, 128),
    no rope.k, the live banks), capture, replay, marker, the quad_built ledger point, and every round's host inputs
    equal to the two pairs' uploads row block for row block;
  - the coordinator: the quad engages only with all four live and packable, 3-live and 2-live rounds are today's
    pairs call for call, a failure falls back, two in a row give up (a success in between resets), a stale quad is
    retired, the first success releases the single-user captures, reads_covered and select_round cover its four
    devices, the audit replays both pair traces before the fence and compares after the selection, never the quad
    with itself;
  - the flag off (unset or '0'): each module this change touches is its PARENT (8b365868) call for call;
  - the arm, the gate, both image copy lists, the CPU allowlist, and untouched T16 / bundle-only sources.

    py -3.11 -B -m unittest test_quad_draft      (from scripts/ci)
"""

from collections import defaultdict
import hashlib
from itertools import count
import os
from pathlib import Path
import re
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

import quad_draft  # noqa: E402
from test_pair_row_exact import Device, TorchOps, coded, keep, same_bits  # noqa: E402

# This change's parent: with the flag off, every module it touches is that commit's there.
PARENT = '8b365868'
FLAG = 'QWEN_FAST_QUAD_DRAFT'
REQUIRED = {name: '1' for name in quad_draft.REQUIRED_FLAGS}
PROBE = ROOT / 'optimisation' / 'ttnn-op' / 'quad_draft_probe'
CPU_WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml'


def clean_environment(**flags):
    """No QWEN_FAST_* flag but the ones given."""
    environment = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
    environment.update(flags)
    return patch.dict(os.environ, environment, clear=True)


def parent_module(relative, name):
    from test_dflash_proposal_trace import pinned_module

    return pinned_module(relative, name, commit=PARENT)


def logged():
    """Capture every loguru line (the modules log through loguru, else print)."""
    from test_dflash_proposal_trace import logged as capture

    return capture(None)


# ---------------------------------------------------------------------------------------------
# The flags.
# ---------------------------------------------------------------------------------------------

class FlagTests(unittest.TestCase):
    def test_zero_or_one_only_and_off_by_default(self):
        self.assertFalse(quad_draft.enabled({}))
        self.assertFalse(quad_draft.enabled({FLAG: '0'}))
        self.assertTrue(quad_draft.enabled({FLAG: '1'}))
        for value in ('', 'yes', 'true', '2', ' 1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                quad_draft.enabled({FLAG: value})
        with clean_environment():
            self.assertFalse(quad_draft.enabled())

    def test_the_sub_flags_default_to_what_q0_proved(self):
        self.assertEqual(quad_draft.sdpa_mode({}), 'fold')
        self.assertEqual(quad_draft.conv_mode({}), '110')
        self.assertEqual(quad_draft.sdpa_mode({quad_draft.SDPA_FLAG: 'pairs'}), 'pairs')
        for value in ('80', 'halves', '110'):
            self.assertEqual(quad_draft.conv_mode({quad_draft.CONV_FLAG: value}), value)
        for reader, name, value in ((quad_draft.sdpa_mode, quad_draft.SDPA_FLAG, 'dense'),
                                    (quad_draft.conv_mode, quad_draft.CONV_FLAG, '64')):
            with self.subTest(name=name), self.assertRaises(ValueError):
                reader({name: value})

    def test_the_audit_is_all_or_the_first_n_quad_rounds(self):
        self.assertIsNone(quad_draft.audit_rounds({}))
        self.assertIsNone(quad_draft.audit_rounds({quad_draft.AUDIT_FLAG: ''}))
        self.assertEqual(quad_draft.audit_rounds({quad_draft.AUDIT_FLAG: 'all'}), 'all')
        self.assertEqual(quad_draft.audit_rounds({quad_draft.AUDIT_FLAG: '3'}), 3)
        for value in ('0', '-1', '03', 'x', '1.5'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                quad_draft.audit_rounds({quad_draft.AUDIT_FLAG: value})
        self.assertEqual([quad_draft.audit_selected(n, {quad_draft.AUDIT_FLAG: '2'}) for n in (1, 2, 3)],
                         [True, True, False])
        self.assertTrue(quad_draft.audit_selected(99, {quad_draft.AUDIT_FLAG: 'all'}))
        self.assertFalse(quad_draft.audit_selected(1, {}))

    def test_the_four_required_flags(self):
        self.assertEqual(quad_draft.REQUIRED_FLAGS, ('QWEN_FAST_PACKED_PROPOSAL', 'QWEN_FAST_PAIR_ROW_EXACT',
                                                     'QWEN_FAST_ROUND_B1', 'QWEN_FAST_FUSED_COMMIT_LIVE_BANKS'))
        self.assertEqual(quad_draft.missing_requirements(REQUIRED), [])
        self.assertEqual(quad_draft.missing_requirements(dict(REQUIRED, QWEN_FAST_ROUND_B1='0')), ['QWEN_FAST_ROUND_B1'])
        self.assertEqual(quad_draft.missing_requirements({}), list(quad_draft.REQUIRED_FLAGS))

    def test_the_marker_is_logged_once_per_process(self):
        lines = []
        with patch.object(quad_draft, '_NOTED', []):
            self.assertTrue(quad_draft.note([0, 1, 2, 3], 'fold', '110', log=lines.append))
            self.assertFalse(quad_draft.note([0, 1, 2, 3], 'fold', '110', log=lines.append))
        with patch.object(quad_draft, '_NOTED', []):
            quad_draft.note([0, 1, 2, 3], 'pairs', 'halves', log=lines.append)
        self.assertEqual(lines, ['[PINDIAG] quad draft engaged slots=[0,1,2,3] heads=64/16 rows=64 sdpa=fold conv=110',
                                 '[PINDIAG] quad draft engaged slots=[0,1,2,3] heads=32/8x2 rows=64 sdpa=pairs '
                                 'conv=halves'])


# ---------------------------------------------------------------------------------------------
# The fold and the K/V plan, on the CPU stand-in.
# ---------------------------------------------------------------------------------------------

class QuadFixture:
    """Four users' operands built three ways: as the quad assembles them (one 64-row live block, the 12-piece
    plan), as each pair assembles them (its own 32-row block, key_value_plan's pieces, pads from its row 0), and as
    each user's single-user trace does (its own live block with pad rows, its own pad query rows)."""

    def __init__(self, seed=0, *, scale=1.0):
        from dflash_batched_mask import key_value_plan

        generator = torch.Generator().manual_seed(seed)

        def normal(*shape, factor=1.0):
            return (torch.randn(*shape, generator=generator) * factor).bfloat16()

        self.cache = [{name: normal(1, 4, 2048, 128) for name in 'kv'} for _ in range(4)]
        self.live = [{name: normal(1, 4, 16, 128) for name in 'kv'} for _ in range(4)]
        self.live_pad = [{name: normal(1, 4, 16, 128) for name in 'kv'} for _ in range(4)]
        self.query = [normal(1, 16, 16, 128, factor=scale) for _ in range(4)]
        self.query_pad = [normal(1, 16, 16, 128, factor=scale) for _ in range(4)]
        self.block = {name: torch.cat([self.live[user][name] for user in range(4)], dim=2) for name in 'kv'}
        self.quad_query = torch.cat(self.query, dim=2)
        self.quad_keys = self.assemble(quad_draft.quad_key_value_plan(), self.block, self.cache)
        plan, _, key_rows = key_value_plan([2048, 2048], 16)
        assert key_rows == 4160
        self.pairs = []
        for pair in range(2):
            users = (2 * pair, 2 * pair + 1)
            block = {name: torch.cat([self.live[user][name] for user in users], dim=2) for name in 'kv'}
            self.pairs.append(dict(query=torch.cat([self.query[user] for user in users], dim=2),
                                   keys=self.assemble(plan, block, [self.cache[user] for user in users])))

    @staticmethod
    def assemble(plan, block, caches):
        """The branch's assembly loop on the host: a pad without a 'source' starts at row 0 (the pair's rule)."""
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
        keys = {name: torch.cat([self.cache[user][name], self.live[user][name], self.live_pad[user][name]], dim=2)
                for name in 'kv'}
        return torch.cat([self.query[user], self.query_pad[user]], dim=2), keys['k'], keys['v']


def sdpa_alone(query, keys, values, mask):
    return TorchOps().sdpa(Device(query), Device(keys), Device(values), attn_mask=Device(mask), is_causal=False,
                           scale=128 ** -0.5, program_config=None, compute_kernel_config=None,
                           memory_config='dram').value


class FoldMappingTests(unittest.TestCase):
    def test_folded_query_head_16h_4u_j_is_head_4h_j_with_user_u_first_in_its_pairs_half(self):
        ops, owned = TorchOps(), []
        query = coded(16, 64)
        folded = quad_draft.quad_fold_query(ops, Device(query), keep(owned)).value
        self.assertEqual(tuple(folded.shape), (1, 64, 32, 128))
        head_map = quad_draft.quad_head_map()
        self.assertEqual(len(head_map), 64)
        for h in range(4):
            for u in range(4):
                for j in range(4):
                    half = query[0, 4 * h + j, 32 * (u // 2):32 * (u // 2) + 32]
                    expected = half if u % 2 == 0 else torch.cat([half[16:], half[:16]])
                    self.assertTrue(torch.equal(folded[0, 16 * h + 4 * u + j], expected), (h, u, j))
                    # GQA group 64 / 16 = 4 sends it to KV head 4h + u
                    self.assertEqual((16 * h + 4 * u + j) // (64 // 16), 4 * h + u)
                    self.assertEqual(head_map[16 * h + 4 * u + j], (h, u, j, 4 * h + u))

    def test_folded_kv_head_4h_u_is_user_us_segment_of_head_h(self):
        ops, owned = TorchOps(), []
        keys = coded(4, 8320)
        folded = quad_draft.quad_fold_keys(ops, Device(keys), keep(owned)).value
        self.assertEqual(tuple(folded.shape), (1, 16, 2080, 128))
        for h in range(4):
            for u in range(4):
                self.assertTrue(torch.equal(folded[0, 4 * h + u], keys[0, h, 2080 * u:2080 * (u + 1)]))
        self.assertEqual(ops.calls, [('reshape', (1, 4, 8320, 128), (1, 16, 2080, 128))], 'a view, no copy')

    def test_the_unfold_returns_every_users_rows_to_its_block_rows(self):
        ops, owned = TorchOps(), []
        output = coded(64, 32)
        unfolded = quad_draft.quad_unfold_output(ops, Device(output), keep(owned)).value
        self.assertEqual(tuple(unfolded.shape), (1, 16, 64, 128))
        for h in range(4):
            for j in range(4):
                for u in range(4):
                    self.assertTrue(torch.equal(unfolded[0, 4 * h + j, 16 * u:16 * u + 16],
                                                output[0, 16 * h + 4 * u + j, :16]), (h, j, u))

    def test_fold_then_unfold_of_an_identity_attention_is_the_query(self):
        ops, owned = TorchOps(), []
        query = coded(16, 64)
        folded = quad_draft.quad_fold_query(ops, Device(query), keep(owned))
        self.assertTrue(torch.equal(quad_draft.quad_unfold_output(ops, folded, keep(owned)).value, query))

    def test_the_fold_is_built_from_the_pair_folds_own_functions(self):
        import pair_row_exact

        ops, owned = TorchOps(), []
        fixture = QuadFixture(0)
        with patch('pair_row_exact.fold_query', side_effect=pair_row_exact.fold_query) as fold_query, \
                patch('pair_row_exact.unfold_output', side_effect=pair_row_exact.unfold_output) as unfold, \
                patch('pair_row_exact.folded_sdpa', side_effect=pair_row_exact.folded_sdpa) as sdpa:
            quad_draft.fold_attention(ops, Device(fixture.quad_query), Device(fixture.quad_keys['k']),
                                      Device(fixture.quad_keys['v']), Device(self.mask()), keep(owned),
                                      mask_validated=True)
        self.assertEqual(fold_query.call_count, 2)
        self.assertEqual(unfold.call_count, 2)
        self.assertEqual(sdpa.call_count, 1)
        calls = [call for call in ops.calls if call[0] == 'sdpa']
        self.assertEqual(calls, [('sdpa', (1, 64, 32, 128), (1, 16, 2080, 128), (1, 1, 32, 2080))])

    @staticmethod
    def mask():
        from pair_row_exact import fold_mask

        return fold_mask((2048, 2048), 16)

    def test_the_operands_are_validated(self):
        fixture, ops, owned = QuadFixture(1), TorchOps(), []
        query, keys, values = (Device(fixture.quad_query), Device(fixture.quad_keys['k']),
                               Device(fixture.quad_keys['v']))
        mask = Device(self.mask())
        with self.assertRaises(ValueError):
            quad_draft.fold_attention(ops, query, keys, values, mask, keep(owned))
        for index, operands in enumerate([(Device(fixture.quad_query[:, :, :32]), keys, values, mask),
                                          (query, Device(fixture.quad_keys['k'][:, :, :4160]), values, mask),
                                          (query, keys, values, Device(torch.zeros(1, 1, 32, 4160))),
                                          (Device(fixture.quad_query, dtype='fp32'), keys, values, mask),
                                          (query, Device(fixture.quad_keys['k'], layout='row'), values, mask),
                                          (query, keys, values, Device(mask.value, memory='l1'))]):
            with self.subTest(case=index), self.assertRaises(ValueError):
                quad_draft.validate_quad(ops, *operands)
        self.assertEqual(ops.calls, [])


class FoldEqualsPairAndAloneTests(unittest.TestCase):
    """Every quad work unit is its pair-fold unit (and so its single-user unit): the rows are bit-identical."""

    def quad(self, fixture, attend=quad_draft.fold_attention):
        ops, owned = TorchOps(), []
        return attend(ops, Device(fixture.quad_query), Device(fixture.quad_keys['k']), Device(fixture.quad_keys['v']),
                      Device(FoldMappingTests.mask()), keep(owned), mask_validated=True).value

    def pair(self, fixture, index):
        from pair_row_exact import fold_attention

        ops, owned = TorchOps(), []
        pair = fixture.pairs[index]
        return fold_attention(ops, Device(pair['query']), Device(pair['keys']['k']), Device(pair['keys']['v']),
                              Device(FoldMappingTests.mask()), keep(owned), mask_validated=True).value

    def test_every_users_rows_equal_the_pair_fold_and_single_user_bit_for_bit(self):
        mask = FoldMappingTests.mask()
        for seed, scale in ((0, 1.0), (1, 4.0), (2, 0.25)):
            with self.subTest(seed=seed, scale=scale):
                fixture = QuadFixture(seed, scale=scale)
                quad = self.quad(fixture)
                pairs = [self.pair(fixture, index) for index in range(2)]
                for user in range(4):
                    rows = quad[:, :, 16 * user:16 * user + 16]
                    paired = pairs[user // 2][:, :, 16 * (user % 2):16 * (user % 2) + 16]
                    self.assertTrue(same_bits(rows, paired), 'user %d against its pair fold' % user)
                    alone = sdpa_alone(*fixture.single(user), mask)[:, :, :16]
                    self.assertTrue(same_bits(rows, alone), 'user %d against drafting alone' % user)

    def test_the_pairs_mode_gives_the_same_bits(self):
        fixture = QuadFixture(3)
        self.assertTrue(same_bits(self.quad(fixture, quad_draft.pairs_attention), self.quad(fixture)))

    def test_the_quad_pads_are_the_two_pair_assemblies_bytes(self):
        """R2: the quad's key axis is pair 0's assembly then pair 1's, byte for byte (C5 on the host)."""
        from dflash_batched_mask import key_value_plan

        fixture = QuadFixture(4)
        for name in 'kv':
            expected = torch.cat([fixture.pairs[0]['keys'][name], fixture.pairs[1]['keys'][name]], dim=2)
            self.assertTrue(same_bits(fixture.quad_keys[name], expected), name)
        # today's rule (every pad from row 0 of the shared block) would give users 2 and 3 user 0's rows
        today = [dict(part, source=part['source']) if part['kind'] == 'live' else
                 {key: value for key, value in part.items() if key != 'source'}
                 for part in quad_draft.quad_key_value_plan()]
        wrong = QuadFixture.assemble(today, fixture.block, fixture.cache)
        self.assertFalse(same_bits(wrong['k'], fixture.quad_keys['k']))
        self.assertTrue(same_bits(wrong['k'][:, :, :4160], fixture.quad_keys['k'][:, :, :4160]),
                        "pair 0's segments carry row 0 either way")
        plan, spans, key_rows = quad_draft.key_value_plan([2048] * 4, 16)
        self.assertEqual(key_rows, 8320)
        self.assertEqual([(span['rows'].start, span['rows'].stop) for span in spans], [(0, 16), (16, 32), (32, 48), (48, 64)])
        self.assertEqual([piece['source'].start for piece in plan if piece['kind'] == 'pad'], [0, 0, 32, 32])
        self.assertEqual(len(plan), 12)
        self.assertFalse([piece for piece in key_value_plan([2048, 2048], 16)[0] if piece['kind'] == 'pad'
                          and 'source' in piece], "the pair's pads carry no source: the branch starts them at 0")
        for contexts, rows in ((([2048] * 3), 16), ([2048] * 4, 8), ([2048, 2048, 2048, 1024], 16)):
            with self.subTest(contexts=contexts), self.assertRaises(ValueError):
                quad_draft.key_value_plan(contexts, rows)

    def test_a_hundredfold_partner_changes_nothing(self):
        mask = FoldMappingTests.mask()
        fixture = QuadFixture(5)
        quad = self.quad(fixture)
        loud = QuadFixture(5)
        # users 0 and 2 x100: their rows reach users 1 and 3 only as masked pad rows
        for user in (0, 2):
            for name in 'kv':
                for part in ('cache', 'live'):
                    getattr(loud, part)[user][name] = (getattr(loud, part)[user][name].float() * 100).bfloat16()
        loud.block = {name: torch.cat([loud.live[user][name] for user in range(4)], dim=2) for name in 'kv'}
        loud.quad_keys = QuadFixture.assemble(quad_draft.quad_key_value_plan(), loud.block, loud.cache)
        louder = self.quad(loud)
        for user in (1, 3):
            self.assertTrue(same_bits(louder[:, :, 16 * user:16 * user + 16], quad[:, :, 16 * user:16 * user + 16]))
        self.assertTrue(same_bits(louder[:, :, 16:32], sdpa_alone(*fixture.single(1), mask)[:, :, :16]))


# ---------------------------------------------------------------------------------------------
# A shape-tracking stand-in for the ttnn surface the branches use: every call logged with its shapes.
# ---------------------------------------------------------------------------------------------

ADDRESSES = count(0x10000, 16)


class Tensor:
    def __init__(self, shape, dtype='bf16', layout='tile', memory='dram'):
        self.shape, self.dtype, self.layout, self.memory = tuple(shape), dtype, layout, memory
        self.address = next(ADDRESSES)

    def memory_config(self):
        return self.memory


class Shard:
    def __init__(self, address):
        self.address = address

    def buffer_address(self):
        return self.address


class ShapeOps:
    bfloat16, float32, uint32, bfloat8_b = 'bf16', 'fp32', 'u32', 'bf8'
    TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = 'tile', 'row', 'dram'
    MathFidelity = SimpleNamespace(HiFi4='hifi4')
    Topology = SimpleNamespace(Linear='linear')

    def __init__(self):
        self.events = []
        self.experimental = SimpleNamespace(nlp_create_qkv_heads=self.create_heads, rotary_embedding_hf=self.rotary,
                                            nlp_concat_heads=self.concat_heads, all_gather_async=self.all_gather)
        self.transformer = SimpleNamespace(scaled_dot_product_attention=self.sdpa)

    def log(self, *event):
        self.events.append(event)

    def WormholeComputeKernelConfig(self, **options):
        return ('kernel', tuple(sorted(options.items())))

    def SDPAProgramConfig(self, **options):
        return ('sdpa-program', tuple(sorted(options.items())))

    def MatmulMultiCoreReuseMultiCast1DProgramConfig(self, **options):
        self.log('program', tuple(sorted(options.items())))
        return ('program', tuple(sorted(options.items())))

    def get_device_tensors(self, tensor):
        return [Shard(tensor.address), Shard(tensor.address + 1)]

    def rms_norm(self, value, **options):
        self.log('rms_norm', value.shape)
        return Tensor(value.shape)

    def matmul(self, left, right, **options):
        self.log('matmul', left.shape, right.shape, options['dtype'], options['program_config'])
        return Tensor(left.shape[:-1] + right.shape[-1:], options['dtype'])

    def linear(self, left, right, **options):
        self.log('linear', left.shape)
        return Tensor(left.shape[:-1] + (124160,))

    def typecast(self, value, dtype):
        self.log('typecast', value.shape, dtype)
        return Tensor(value.shape, dtype)

    def slice(self, value, start, end, step=None):
        self.log('slice', value.shape, tuple(start), tuple(end))
        return Tensor(tuple(high - low for low, high in zip(start, end)), value.dtype)

    def concat(self, parts, dim, memory_config=None):
        shape = list(parts[0].shape)
        shape[dim] = sum(part.shape[dim] for part in parts)
        self.log('concat', tuple(part.shape for part in parts), dim)
        return Tensor(shape, parts[0].dtype)

    def reshape(self, value, shape):
        self.log('reshape', value.shape, tuple(shape))
        return Tensor(shape, value.dtype)

    def pad(self, value, padding, fill):
        self.log('pad', value.shape, tuple(map(tuple, padding)))
        return Tensor(tuple(size + low + high for size, (low, high) in zip(value.shape, padding)), value.dtype)

    def add(self, left, right, **options):
        self.log('add', left.shape, right.shape, options.get('dtype'))
        return Tensor(left.shape, options.get('dtype', left.dtype))

    def silu(self, value, **options):
        self.log('silu', value.shape)
        return Tensor(value.shape, value.dtype)

    def multiply(self, left, right, **options):
        self.log('multiply', left.shape, right.shape)
        return Tensor(left.shape, options.get('dtype', left.dtype))

    def topk(self, value, **options):
        self.log('topk', value.shape)
        shape = value.shape[:-1] + (16,)
        return Tensor(shape), Tensor(shape, 'u16')

    def create_heads(self, query, combined, **options):
        rows = query.shape[2]
        self.log('create_heads', query.shape, combined.shape, options['num_heads'], options['num_kv_heads'])
        return (Tensor((1, options['num_heads'], rows, 128)), Tensor((1, options['num_kv_heads'], rows, 128)),
                Tensor((1, options['num_kv_heads'], rows, 128)))

    def rotary(self, value, cosine, sine, **options):
        self.log('rotary', value.shape, cosine.shape)
        return Tensor(value.shape, 'fp32')

    def concat_heads(self, value, **options):
        self.log('concat_heads', value.shape)
        return Tensor((1, 1, value.shape[2], value.shape[1] * value.shape[3]))

    def all_gather(self, value, **options):
        shape = list(value.shape)
        shape[options['dim']] *= 2
        self.log('all_gather', value.shape, options['dim'])
        return Tensor(shape, value.dtype)

    def sdpa(self, query, key, value, **options):
        self.log('sdpa', query.shape, key.shape, options['attn_mask'].shape)
        return Tensor(query.shape)


MESH = SimpleNamespace(shape=[1, 2])
COLLECTIVES = SimpleNamespace(get_and_cycle_ag_semaphore_handles=lambda: 'ag',
                              get_and_cycle_barrier_semaphore_handle=lambda: 'barrier')


def attention_parameters(ops):
    return dict(operations=ops, mesh=MESH, block_rows=16, native_head_layout=True, native_proposal_attention=True,
                kernel='kernel', norm=Tensor((1, 1, 160, 32)), convolution=Tensor((5120, 1280)),
                bases=[Tensor((1, 1, 1, 5120)) for _ in range(4)],
                projections=dict(q=Tensor((5120, 2048)), k=Tensor((5120, 512)), v=Tensor((5120, 512))),
                head_norms=dict(q=Tensor((1, 1, 4, 32)), k=Tensor((1, 1, 4, 32))), output_projection=Tensor((2048, 5120)))


def recording_convolution(ops):
    def convolve(operations, mesh, hidden, dynamic, base, **options):
        ops.log('convolve', hidden.shape, tuple(value.shape for value in dynamic), options.get('boundaries'))
        output = Tensor(hidden.shape)
        if callable(options.get('retain_temporaries')):
            options['retain_temporaries'](output)
        return output
    return convolve


class RecordingQuad(quad_draft.QuadPass):
    """QuadPass with the conv recorded (the E1b program has its own tests below)."""

    def __init__(self, ops, **options):
        super().__init__(**options)
        self.ops = ops

    def convolution(self, operations, mesh, hidden, dynamic, base, **options):
        return recording_convolution(self.ops)(operations, mesh, hidden, dynamic, base, **options)


def attention_run(module, *, quad=False, row_exact=True, sdpa='fold', users=None):
    """execute_attention_branch of `module` on ShapeOps: the packed pair (row_exact or the native call, patched) or
    the quad. Returns the event log."""
    ops = ShapeOps()
    users = users or (4 if quad else 2)
    rows = 16 * users if quad else 32
    rope = {'q': (Tensor((1, 1, rows, 128)), Tensor((1, 1, rows, 128))),
            'live_k': (Tensor((1, 1, rows, 128)), Tensor((1, 1, rows, 128)))}
    if not quad:
        rope['k'] = (Tensor((1, 1, 4160, 128)), Tensor((1, 1, 4160, 128)))
    caches = [{name: Tensor((1, 4, 2048, 128)) for name in 'kv'} for _ in range(users)]
    pack = [dict(position=4096 + 1000 * user, history_rows=2048) for user in range(users)]
    mask = Tensor((1, 1, 32, 2080 if (quad or row_exact) else 4160))
    extra = {}
    if quad:
        extra['quad'] = RecordingQuad(ops, sdpa=sdpa)
        convolve = extra['quad'].convolution
    else:
        convolve = recording_convolution(ops)
        if row_exact:
            extra['row_exact'] = True
    native = Mock(side_effect=lambda operations, q, k, v, m, **options: ops.log('native', q.shape, k.shape) or Tensor(q.shape))
    owned = []
    with patch('dflash_t16_native_scope.require_active', return_value=None), \
            patch('feature_collective.projection_links', return_value=1), \
            patch('projection_link_policy.projection_links', return_value=1), \
            patch('dflash_t16_native_attention.attention', native):
        output = module.execute_attention_branch(ops, MESH, COLLECTIVES, Tensor((1, 1, rows, 5120)), None, mask, rope,
            lambda value: owned.append(value) or value, parameters=attention_parameters(ops), context=None,
            pack=pack, convolution_operation=convolve, cached_history=caches, native_proposal_mask_validated=True,
            **extra)
    ops.log('output', output.shape)
    return ops.events


def mlp_parameters(ops, weights, convolution):
    return dict(operations=ops, mesh=MESH, source_weights=weights, source_convolution=convolution, kernel='kernel',
                device_norm=Tensor((1, 1, 160, 32)), device_conv=Tensor((5120, 1280)),
                bases=[Tensor((1, 1, 1, 5120)) for _ in range(4)],
                device_projections=[Tensor((5120, 8704)), Tensor((5120, 8704)), Tensor((8704, 5120))],
                shards='shards', norm_weight='norm', conv_weight='conv', base_weight='base')


def mlp_run(module, *, quad=False, trace_safe=True, boundaries=True):
    ops = ShapeOps()
    rows = 64 if quad else 32
    weights, convolution = object(), object()
    seams = tuple((start, start + 16) for start in range(0, rows, 16)) if boundaries else None
    extra = dict(quad=RecordingQuad(ops)) if quad else {}
    convolve = extra['quad'].convolution if quad else recording_convolution(ops)
    owned = []
    with patch('feature_collective.projection_links', return_value=1), \
            patch('projection_link_policy.projection_links', return_value=1):
        output = module.execute_mlp_branch(ops, MESH, COLLECTIVES, Tensor((1, 1, rows, 5120)), weights, convolution,
            lambda value: owned.append(value) or value, parameters=mlp_parameters(ops, weights, convolution),
            trace_safe=trace_safe, convolution_operation=convolve,
            **(dict(boundaries=seams) if seams is not None else {}), **extra)
    ops.log('output', output['output'].shape)
    return ops.events


def without_per_core_m(program):
    return tuple(item for item in program[1] if item[0] != 'per_core_M') if program[0] == 'program' else program


def per_core_m(program):
    return dict(program[1])['per_core_M']


CORE_OPS = ('rms_norm', 'matmul', 'typecast', 'create_heads', 'rotary', 'concat_heads', 'all_gather', 'add',
            'convolve', 'program', 'silu', 'multiply', 'output')


class BranchTests(unittest.TestCase):
    def test_the_attention_branch_is_todays_at_64_rows_with_per_core_m_2(self):
        import draft_attention_branch

        pair = attention_run(draft_attention_branch)
        quad = attention_run(draft_attention_branch, quad=True)
        names = lambda events: [event[0] for event in events if event[0] in CORE_OPS]
        self.assertEqual(names(quad), names(pair))
        programs = [event[1] for event in pair if event[0] == 'program']
        quad_programs = [event[1] for event in quad if event[0] == 'program']
        # conv-kernel (8,5), q (8,8), k and v (8,8, one program), o (8,10)
        self.assertEqual(len(programs), 4)
        self.assertEqual([dict(item)['compute_with_storage_grid_size'] for item in programs],
                         [(8, 5), (8, 8), (8, 8), (8, 10)])
        self.assertEqual([item for item in quad_programs if dict(item)['per_core_M'] != 2], [])
        self.assertEqual([item for item in programs if dict(item)['per_core_M'] != 1], [])
        strip = lambda items: [tuple(pair for pair in item if pair[0] != 'per_core_M') for item in items]
        self.assertEqual(strip(quad_programs), strip(programs), 'every other program field is today')
        for event in quad:
            if event[0] in ('matmul', 'rms_norm', 'typecast') and len(event[1]) == 4 and event[1][:2] == (1, 1):
                self.assertIn(event[1][2], (64,), event)
        heads = [event for event in quad if event[0] == 'create_heads']
        self.assertEqual(heads, [('create_heads', (1, 1, 64, 2048), (1, 1, 64, 1024), 16, 4)], 'no query pad at 64')
        self.assertEqual([event for event in quad if event[0] == 'concat_heads'], [('concat_heads', (1, 16, 64, 128))])
        self.assertEqual([event for event in quad if event[0] == 'all_gather'], [('all_gather', (1, 1, 64, 5120), 0)])
        self.assertEqual([event for event in quad if event[0] == 'sdpa'],
                         [('sdpa', (1, 64, 32, 128), (1, 16, 2080, 128), (1, 1, 32, 2080))])
        self.assertEqual([event[3] for event in quad if event[0] == 'convolve'],
                         [((0, 16), (16, 32), (32, 48), (48, 64))] * 2)
        self.assertEqual(quad[-1], ('output', (1, 1, 64, 5120)))

    def test_the_kv_assembly_is_twelve_pieces_with_pair_pads(self):
        import draft_attention_branch

        quad = attention_run(draft_attention_branch, quad=True)
        assembly = [event for event in quad if event[0] == 'concat' and len(event[1]) == 12]
        self.assertEqual(len(assembly), 2, 'one concat each for k and v')
        self.assertEqual(assembly[0][1], ((1, 4, 2048, 128), (1, 4, 16, 128), (1, 4, 16, 128)) * 4)
        live = [event for event in quad if event[0] == 'slice' and event[1] == (1, 4, 64, 128)]
        # per k and v: user u's live rows [16u, 16u + 16), then its pad [32p, 32p + 16)
        starts = [event[2][2] for event in live]
        self.assertEqual(starts, [0, 0, 16, 0, 32, 32, 48, 32] * 2)

    def test_the_pairs_sdpa_mode_runs_two_pair_folds(self):
        import draft_attention_branch

        events = attention_run(draft_attention_branch, quad=True, sdpa='pairs')
        self.assertEqual([event for event in events if event[0] == 'sdpa'],
                         [('sdpa', (1, 32, 32, 128), (1, 8, 2080, 128), (1, 1, 32, 2080))] * 2)

    def test_the_mlp_branch_is_todays_at_64_rows_with_per_core_m_2(self):
        import draft_mlp_branch

        pair, quad = mlp_run(draft_mlp_branch), mlp_run(draft_mlp_branch, quad=True)
        self.assertEqual([event[0] for event in quad], [event[0] for event in pair], 'op for op')
        programs = [event[1] for event in pair if event[0] == 'program']
        quad_programs = [event[1] for event in quad if event[0] == 'program']
        self.assertEqual(len(programs), 4, 'conv-kernel, gate, up, down')
        self.assertEqual([dict(item)['per_core_M'] for item in programs], [1] * 4)
        self.assertEqual([dict(item)['per_core_M'] for item in quad_programs], [2] * 4)
        strip = lambda items: [tuple(pair for pair in item if pair[0] != 'per_core_M') for item in items]
        self.assertEqual(strip(quad_programs), strip(programs))
        rows = lambda events: {event[1][2] for event in events if event[0] in ('matmul', 'rms_norm', 'silu', 'multiply')}
        self.assertEqual(rows(pair), {32})
        self.assertEqual(rows(quad), {64})
        self.assertEqual([event for event in quad if event[0] == 'all_gather'], [('all_gather', (1, 1, 64, 5120), 0)])

    def test_the_quad_is_refused_off_its_path(self):
        import draft_attention_branch
        import draft_mlp_branch

        with self.assertRaises(ValueError):
            attention_run(draft_attention_branch, quad=True, users=2)
        with self.assertRaises(ValueError):
            mlp_run(draft_mlp_branch, quad=True, trace_safe=False)
        with self.assertRaises(ValueError):
            mlp_run(draft_mlp_branch, quad=True, boundaries=False)

    def test_the_64_row_helpers_are_the_bundle_helpers_call_for_call(self):
        """project_key_value, the head split, the head concat and the gather-add at 64 rows make the calls the
        bundle-only helpers make at 32, rows aside."""
        import draft_head_layout
        import draft_kv_projection
        import feature_collective

        def run(project, split, concat, gather, rows):
            ops = ShapeOps()
            owned = []
            retain = lambda value: owned.append(value) or value
            parameters = dict(operations=ops, native_head_layout=True, kernel='kernel',
                              projections=dict(k=Tensor((5120, 512)), v=Tensor((5120, 512))),
                              head_norms=dict(k=Tensor((1, 1, 4, 32))))
            with patch('feature_collective.projection_links', return_value=1), \
                    patch('projection_link_policy.projection_links', return_value=1):
                project(ops, Tensor((1, 1, rows, 5120)), Tensor((1, 1, rows, 2048)),
                        (Tensor((1, 1, rows, 128)), Tensor((1, 1, rows, 128))), retain, parameters=parameters)
                split(ops, Tensor((1, 1, rows, 2048)), Tensor((1, 1, rows, 512)), Tensor((1, 1, rows, 512)), retain)
                concat(ops, Tensor((1, 16, rows, 128)), retain)
                ops.log('gathered', gather(ops, MESH, COLLECTIVES, Tensor((1, 1, rows, 5120), 'fp32'),
                                           retain_temporaries=retain).shape)
            return ops.events

        today = run(draft_kv_projection.project_key_value, draft_head_layout.split_projected_heads,
                    draft_head_layout.concatenate_query_heads, feature_collective.gather_add_projection, 32)
        quad = run(quad_draft.project_key_value, quad_draft.split_projected_heads,
                   quad_draft.concatenate_query_heads, quad_draft.gather_add_projection, 64)
        self.assertGreater(len(today), 15)
        self.assertEqual(quad, [widen(event) for event in today])


def widen(value):
    """A 32-row event as the same call at 64 rows: every 4-d shape's row axis 32 -> 64, per_core_M 1 -> 2."""
    if isinstance(value, tuple):
        if len(value) == 2 and value[0] == 'per_core_M':
            return ('per_core_M', 2 * value[1])
        if len(value) == 4 and all(type(item) is int for item in value) and value[2] == 32:
            return value[:2] + (64,) + value[3:]
        return tuple(widen(item) for item in value)
    return value


# ---------------------------------------------------------------------------------------------
# execute_proposal under the quad.
# ---------------------------------------------------------------------------------------------

class ExecuteProposalTests(unittest.TestCase):
    def fixture(self):
        from test_dflash_device_audit import ExecuteProposalTests as Audit

        return Audit()

    def pack(self, users=4):
        return [dict(position=4096 + 100 * user, history_rows=2048) for user in range(users)]

    def run_quad(self, **overrides):
        import dflash_device

        audit = self.fixture()
        device = audit.device()
        device.kv_history, device.native_proposal_attention, device.fused_convolution = object(), True, True
        quad = Mock(spec=quad_draft.QuadPass, rows=64)
        quad.head_candidates = Mock(return_value=['quad-chunks'])
        extra = dict(pack=self.pack(), cached_history=[[object()] * 5] * 4, quad=quad)
        extra.update(overrides)
        with patch('dflash_device.addresses', return_value=('a', 'b')):
            device.validated_native_proposal_masks.add(('a', 'b'))
            outputs, attention, mlp = audit.run_proposal(device, context=None, **extra)
        return device, quad, outputs, attention, mlp

    def test_the_quad_reaches_both_branches_the_selector_and_the_head(self):
        device, quad, outputs, attention, mlp = self.run_quad()
        self.assertEqual([call.kwargs.get('quad') for call in attention.call_args_list], [quad] * 5)
        self.assertEqual([call.kwargs.get('quad') for call in mlp.call_args_list], [quad] * 5)
        self.assertEqual({call.kwargs['convolution_operation'] for call in attention.call_args_list}, {quad.convolution})
        self.assertEqual([call.kwargs['boundaries'] for call in mlp.call_args_list],
                         [((0, 16), (16, 32), (32, 48), (48, 64))] * 5)
        operations = device.operations
        operations.reshape.assert_called_once()
        self.assertEqual(operations.reshape.call_args.args[1], (1, 1, 64, 2560))
        operations.pad.assert_not_called()
        program = operations.MatmulMultiCoreReuseMultiCast1DProgramConfig.call_args.kwargs
        self.assertEqual(program['per_core_M'], 2)
        self.assertEqual(program['compute_with_storage_grid_size'], (8, 1))
        audit = self.fixture()
        pair = audit.device()
        pair.kv_history, pair.native_proposal_attention = object(), True
        with patch('dflash_device.addresses', return_value=('a', 'b')):
            pair.validated_native_proposal_masks.add(('a', 'b'))
            audit.run_proposal(pair, context=None, pack=self.pack(2), cached_history=[[object()] * 5] * 2)
        today = pair.operations.MatmulMultiCoreReuseMultiCast1DProgramConfig.call_args.kwargs
        self.assertEqual(today['per_core_M'], 1)
        self.assertEqual({key: value for key, value in program.items() if key != 'per_core_M'},
                         {key: value for key, value in today.items() if key != 'per_core_M'},
                         "the selector program is today's but for per_core_M")
        quad.head_candidates.assert_called_once()
        self.assertEqual(outputs.chunks, ['quad-chunks'])
        operations.slice.assert_not_called()

    def test_the_quad_is_refused_off_its_path(self):
        for overrides in (dict(pack=self.pack(2), cached_history=[[object()] * 5] * 2), dict(row_exact=True),
                          dict(cached_history=None), dict(audit_convolution=True)):
            with self.subTest(overrides=list(overrides)), self.assertRaises(ValueError):
                self.run_quad(**overrides)

    def test_without_the_fused_conv_the_quad_is_refused(self):
        audit = self.fixture()
        device = audit.device()
        device.kv_history, device.native_proposal_attention = object(), True
        with patch('dflash_device.addresses', return_value=('a', 'b')), self.assertRaises(ValueError):
            device.validated_native_proposal_masks.add(('a', 'b'))
            audit.run_proposal(device, context=None, pack=self.pack(), cached_history=[[object()] * 5] * 4,
                               quad=Mock(rows=64))

    def test_without_it_no_branch_sees_the_keyword(self):
        audit = self.fixture()
        device = audit.device()
        device.kv_history, device.native_proposal_attention = object(), True
        with patch('dflash_device.addresses', return_value=('a', 'b')):
            device.validated_native_proposal_masks.add(('a', 'b'))
            _, attention, mlp = audit.run_proposal(device, context=None, pack=self.pack(2),
                                                   cached_history=[[object()] * 5] * 2)
        self.assertTrue(all('quad' not in call.kwargs for call in [*attention.call_args_list, *mlp.call_args_list]))

    def test_head_candidates_are_todays_per_half_then_concatenated(self):
        ops = ShapeOps()
        owned = []
        model = SimpleNamespace(num_devices=2, vocab_size=248320, _lmhead_vocab_sharded=True, lm_head_weight='head')
        calls = []

        def head(operations, model_, block, owned_):
            calls.append(block.shape)
            return [dict(start=start, stop=stop, values=Tensor((1, 1, 32, 16)), indices=Tensor((1, 1, 32, 16), 'u16'))
                    for start, stop in ((0, 32768), (32768, 65536), (65536, 98304), (98304, 124160))]

        with patch('draft_shared_head.shared_head_candidates', side_effect=head):
            chunks = quad_draft.head_candidates(ops, model, Tensor((1, 1, 64, 5120)), owned,
                                                lambda value: owned.append(value) or value)
        self.assertEqual(calls, [(1, 1, 32, 5120)] * 2)
        slices = [event for event in ops.events if event[0] == 'slice']
        self.assertEqual(slices, [('slice', (1, 1, 64, 5120), (0, 0, 0, 0), (1, 1, 32, 5120)),
                                  ('slice', (1, 1, 64, 5120), (0, 0, 32, 0), (1, 1, 64, 5120))])
        concats = [event for event in ops.events if event[0] == 'concat']
        self.assertEqual(concats, [('concat', ((1, 1, 32, 16), (1, 1, 32, 16)), 2)] * 8, 'values then indices, per chunk')
        self.assertEqual([(chunk['start'], chunk['stop'], chunk['values'].shape, chunk['indices'].dtype)
                          for chunk in chunks],
                         [(0, 32768, (1, 1, 64, 16), 'u16'), (32768, 65536, (1, 1, 64, 16), 'u16'),
                          (65536, 98304, (1, 1, 64, 16), 'u16'), (98304, 124160, (1, 1, 64, 16), 'u16')])


# ---------------------------------------------------------------------------------------------
# The readback.
# ---------------------------------------------------------------------------------------------

class HostTensor:
    def __init__(self, values):
        self.values = values


class HostOps:
    """get_device_tensors / to_torch over per-chip host values."""

    def get_device_tensors(self, tensor):
        return [HostTensor(value) for value in tensor.chips]

    def to_torch(self, shard):
        return shard.values.clone()


def chips(first, second):
    return SimpleNamespace(chips=[first, second])


def pair_outputs(generator):
    from draft_shared_head import candidate_chunks

    chunks = []
    for start, stop in candidate_chunks():
        values, indices = [], []
        for _ in range(2):
            values.append(torch.randn(1, 1, 32, 16, generator=generator).bfloat16())
            rows = [torch.randperm(stop - start, generator=generator)[:16] for _ in range(32)]
            indices.append(torch.stack(rows).reshape(1, 1, 32, 16).to(torch.int32))
        chunks.append(dict(start=start, stop=stop, values=values, indices=indices))
    projected = torch.randn(1, 1, 32, 256, generator=generator).bfloat16()
    return chunks, projected


class ReadbackTests(unittest.TestCase):
    def test_every_users_parts_are_its_pairs_bit_for_bit(self):
        from dflash_packed_proposal import read_device_outputs

        generator = torch.Generator().manual_seed(7)
        pairs = [pair_outputs(generator) for _ in range(2)]
        device = SimpleNamespace(operations=HostOps())
        expected = []
        for chunks, projected in pairs:
            outputs = SimpleNamespace(chunks=[dict(start=chunk['start'], stop=chunk['stop'],
                                                   values=chips(*chunk['values']), indices=chips(*chunk['indices']))
                                              for chunk in chunks], projected=chips(projected, projected.clone()))
            expected.extend(read_device_outputs(device, outputs, 2, 16))
        quad_chunks = []
        for number in range(4):
            first, second = pairs[0][0][number], pairs[1][0][number]
            quad_chunks.append(dict(start=first['start'], stop=first['stop'],
                                    values=chips(*(torch.cat([first['values'][chip], second['values'][chip]], dim=2)
                                                   for chip in range(2))),
                                    indices=chips(*(torch.cat([first['indices'][chip], second['indices'][chip]], dim=2)
                                                    for chip in range(2)))))
        projected = torch.cat([pairs[0][1], pairs[1][1]], dim=2)
        parts = quad_draft.read_quad_outputs(device, SimpleNamespace(chunks=quad_chunks,
                                                                     projected=chips(projected, projected.clone())))
        self.assertEqual(len(parts), 4)
        for user in range(4):
            for key in ('hidden', 'candidates', 'unary'):
                with self.subTest(user=user, key=key):
                    mine, theirs = parts[user][key], expected[user][key]
                    self.assertEqual(tuple(mine.shape), tuple(theirs.shape))
                    self.assertTrue(torch.equal(mine, theirs) and mine.dtype == theirs.dtype)

    def test_no_user_slice_reads_the_stitchs_dummy_row_or_an_anchor(self):
        from dflash_packed_proposal import user_slices

        slices = user_slices(4, 16, block_width=64)
        drafts = {index for part in slices for index in range(part['drafts'].start, part['drafts'].stop)}
        selector = {index for part in slices for index in range(part['selector'].start, part['selector'].stop)}
        self.assertNotIn(31, drafts, 'merged index 31 is the dummy (block row 32, user 2 anchor)')
        self.assertEqual(sorted(drafts), [index for index in range(63) if index not in (15, 31, 47)])
        self.assertTrue({0, 16, 32, 48}.isdisjoint(selector))
        self.assertEqual([part['drafts'] for part in slices], [slice(0, 15), slice(16, 31), slice(32, 47), slice(48, 63)])

    def test_replicated_features_that_differ_are_refused(self):
        generator = torch.Generator().manual_seed(8)
        chunks, projected = pair_outputs(generator)
        quad_chunks = [dict(start=chunk['start'], stop=chunk['stop'],
                            values=chips(*(torch.cat([value, value], dim=2) for value in chunk['values'])),
                            indices=chips(*(torch.cat([value, value], dim=2) for value in chunk['indices'])))
                       for chunk in chunks]
        wide = torch.cat([projected, projected], dim=2)
        other = wide.clone()
        other[0, 0, 5, 3] += 1
        with self.assertRaises(AssertionError):
            quad_draft.read_quad_outputs(SimpleNamespace(operations=HostOps()),
                                         SimpleNamespace(chunks=quad_chunks, projected=chips(wide, other)))


# ---------------------------------------------------------------------------------------------
# The conv.
# ---------------------------------------------------------------------------------------------

class DescriptorOps:
    bfloat16, TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'bf16', 'tile', 'dram'
    MathFidelity = SimpleNamespace(HiFi4='hifi4')
    DataMovementProcessor = SimpleNamespace(RISCV_0='riscv0')
    NOC = SimpleNamespace(RISCV_0_default='noc0')

    def __init__(self):
        self.calls, self.freed = [], []

    CoreCoord = staticmethod(lambda x, y: (x, y))
    CoreRange = staticmethod(lambda start, end: (start, end))
    CoreRangeSet = staticmethod(lambda ranges: tuple(ranges))
    Tile = staticmethod(lambda shape: tuple(shape))
    TileDescriptor = staticmethod(lambda tile: ('tile', tile))
    CBFormatDescriptor = staticmethod(lambda **options: options)
    CBDescriptor = staticmethod(lambda **options: options)
    KernelDescriptor = staticmethod(lambda **options: options)
    ComputeConfigDescriptor = staticmethod(lambda **options: options)
    DataMovementConfigDescriptor = staticmethod(lambda **options: options)
    MeshCoordinate = staticmethod(lambda row, column: (row, column))
    MeshCoordinateRange = staticmethod(lambda start, end: (start, end))
    ProgramDescriptor = staticmethod(lambda **options: options)
    MeshProgramDescriptor = staticmethod(dict)

    def RuntimeArgs(self):
        return defaultdict(dict)

    def TensorAccessorArgs(self, shard):
        return SimpleNamespace(get_compile_time_args=lambda: [shard.address % 97])

    def empty(self, shape, **options):
        return Tensor(shape)

    def get_device_tensors(self, tensor):
        return [Shard(tensor.address), Shard(tensor.address + 1)]

    def generic_op(self, tensors, program):
        self.calls.append((tensors, program))

    def deallocate(self, tensor):
        self.freed.append(tensor)


def conv_operands(rows=64):
    return Tensor((1, 1, rows, 5120)), [Tensor((1, 1, rows, 320)) for _ in range(2)], [Tensor((1, 1, 1, 5120))
                                                                                         for _ in range(2)]


QUAD_SEAMS = ((0, 16), (16, 32), (32, 48), (48, 64))


class ConvTests(unittest.TestCase):
    def test_the_promoted_kernel_is_the_probed_bytes_and_pinned(self):
        served = (HERE / quad_draft.CONV_KERNEL).read_bytes()
        probed = (PROBE / 'quad_conv_io.cpp').read_bytes()
        self.assertEqual(served, probed)
        self.assertEqual(hashlib.sha256(served).hexdigest(), quad_draft.CONV_KERNEL_SHA256)
        self.assertEqual(quad_draft.CONV_KERNEL_SHA256, '8c4e8f93f8ced3a3c8ef8f551b8f711e5fdc03e7bcd0aeeade3e15ee94e0492b')
        self.assertNotIn(b'\r', served)
        with patch.object(quad_draft, '_KERNEL_CHECKED', []), \
                patch.object(quad_draft, 'CONV_KERNEL_SHA256', '0' * 64), self.assertRaises(ValueError):
            quad_draft.conv_kernel_path()
        with patch.object(quad_draft, '_KERNEL_CHECKED', []):
            self.assertEqual(quad_draft.conv_kernel_path(), HERE / quad_draft.CONV_KERNEL)

    def test_the_seam_words_are_per_tile_row(self):
        self.assertEqual(quad_draft.seam_words(QUAD_SEAMS, 64), (0x00010001, 0x00010001))
        self.assertEqual(quad_draft.seam_words(((0, 16), (16, 32)), 32), (0x00010001, 0))
        for spans in (((0, 16), (16, 48), (48, 64)), ((0, 64),)):
            with self.subTest(spans=spans), self.assertRaises(ValueError):
                quad_draft.seam_words(spans, 64)

    def test_every_variant_covers_the_320_pages_once(self):
        for workers, counts in ((110, {3: 100, 2: 10}), (80, {4: 80})):
            with self.subTest(workers=workers):
                pages = quad_draft.conv_pages(workers)
                self.assertEqual(sorted(page for owned in pages.values() for page in owned), list(range(320)))
                tally = {}
                for owned in pages.values():
                    tally[len(owned)] = tally.get(len(owned), 0) + 1
                self.assertEqual(tally, counts)

    def program(self, conv):
        ops = DescriptorOps()
        hidden, dynamic, base = conv_operands()
        with patch.object(quad_draft, '_KERNEL_CHECKED', []):
            output = quad_draft.quad_fused_convolution(ops, MESH, hidden, dynamic, base, boundaries=QUAD_SEAMS,
                                                       conv=conv)
        self.assertEqual(len(ops.calls), 1)
        tensors, program = ops.calls[0]
        self.assertIs(tensors[-1], output)
        self.assertEqual(tensors[:-1], [hidden, *dynamic, *base])
        return tensors, program

    def test_e1b_runs_110_workers_on_the_11_by_10_grid(self):
        tensors, program = self.program('110')
        self.assertEqual(sorted(program), [((0, 0), (0, 0)), ((0, 1), (0, 1))], 'one program per chip')
        for chip in range(2):
            descriptor = program[((0, chip), (0, chip))]
            reader, *computes = descriptor['kernels']
            self.assertTrue(reader['kernel_source'].endswith('quad_conv_io.cpp'))
            self.assertEqual(reader['core_ranges'], tuple(((x, 0), (x, 9)) for x in range(11)))
            self.assertEqual([(kernel['compile_time_args'], kernel['core_ranges']) for kernel in computes],
                             [([2], (((10, 0), (10, 9)),)), ([3], tuple(((x, 0), (x, 9)) for x in range(10)))])
            self.assertTrue(all(kernel['kernel_source'].endswith('draft_convolution_fused_compute.cpp')
                                for kernel in computes))
            self.assertEqual([buffer['total_size'] for buffer in descriptor['cbs']], [7 * 2048, 2 * 2048, 2048])
            addresses = [tensor.address + chip for tensor in tensors]
            for worker in range(110):
                x, y = worker // 10, worker % 10
                self.assertEqual(reader['runtime_args'][x][y], addresses + [64, worker, 110, 0x10001, 0x10001])

    def test_e1_runs_the_served_80_workers(self):
        tensors, program = self.program('80')
        reader, *computes = program[((0, 0), (0, 0))]['kernels']
        self.assertEqual(reader['core_ranges'], tuple(((x, 0), (x, 9)) for x in range(8)))
        self.assertEqual([kernel['compile_time_args'] for kernel in computes], [[4]])
        for worker in range(80):
            self.assertEqual(reader['runtime_args'][worker % 8][worker // 8][-5:], [64, worker, 80, 0x10001, 0x10001])

    def test_the_conv_refuses_what_the_kernel_cannot_serve(self):
        ops = DescriptorOps()
        hidden, dynamic, base = conv_operands()
        with self.assertRaises(ValueError):
            quad_draft.quad_fused_convolution(ops, MESH, *conv_operands(32), boundaries=((0, 16), (16, 32)))
        with self.assertRaises(ValueError):
            quad_draft.quad_fused_convolution(ops, MESH, hidden, dynamic, base, boundaries=((0, 64),))
        with self.assertRaises(ValueError):
            quad_draft.quad_fused_convolution(ops, MESH, hidden, dynamic, base, boundaries=QUAD_SEAMS, conv='halves')
        with self.assertRaises(ValueError):
            quad_draft.quad_fused_convolution(ops, MESH, Tensor((1, 1, 64, 5120), 'fp32'), dynamic, base,
                                              boundaries=QUAD_SEAMS)
        self.assertEqual(ops.calls, [])

    def test_the_halves_mode_calls_the_served_kernel_per_half(self):
        ops = ShapeOps()
        owned = []
        calls = []

        def served(operations, mesh, hidden, dynamic, base, *, boundaries=None):
            calls.append((hidden.shape, tuple(value.shape for value in dynamic), boundaries))
            return Tensor(hidden.shape)

        with patch('draft_convolution_fused.fused_convolution', side_effect=served):
            output = quad_draft.QuadPass(conv='halves').convolution(ops, MESH, *conv_operands(), fp32_intermediates=True,
                retain_temporaries=lambda value: owned.append(value) or value, boundaries=QUAD_SEAMS)
        self.assertEqual(calls, [((1, 1, 32, 5120), ((1, 1, 32, 320),) * 2, ((0, 16), (16, 32)))] * 2)
        self.assertEqual(output.shape, (1, 1, 64, 5120))
        self.assertIs(owned[-1], output)

    def test_the_conv_operation_keeps_checked_convolutions_contract(self):
        quad = quad_draft.QuadPass()
        for options in (dict(fp32_intermediates=False, retain_temporaries=lambda value: value, boundaries=QUAD_SEAMS),
                        dict(fp32_intermediates=True, retain_temporaries=None, boundaries=QUAD_SEAMS),
                        dict(fp32_intermediates=True, retain_temporaries=lambda value: value)):
            with self.subTest(options=sorted(options)), self.assertRaises(ValueError):
                quad.convolution(DescriptorOps(), MESH, *conv_operands(), **options)

    def test_the_grid_check(self):
        mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))
        self.assertTrue(quad_draft.grid_fits(mesh, '110'))
        small = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=8, y=10))
        self.assertFalse(quad_draft.grid_fits(small, '110'))
        self.assertTrue(quad_draft.grid_fits(small, '80'))
        self.assertTrue(quad_draft.grid_fits(small, 'halves'))
        self.assertIsNone(quad_draft.grid_fits(object(), '110'))


# ---------------------------------------------------------------------------------------------
# The trace.
# ---------------------------------------------------------------------------------------------

def quad_devices(ops, *, context=2048):
    """Four devices on one mesh, pooled slots 0-3, live banks = their kv_history.active (F4), each bank a device
    tensor of its own."""
    from test_dflash_proposal_trace import FakeTensor, pair_devices

    made = [*pair_devices(ops, context=context), *pair_devices(ops, context=context)]
    mesh = made[0].mesh
    for index, device in enumerate(made):
        device.mesh = mesh
        device.pool_slot = SimpleNamespace(index=index)
        device.position = 4096 + 1000 * index + 7
        device.kv_history.active = [{name: FakeTensor(torch.zeros(1), 'bf16', 'tile', True, ops.addresses)
                                     for name in ('k', 'v')} for _ in range(5)]
    return made


def live_banks(*devices):
    return [[{name: layer[name] for name in ('k', 'v')} for layer in device.kv_history.active] for device in devices]


class TraceTests(unittest.TestCase):
    def setUp(self):
        import dflash_proposal_trace
        import fused_commit

        environment = clean_environment(**{FLAG: '1'}, **REQUIRED)
        environment.start()
        self.addCleanup(environment.stop)
        for target, value in ((quad_draft, '_NOTED'), (fused_commit, '_LIVE_NOTED'),
                              (dflash_proposal_trace, '_PAIR_MASK_REFRESH_NOTED')):
            patcher = patch.object(target, value, [])
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch('fused_commit.live_bank_history', side_effect=live_banks)
        patcher.start()
        self.addCleanup(patcher.stop)

    def build(self, **options):
        from test_dflash_proposal_trace import RecordingOps

        ops = RecordingOps()
        devices = quad_devices(ops)
        trace = quad_draft.PreparedQuadDFlashProposal(devices, **options)
        return ops, devices, trace

    def test_the_bucket_placeholders_capture_replay_marker_and_ledger_point(self):
        ops, devices, trace = self.build()
        lines, loguru = logged()
        with loguru, patch('memory_ledger.record') as ledger:
            self.assertTrue(trace.prepare_device((11, 22, 33, 44)))
        uploads = [event for event in ops.events if event[0] == 'from_torch' and event[2] is True]
        shapes = [event[1][1] for event in uploads]
        self.assertEqual(shapes, [(1, 64), (1, 1, 32, 2080), (1, 1, 64, 128), (1, 1, 64, 128), (1, 1, 64, 128),
                                  (1, 1, 64, 128)])
        bucket = trace.buckets[(2048,) * 4]
        self.assertEqual(set(bucket.rope), {'q', 'live_k'}, 'no rope.k: nothing reads it')
        from pair_row_exact import fold_mask

        self.assertTrue(same_bits(bucket.host_mask, fold_mask((2048, 2048), 16)))
        self.assertEqual(bucket.cached_history, live_banks(*devices))
        kinds = [event[0] for event in ops.events]
        self.assertEqual(kinds.count('begin_trace_capture'), 1)
        self.assertEqual(kinds.count('end_trace_capture'), 1)
        replays = [event for event in ops.events if event[0] == 'execute_trace']
        self.assertEqual([event[3] for event in replays], [True, False], 'the build replay, then the round')
        execute = devices[0].execute_proposal
        self.assertEqual(execute.call_count, 2, 'the eager warm-up and the capture')
        call = execute.call_args
        self.assertEqual(call.kwargs['pack'], [dict(position=device.position, history_rows=2048) for device in devices])
        self.assertIsInstance(call.kwargs['quad'], quad_draft.QuadPass)
        self.assertEqual((call.kwargs['quad'].sdpa, call.kwargs['quad'].conv), ('fold', '110'))
        self.assertEqual(set(call.args[3]), {'q', 'live_k'})
        self.assertIsNone(call.args[1], 'no history tensor on the cached path')
        self.assertIs(call.kwargs['cached_history'], bucket.cached_history)
        for device in devices[1:]:
            device.execute_proposal.assert_not_called()
        self.assertEqual([line for line in lines if line.startswith(quad_draft.MARKER)],
                         ['[PINDIAG] quad draft engaged slots=[0,1,2,3] heads=64/16 rows=64 sdpa=fold conv=110'])
        self.assertEqual([line for line in lines if 'pair live banks engaged' in line],
                         ['[PINDIAG] pair live banks engaged pair=[0, 1, 2, 3] context=(2048, 2048, 2048, 2048)'])
        ledger.assert_called_once()
        self.assertEqual(ledger.call_args.args, ('quad_built',))
        self.assertEqual(ledger.call_args.kwargs['point'], 'slots=0,1,2,3')
        self.assertEqual(set(ledger.call_args.kwargs), {'point', 'quad_placeholders', 'quad_intermediates'},
                         "the quad's own buffers, never the devices it reads")
        self.assertEqual(len(ledger.call_args.kwargs['quad_placeholders']), 6)
        self.assertTrue(trace.last_built)
        self.assertEqual(len(devices[0].validated_native_proposal_masks), 1)

    def test_every_rounds_host_inputs_are_the_two_pairs_uploads(self):
        """Rows [32p, 32p + 32) of the quad's ids, rope.q and live_k are pair p's, and the mask is the pair fold's,
        bit for bit, under the pair's own QWEN_FAST_ROUND_B1 build."""
        import dflash_proposal_trace

        ops, devices, trace = self.build()
        seeds = (101, 202, 303, 404)
        with logged()[1], patch('memory_ledger.record'):
            trace.prepare_device(seeds)
            pairs = []
            for first, second in quad_draft.PAIRS:
                pair = dflash_proposal_trace.PreparedPackedDFlashProposal(devices[first], devices[second])
                pair.prepare_device(seeds[first], seeds[second])
                pairs.append(next(iter(pair.buckets.values())))
        bucket = trace.buckets[(2048,) * 4]
        for index, pair in enumerate(pairs):
            rows = slice(32 * index, 32 * (index + 1))
            with self.subTest(pair=index):
                self.assertTrue(torch.equal(bucket.identifiers.value[:, rows], pair.identifiers.value))
                for name in ('q', 'live_k'):
                    for table, theirs in zip(bucket.rope[name], pair.rope[name]):
                        self.assertTrue(same_bits(table.value[:, :, rows], theirs.value), name)
                self.assertTrue(same_bits(bucket.mask.value, pair.mask.value))
                self.assertTrue(getattr(pair, 'row_exact', False))

    def test_the_mask_refresh_and_audit(self):
        os.environ['QWEN_FAST_PAIR_MASK_REFRESH'] = '1'
        os.environ['QWEN_FAST_PAIR_MASK_AUDIT'] = '1'
        ops, devices, trace = self.build()
        lines, loguru = logged()
        with loguru, patch('memory_ledger.record'):
            trace.round_number = 7
            trace.prepare_device((1, 2, 3, 4))
            bucket = trace.buckets[(2048,) * 4]
            bucket.mask.chips[1][0, 0, 3, :] = 5.0
            trace.discard_pending()
            trace.round_number = 8
            trace.prepare_device((1, 2, 3, 4))
        copies = [event for event in ops.events if event[0] == 'copy_host_to_device_tensor'
                  and event[2] == ops.normalize(bucket.mask)]
        self.assertEqual(len(copies), 3, 'the build update and two rounds')
        self.assertEqual([line for line in lines if 'pair mask refresh' in line][:1],
                         ['[PINDIAG] pair mask refresh engaged pair=[0, 1, 2, 3] context=2048,2048,2048,2048 '
                          'bytes=133120'])
        masks = [line for line in lines if line.startswith('[QUAD-DRAFT] mask')]
        self.assertIn('[QUAD-DRAFT] mask round=8 intact=0 mismatched=2080 chip=1', masks)
        self.assertEqual(len(masks), 6)
        for chip in bucket.mask.chips:
            self.assertTrue(same_bits(chip, bucket.host_mask), 'the refresh healed it')

    def test_prepare_collect_adopt_finish(self):
        ops, devices, trace = self.build()
        parts = [dict(user=user) for user in range(4)]
        with logged()[1], patch('memory_ledger.record'), \
                patch.object(quad_draft, 'read_quad_outputs', return_value=parts):
            trace.prepare_device((11, 22, 33, 44))
            for which, seed in enumerate((11, 22, 33, 44)):
                self.assertTrue(trace.has_pending(which, seed))
                self.assertFalse(trace.has_pending(which, seed + 1))
            collected = trace.collect()
            self.assertEqual(collected, dict(parts=parts, seeds=(11, 22, 33, 44), counts=(15,) * 4))
            with self.assertRaises(ValueError):
                trace.adopt([(1,), (2,)])
            trace.adopt([(1, 2, 3), (4,), (5, 6), (7,)])
            with self.assertRaises(ValueError):
                trace.collect()
            self.assertEqual(trace.finish(2, 1), (5,))
            self.assertFalse(trace.has_pending(2, 33))
            self.assertEqual(trace.finish(0, 2), (1, 2))
            self.assertEqual(trace.finish(1, 5), (4,))
            self.assertIsNotNone(trace._pending)
            self.assertEqual(trace.finish(3, 1), (7,))
            self.assertIsNone(trace._pending)

    def test_finish_without_a_batched_selection_selects_itself(self):
        ops, devices, trace = self.build()
        with logged()[1], patch('memory_ledger.record'), \
                patch.object(quad_draft, 'select_quad_outputs', return_value=((1,), (2,), (3,), (4,))) as select:
            trace.prepare_device((11, 22, 33, 44))
            self.assertEqual(trace.finish(1, 1), (2,))
            self.assertEqual(select.call_args.args[2:], ((11, 22, 33, 44), (15,) * 4))
            self.assertEqual(trace.audit_selection(), ((1,), (2,), (3,), (4,)))

    def test_a_device_mid_publication_or_closed_declines(self):
        for attribute, value in (('pending', object()), ('closed', True), ('progress', object())):
            ops, devices, trace = self.build()
            setattr(devices[3], attribute, value)
            with self.subTest(attribute=attribute):
                self.assertFalse(trace.prepare_device((1, 2, 3, 4)))
                self.assertEqual(trace.buckets, {})

    def test_a_failed_build_releases_every_placeholder(self):
        ops, devices, trace = self.build()
        devices[0].execute_proposal.side_effect = RuntimeError('capture failed')
        with logged()[1], self.assertRaises(RuntimeError):
            trace.prepare_device((1, 2, 3, 4))
        self.assertEqual(trace.buckets, {})
        self.assertEqual(trace.owned, [])
        freed = [event for event in ops.events if event[0] == 'deallocate']
        self.assertEqual(len(freed), 6, 'ids, mask and the four rope tables')
        self.assertEqual(devices[0].validated_native_proposal_masks, set())

    def test_without_live_banks_the_build_is_refused(self):
        ops, devices, trace = self.build()
        with patch('fused_commit.live_bank_history', return_value=None), logged()[1], self.assertRaises(ValueError):
            trace.prepare_device((1, 2, 3, 4))
        self.assertEqual(trace.buckets, {})

    def test_close_releases_the_trace_and_the_placeholders(self):
        ops, devices, trace = self.build()
        with logged()[1], patch('memory_ledger.record'):
            trace.prepare_device((1, 2, 3, 4))
        trace.close()
        self.assertTrue(trace.closed)
        self.assertEqual([event[0] for event in ops.events].count('release_trace'), 1)
        self.assertEqual(trace.owned, [])
        self.assertFalse(trace.prepare_device((1, 2, 3, 4)))

    def test_the_constructor_refuses_what_the_quad_cannot_serve(self):
        from test_dflash_proposal_trace import RecordingOps

        ops = RecordingOps()
        devices = quad_devices(ops)
        with self.assertRaises(ValueError):
            quad_draft.PreparedQuadDFlashProposal(devices[:3])
        devices[2].block_rows = 8
        with self.assertRaises(ValueError):
            quad_draft.PreparedQuadDFlashProposal(devices)

    def test_a_live_bank_on_the_spare_side_is_copied_into_the_pools_active_bank(self):
        """Round-fence plan H1b, F4: a device whose live bank sits on the pool's spare side (an odd swap count before
        its first in-place commit) has its rows copied into the pool's active bank, which the quad reads - the pair's
        own normalisation, at the build's update and every round's; a bank the quad reads itself copies nothing."""
        import fused_commit
        from test_dflash_proposal_trace import FakeTensor

        ops, devices, trace = self.build()
        spare = [{name: FakeTensor(torch.zeros(1), 'bf16', 'tile', True, ops.addresses) for name in ('k', 'v')}
                 for _ in range(5)]

        def banks(first, second):
            out = live_banks(first, second)
            if second is devices[3]:
                out[1] = spare
            return out

        lines, loguru = logged()
        with loguru, patch('fused_commit.live_bank_history', side_effect=banks), patch('memory_ledger.record'):
            self.assertTrue(trace.prepare_device((1, 2, 3, 4)))
        bucket = trace.buckets[(2048,) * 4]
        self.assertIs(bucket.cached_history[3], spare)
        copies = [event for event in ops.events if event[0] == 'copy']
        self.assertEqual(len(copies), 2 * 10, "the build's update and the round's, device 3 only")
        self.assertEqual({event[2] for event in copies},
                         {ops.normalize(layer[name]) for layer in spare for name in ('k', 'v')})
        self.assertEqual([line for line in lines if line.startswith(fused_commit.LIVE_BANKS_NORMALISED)],
                         [fused_commit.LIVE_BANKS_NORMALISED + ' pair=[0, 1, 2, 3] normalised=10'] * 2)

    def test_a_user_that_never_reads_its_drafts_leaves_nothing_for_the_next_round(self):
        """EOS: a user whose finish() never comes leaves the round pending for it alone - the users that read theirs
        are never pending again for those seeds - and the next prepare_device discards it."""
        ops, devices, trace = self.build()
        with logged()[1], patch('memory_ledger.record'), \
                patch.object(quad_draft, 'read_quad_outputs', return_value=[dict(user=user) for user in range(4)]):
            trace.prepare_device((11, 22, 33, 44))
            trace.collect()
            trace.adopt([(1,), (2,), (3,), (4,)])
            for which in range(3):
                self.assertEqual(trace.finish(which, 1), (which + 1,))
            self.assertEqual([trace.has_pending(which, seed) for which, seed in enumerate((11, 22, 33, 44))],
                             [False, False, False, True])
            trace.prepare_device((11, 22, 33, 55))
            self.assertFalse(trace.has_pending(3, 44))
            self.assertEqual([trace.has_pending(which, seed) for which, seed in enumerate((11, 22, 33, 55))],
                             [True] * 4)
            self.assertIsNone(trace._pending[1].tokens, 'the new round selects afresh')


# ---------------------------------------------------------------------------------------------
# The shadow audit's comparison.
# ---------------------------------------------------------------------------------------------

class CompareTests(unittest.TestCase):
    def fixture(self):
        generator = torch.Generator().manual_seed(9)
        ops = HostOps()
        pair_buckets, quad_chunks, quad_parts, pairs = [], [], [], []
        tokens = []
        for index in range(2):
            chunks = [dict(values=chips(torch.randn(1, 1, 32, 16, generator=generator),
                                        torch.randn(1, 1, 32, 16, generator=generator)),
                           indices=chips(torch.randint(0, 999, (1, 1, 32, 16), generator=generator),
                                         torch.randint(0, 999, (1, 1, 32, 16), generator=generator)))
                      for _ in range(4)]
            projected = torch.randn(1, 1, 32, 256, generator=generator).bfloat16()
            parts = [dict(hidden=torch.randn(1, 15, 256, generator=generator).bfloat16(),
                          candidates=torch.randint(0, 999, (1, 15, 16), generator=generator),
                          unary=torch.randn(1, 15, 16, generator=generator)) for _ in range(2)]
            pair_bucket = SimpleNamespace(outputs=SimpleNamespace(chunks=chunks, projected=chips(projected, projected)))
            seeds = (10 * (2 * index + 1), 10 * (2 * index + 2))
            pair = SimpleNamespace(device_a=SimpleNamespace(predecessors='p', successors='s'), discarded=0)
            pair.collect = Mock(return_value=dict(parts=parts, seeds=seeds, counts=(15, 15)))
            pair._pending = (seeds[0], seeds[1], pair_bucket, [])
            pairs.append(([2 * index, 2 * index + 1], pair))
            quad_parts.extend(dict((key, value.clone()) for key, value in part.items()) for part in parts)
            pair_buckets.append(pair_bucket)
            tokens.extend([(index, 1), (index, 2)])
        for number in range(4):
            quad_chunks.append({key: chips(*(torch.cat([pair_buckets[0].outputs.chunks[number][key].chips[chip],
                                                        pair_buckets[1].outputs.chunks[number][key].chips[chip]], dim=2)
                                             for chip in range(2))) for key in ('values', 'indices')})
        wide = torch.cat([bucket.outputs.projected.chips[0] for bucket in pair_buckets], dim=2)
        bucket = SimpleNamespace(parts=quad_parts, tokens=tuple(tokens),
                                 outputs=SimpleNamespace(chunks=quad_chunks, projected=chips(wide, wide)))
        quad = SimpleNamespace(operations=ops, _pending=((10, 20, 30, 40), bucket, []))
        return quad, pairs

    def selected(self, parts, seeds, counts, predecessors, successors):
        return ((seeds[0] // 10 - 1 >= 2 and 1 or 0, 1), (seeds[0] // 10 - 1 >= 2 and 1 or 0, 2))

    def compare(self, quad, pairs):
        if getattr(quad, '_audit_snapshot', None) is None:
            # snapshot_audit's copy: the quad's outputs as its replay left them, before the pairs replayed.
            quad._audit_snapshot = quad_draft.raw_outputs(quad.operations, quad._pending[1].outputs)
        with patch('dflash_packed_proposal.select_packed_batched', side_effect=self.selected):
            return quad_draft.compare_with_pairs(quad, pairs)

    def test_equal_rounds_compare_every_user_and_every_raw_read(self):
        quad, pairs = self.fixture()
        equal, stage, checks = self.compare(quad, pairs)
        self.assertEqual((equal, stage), (True, 'all'))
        # the quad against its snapshot: 4 chunks x 2 keys x 2 chips and 2 feature reads; per pair: 2 users x 4
        # checks, 4 chunks x 2 keys x 2 chips, 2 feature reads
        self.assertEqual(checks, (16 + 2) + 2 * (2 * 4 + 16 + 2))

    def test_a_pair_replay_that_wrote_into_the_quads_outputs_is_named_never_compared(self):
        """A quad output the capture was given out of a pair capture's freed holes takes the audit pairs' writes:
        even when those bytes are the pair's own (the vacuous case), the snapshot names it."""
        quad, pairs = self.fixture()
        quad._audit_snapshot = quad_draft.raw_outputs(quad.operations, quad._pending[1].outputs)
        quad._pending[1].outputs.chunks[1]['indices'].chips[0][0, 0, 5, 2] += 1
        self.assertEqual(self.compare(quad, pairs)[:2], (False, 'quad-overwritten:indices:chunk1:chip0'))
        quad, pairs = self.fixture()
        quad._audit_snapshot = quad_draft.raw_outputs(quad.operations, quad._pending[1].outputs)
        quad._pending[1].outputs.projected.chips[0][0, 0, 40, 9] = 3.0
        self.assertEqual(self.compare(quad, pairs)[:2], (False, 'quad-overwritten:projected:chip0'))

    def test_the_reads_are_taken_once_and_a_later_write_changes_nothing(self):
        """collect() takes read_audit before the round's GDN flush; the comparison after the selection reads only
        those host copies, so a replay enqueued in between cannot reach it."""
        quad, pairs = self.fixture()
        quad._audit_snapshot = quad_draft.raw_outputs(quad.operations, quad._pending[1].outputs)
        reads = quad_draft.read_audit(quad, pairs)
        for _, pair in pairs:
            pair.collect.assert_called_once_with()
            pair._pending[2].outputs.chunks[0]['values'].chips[0].fill_(7.0)
        quad._pending[1].outputs.projected.chips[0].fill_(7.0)
        with patch('dflash_packed_proposal.select_packed_batched', side_effect=self.selected):
            self.assertEqual(quad_draft.compare_with_pairs(quad, pairs, reads=reads)[:2], (True, 'all'))
        for _, pair in pairs:
            pair.collect.assert_called_once_with()
        with self.assertRaises(ValueError):
            quad_draft.compare_with_pairs(quad, pairs, reads=dict(reads, pairs=reads['pairs'][::-1]))

    def test_the_reads_need_the_snapshot(self):
        quad, pairs = self.fixture()
        with self.assertRaises(ValueError):
            quad_draft.read_audit(quad, pairs)
        for _, pair in pairs:
            pair.collect.assert_not_called()

    def test_each_difference_names_its_stage(self):
        cases = (('features:u2', lambda quad, pairs: quad._pending[1].parts[2]['hidden'].add_(1)),
                 ('candidates:u1', lambda quad, pairs: quad._pending[1].parts[1]['candidates'].add_(1)),
                 ('scores:u3', lambda quad, pairs: quad._pending[1].parts[3]['unary'].add_(1)),
                 ('seeds:pair1', lambda quad, pairs: setattr(quad, '_pending', ((10, 20, 30, 41),) + quad._pending[1:])),
                 ('values:chunk2:chip1:pair0', lambda quad, pairs: quad._pending[1].outputs.chunks[2]['values']
                  .chips[1].__setitem__((0, 0, 3, 3), 99.0)),
                 ('indices:chunk0:chip0:pair1', lambda quad, pairs: quad._pending[1].outputs.chunks[0]['indices']
                  .chips[0].__setitem__((0, 0, 40, 0), 1234)),
                 ('projected:chip0:pair1', lambda quad, pairs: quad._pending[1].outputs.projected.chips[0]
                  .__setitem__((0, 0, 33, 0), 7.0)))
        for stage, mutate in cases:
            quad, pairs = self.fixture()
            mutate(quad, pairs)
            with self.subTest(stage=stage):
                self.assertEqual(self.compare(quad, pairs)[:2], (False, stage))
        quad, pairs = self.fixture()
        quad._pending[1].tokens = ((0, 1), (0, 2), (1, 1), (9, 9))
        self.assertEqual(self.compare(quad, pairs)[:2], (False, 'tokens:u3'))

    def test_it_never_compares_the_quad_with_itself(self):
        from test_dflash_proposal_trace import RecordingOps

        quad, pairs = self.fixture()
        itself = quad_draft.PreparedQuadDFlashProposal.__new__(quad_draft.PreparedQuadDFlashProposal)
        with self.assertRaises(ValueError):
            self.compare(quad, [pairs[0], ([2, 3], itself)])
        with self.assertRaises(ValueError):
            self.compare(quad, [pairs[0], ([2, 3], quad)])
        with self.assertRaises(ValueError):
            self.compare(quad, pairs[:1])

    def test_run_audit_logs_one_line_and_discards_the_pairs(self):
        from test_dflash_proposal_trace import RecordingOps

        with clean_environment(**{FLAG: '1'}, **REQUIRED):
            trace = quad_draft.PreparedQuadDFlashProposal(quad_devices(RecordingOps()))
        quad, pairs = self.fixture()
        for _, pair in pairs:
            pair.discard_pending = Mock()
        trace._pending, trace.operations = quad._pending, quad.operations
        trace._audit_snapshot = quad_draft.raw_outputs(quad.operations, quad._pending[1].outputs)
        lines, loguru = logged()
        trace.attach_audit(pairs)
        with loguru, patch('dflash_packed_proposal.select_packed_batched', side_effect=self.selected):
            self.assertEqual(trace.run_audit(12)[:2], (True, 'all'))
            self.assertIsNone(trace._audit_snapshot, 'one round only')
            self.assertIsNone(trace.run_audit(13), 'nothing attached')
            trace.attach_audit('pairs-unavailable:MemoryError')
            trace.run_audit(14)
            trace.attach_audit(pairs)
            with patch.object(quad_draft, 'compare_with_pairs', side_effect=RuntimeError('read failed')):
                trace.run_audit(15)
            trace.attach_audit(pairs)
            trace.run_audit(16)
        self.assertEqual(lines, ['[QUAD-AUDIT] round=12 equal=1 stage=all users=4 checks=70',
                                 '[QUAD-AUDIT] round=14 equal=0 stage=pairs-unavailable:MemoryError users=4 checks=0',
                                 '[QUAD-AUDIT] round=15 equal=0 stage=error:RuntimeError users=4 checks=0',
                                 '[QUAD-AUDIT] round=16 equal=0 stage=error:ValueError users=4 checks=0'])
        for _, pair in pairs:
            self.assertEqual(pair.discard_pending.call_count, 3)

    def test_collect_takes_the_audits_reads_and_a_read_failure_is_the_audits_verdict(self):
        from test_dflash_proposal_trace import RecordingOps

        with clean_environment(**{FLAG: '1'}, **REQUIRED):
            trace = quad_draft.PreparedQuadDFlashProposal(quad_devices(RecordingOps()))
        quad, pairs = self.fixture()
        bucket = SimpleNamespace(tokens=None, consumed=set(), parts=None, inputs=[], addresses=[],
                                 outputs=quad._pending[1].outputs)
        reads = dict(quad='the reads collect() took')
        for failure in (None, RuntimeError('read failed')):
            trace._pending = ((10, 20, 30, 40), bucket, [])
            bucket.tokens, bucket.parts = None, None
            trace.attach_audit(pairs)
            lines, loguru = logged()
            outputs = patch.object(quad_draft, 'read_quad_outputs', return_value=quad._pending[1].parts)
            read = patch.object(quad_draft, 'read_audit', return_value=reads, side_effect=failure)
            compare = patch.object(quad_draft, 'compare_with_pairs', return_value=(True, 'all', 70))
            with loguru, outputs, read as read, compare as compare:
                trace.collect()
                read.assert_called_once_with(trace, pairs)
                trace.adopt(quad._pending[1].tokens)
                trace.run_audit(3)
            with self.subTest(failure=failure):
                if failure is None:
                    compare.assert_called_once_with(trace, pairs, reads=reads)
                    self.assertEqual(lines, ['[QUAD-AUDIT] round=3 equal=1 stage=all users=4 checks=70'])
                else:
                    compare.assert_not_called()
                    self.assertEqual(lines, ['[QUAD-AUDIT] round=3 equal=0 stage=read-error:RuntimeError users=4 '
                                             'checks=0'])

    def test_discarding_the_quad_discards_the_attached_pairs(self):
        from test_dflash_proposal_trace import RecordingOps

        with clean_environment(**{FLAG: '1'}, **REQUIRED):
            trace = quad_draft.PreparedQuadDFlashProposal(quad_devices(RecordingOps()))
        pairs = [([0, 1], Mock()), ([2, 3], Mock())]
        trace._pending = ((1, 2, 3, 4), SimpleNamespace(tokens=None, consumed=set(), parts=None), [])
        trace.attach_audit(pairs)
        trace.discard_pending()
        for _, pair in pairs:
            pair.discard_pending.assert_called_once_with()
        self.assertIsNone(trace._audit)


# ---------------------------------------------------------------------------------------------
# The coordinator.
# ---------------------------------------------------------------------------------------------

LOG = []


class FakeQuadTrace:
    instances = []
    failures = []

    def __init__(self, devices):
        self.devices = tuple(devices)
        self.prepared, self.audits, self.discards = [], [], 0
        self.closed, self.attached, self.last_built, self.round_number = False, None, False, None
        self.buckets = {}
        FakeQuadTrace.instances.append(self)

    @property
    def device_a(self):
        return self.devices[0]

    def prepare_device(self, seeds):
        LOG.append(('quad', tuple(seeds)))
        if FakeQuadTrace.failures and FakeQuadTrace.failures.pop(0):
            raise RuntimeError('quad capture failed')
        self.last_built = not self.buckets
        self.buckets[(2048,) * 4] = True
        self.prepared.append(tuple(seeds))
        return True

    def has_pending(self, which, seed):
        return bool(self.prepared) and seed == self.prepared[-1][which]

    def finish(self, which, count):
        return ('quad', which, count)

    def collect(self):
        LOG.append(('collect', 'quad'))
        return dict(parts=[dict(user=user) for user in range(4)], seeds=self.prepared[-1], counts=(15,) * 4)

    def adopt(self, tokens):
        self.adopted = tuple(tokens)

    def snapshot_audit(self):
        LOG.append(('snapshot',))

    def attach_audit(self, pairs):
        self.attached = pairs

    def run_audit(self, round_number):
        LOG.append(('audit', round_number))
        self.audits.append((round_number, self.attached))
        self.attached = None

    def discard_pending(self):
        self.discards += 1

    def close(self):
        self.closed = True


def pair_trace_class():
    from test_dflash_packed_proposal_coordinator import FakeTrace

    class Pair(FakeTrace):
        instances = []

        def __init__(self, device_a, device_b):
            super().__init__(device_a, device_b)
            type(self).instances.append(self)

        def prepare_device(self, seed_a, seed_b):
            LOG.append(('pair', seed_a, seed_b))
            return super().prepare_device(seed_a, seed_b)

        def collect(self):
            LOG.append(('collect', 'pair'))
            return dict(parts=[dict(user='a'), dict(user='b')], seeds=self.prepared[-1], counts=(15, 15))

        def adopt(self, tokens):
            self.adopted = tuple(tokens)
    return Pair


def quad_bridges(operations, mesh, slots=(0, 1, 2, 3), *, history_rows=2048):
    from test_dflash_packed_proposal_coordinator import make_bridge, make_device

    layers, predecessors, successors = [object()] * 5, object(), object()
    bridges = []
    for slot in slots:
        device = make_device(operations, mesh, slot=slot, history_rows=history_rows)
        device.layers, device.predecessors, device.successors = layers, predecessors, successors
        device.kv_history.active = list(layers)
        device.block_rows, device.fused_convolution = 16, True
        device.prepare_device = Mock(side_effect=lambda seed, slot=slot: LOG.append(('single', slot, seed)) or True)
        bridges.append(make_bridge('r%d' % slot, device, seed=100 + slot))
    return bridges


def strip_ms(line):
    """A log line without its host timings (propose_ms lists, collect_ms and select_ms)."""
    return re.sub(r"propose_ms=\[[^]]*\]|(collect|select)_ms=\S+", '', line)


def selected_tokens(parts, seeds, counts, predecessors, successors):
    return tuple((seed,) for seed in seeds)


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        environment = clean_environment(**{FLAG: '1', 'QWEN_FAST_PACKED_AUDIT': '1'}, **REQUIRED)
        environment.start()
        self.addCleanup(environment.stop)
        del LOG[:]
        FakeQuadTrace.instances, FakeQuadTrace.failures = [], []
        self.Pair = pair_trace_class()
        self.Pair.instances = []
        from test_dflash_packed_proposal_coordinator import FakeSingleUserCapture

        for target, value in (('quad_draft.PreparedQuadDFlashProposal', FakeQuadTrace),
                              ('dflash_proposal_trace.PreparedPackedDFlashProposal', self.Pair),
                              ('dflash_proposal_trace.PreparedDFlashProposal', FakeSingleUserCapture),
                              ('dflash_packed_proposal.select_packed_batched', Mock(side_effect=selected_tokens))):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: LOG.append(('sync',))))
        self.mesh = object()

    def coordinator(self):
        import dflash_packed_proposal_coordinator

        return dflash_packed_proposal_coordinator.PackedProposalCoordinator()

    def prepare(self, coordinator, bridges, **options):
        lines, loguru = logged()
        with loguru:
            prepared = coordinator.prepare(bridges, **options)
        return prepared, lines

    def test_four_live_packable_users_run_one_quad_and_no_pair(self):
        from dflash_packed_proposal_coordinator import _PackedCaptureView

        coordinator, bridges = self.coordinator(), quad_bridges(self.operations, self.mesh)
        prepared, lines = self.prepare(coordinator, bridges)
        self.assertEqual(len(FakeQuadTrace.instances), 1)
        self.assertEqual(self.Pair.instances, [])
        trace = FakeQuadTrace.instances[0]
        self.assertEqual(trace.prepared, [(100, 101, 102, 103)])
        self.assertEqual([id(device) for device in prepared], [id(bridge.request.runtime.drafter) for bridge in bridges])
        for which, bridge in enumerate(bridges):
            capture = bridge.request.runtime.drafter.proposal_capture
            self.assertIsInstance(capture, _PackedCaptureView)
            self.assertIs(capture._trace, trace)
            self.assertEqual(capture._which, which)
            self.assertTrue(bridge.request.runtime.drafter._packed_capture_released, 'R6: singles released')
            bridge.request.runtime.drafter.prepare_device.assert_not_called()
        self.assertEqual(LOG, [('quad', (100, 101, 102, 103)), ('sync',), ('collect', 'quad'), ('audit', 1)])
        self.assertEqual(trace.adopted, ((100,), (101,), (102,), (103,)))
        self.assertIn('[QUAD-DRAFT] round=1 built=1 ms=', lines[0])
        self.assertTrue(any(line.startswith('[PACKED-SELECT] round=1 pairs=[[0, 1, 2, 3]] users=4 ') for line in lines))
        self.assertFalse([line for line in lines if 'fallback' in line or 'disabled' in line])
        self.assertEqual(coordinator.quad_rounds, 1)
        prepared, lines = self.prepare(coordinator, bridges)
        self.assertEqual(len(FakeQuadTrace.instances), 1, 'the same four devices replay the same quad')
        self.assertIn('[QUAD-DRAFT] round=2 built=0 ms=', lines[0])

    def test_three_live_and_two_live_rounds_are_todays_pairs_call_for_call(self):
        import dflash_packed_proposal_coordinator

        def run(flags, slots, history_rows=2048):
            del LOG[:]
            self.Pair.instances, FakeQuadTrace.instances = [], []
            with clean_environment(**flags):
                coordinator = dflash_packed_proposal_coordinator.PackedProposalCoordinator()
                bridges = quad_bridges(self.operations, self.mesh, slots, history_rows=history_rows)
                _, lines = self.prepare(coordinator, bridges)
                _, more = self.prepare(coordinator, bridges)
            return list(LOG), [strip_ms(line) for line in lines + more]

        base = dict(REQUIRED, QWEN_FAST_PACKED_AUDIT='1')
        for slots, rows in (((0, 1, 3), 2048), ((0, 1), 2048), ((2, 3), 2048), ((0, 2), 2048), ((0, 1, 2, 3), 1024)):
            with self.subTest(slots=slots, rows=rows):
                off = run(base, slots, rows)
                on = run(dict(base, **{FLAG: '1'}), slots, rows)
                self.assertEqual(on, off)
                self.assertEqual(FakeQuadTrace.instances, [])

    def test_a_failure_falls_back_to_the_pairs_and_two_in_a_row_give_up(self):
        coordinator, bridges = self.coordinator(), quad_bridges(self.operations, self.mesh)
        FakeQuadTrace.failures = [True, False, True, True]
        _, lines = self.prepare(coordinator, bridges)
        self.assertEqual(len(self.Pair.instances), 2, 'the round fell back to the pairs')
        self.assertTrue(lines[0].startswith('[QUAD-DRAFT] fallback round=1 reason=RuntimeError:quad_capture_failed'))
        self.assertEqual(FakeQuadTrace.instances[0].discards, 1)
        _, lines = self.prepare(coordinator, bridges)
        self.assertEqual(coordinator.quad_failures, 0, 'a success resets the count')
        self.assertEqual(FakeQuadTrace.instances[0].prepared, [(100, 101, 102, 103)])
        _, lines = self.prepare(coordinator, bridges)
        self.assertFalse(coordinator.quad_disabled, 'one failure after a success is not two in a row')
        _, lines = self.prepare(coordinator, bridges)
        self.assertTrue(coordinator.quad_disabled)
        self.assertTrue(any(line.startswith('[PINDIAG] quad draft disabled round=4 failures=2 '
                                            'reason=consecutive_failures=2') for line in lines))
        self.assertTrue(FakeQuadTrace.instances[0].closed)
        self.assertIsNone(coordinator.quad)
        del LOG[:]
        _, lines = self.prepare(coordinator, bridges)
        self.assertNotIn('quad', [entry[0] for entry in LOG], 'never tried again')
        self.assertFalse([line for line in lines if 'QUAD' in line or 'quad' in line])

    def test_a_missing_requirement_disables_it_once_and_the_pairs_run(self):
        os.environ['QWEN_FAST_FUSED_COMMIT_LIVE_BANKS'] = '0'
        coordinator, bridges = self.coordinator(), quad_bridges(self.operations, self.mesh)
        _, lines = self.prepare(coordinator, bridges)
        _, more = self.prepare(coordinator, bridges)
        disabled = [line for line in lines + more if line.startswith(quad_draft.DISABLED_MARKER)]
        self.assertEqual(disabled, ['[PINDIAG] quad draft disabled round=1 failures=0 '
                                    'reason=requires_QWEN_FAST_FUSED_COMMIT_LIVE_BANKS=1'])
        self.assertEqual(FakeQuadTrace.instances, [])
        self.assertEqual(len(self.Pair.instances), 2)

    def test_devices_that_do_not_share_one_weight_set_are_refused(self):
        coordinator, bridges = self.coordinator(), quad_bridges(self.operations, self.mesh)
        bridges[3].request.runtime.drafter.layers = [object()] * 5
        _, lines = self.prepare(coordinator, bridges)
        self.assertTrue(lines[0].endswith('reason=the_four_devices_do_not_share_one_draft_weight_set'))

    def test_too_little_dram_for_a_fresh_build_falls_back(self):
        coordinator, bridges = self.coordinator(), quad_bridges(self.operations, self.mesh)
        with patch('dflash_packed_proposal_coordinator.dram_headroom', return_value=100 * 2 ** 20):
            _, lines = self.prepare(coordinator, bridges)
        self.assertEqual(FakeQuadTrace.instances, [])
        self.assertEqual(lines[0], '[QUAD-DRAFT] fallback round=1 reason=dram_reserve:headroom=104857600')
        self.assertFalse(coordinator.quad_disabled)

    def test_a_bad_flag_value_fails_the_round(self):
        os.environ[FLAG] = 'yes'
        coordinator, bridges = self.coordinator(), quad_bridges(self.operations, self.mesh)
        with self.assertRaises(ValueError):
            self.prepare(coordinator, bridges)

    def test_a_stale_quad_is_retired_and_a_three_live_round_keeps_a_live_one(self):
        coordinator, bridges = self.coordinator(), quad_bridges(self.operations, self.mesh)
        self.prepare(coordinator, bridges)
        trace = FakeQuadTrace.instances[0]
        self.prepare(coordinator, bridges[:3])
        self.assertFalse(trace.closed, 'slot 3 is absent this round, not gone')
        self.assertIs(coordinator.quad[1], trace)
        bridges[3].request.runtime.drafter.closed = True
        replacement = quad_bridges(self.operations, self.mesh, (3,))[0]
        first, new = bridges[0].request.runtime.drafter, replacement.request.runtime.drafter
        new.layers, new.predecessors, new.successors = first.layers, first.predecessors, first.successors
        new.kv_history.active = list(first.layers)
        self.prepare(coordinator, bridges[:3] + [replacement])
        self.assertTrue(trace.closed)
        self.assertEqual(len(FakeQuadTrace.instances), 2, 'the new request on slot 3 builds a new quad')

    def test_reads_covered_and_the_early_drafts_flush_after_the_quads_readback(self):
        from dflash_packed_proposal_coordinator import reads_covered

        coordinator, bridges = self.coordinator(), quad_bridges(self.operations, self.mesh)
        flushed = Mock(side_effect=lambda: LOG.append(('flush',)))
        self.prepare(coordinator, bridges, after_reads=flushed)
        flushed.assert_called_once_with()
        self.assertEqual(LOG[:4], [('quad', (100, 101, 102, 103)), ('sync',), ('collect', 'quad'), ('flush',)])
        trace = FakeQuadTrace.instances[0]
        devices = [bridge.request.runtime.drafter for bridge in bridges]
        self.assertTrue(reads_covered([([0, 1, 2, 3], trace)], devices))
        self.assertFalse(reads_covered([([0, 1, 2, 3], trace)], devices + [SimpleNamespace()]))

    def test_the_audit_replays_both_pairs_before_the_fence_and_compares_after_the_selection(self):
        os.environ[quad_draft.AUDIT_FLAG] = 'all'
        coordinator, bridges = self.coordinator(), quad_bridges(self.operations, self.mesh)
        self.prepare(coordinator, bridges)
        self.assertEqual(LOG, [('quad', (100, 101, 102, 103)), ('snapshot',), ('pair', 100, 101), ('pair', 102, 103),
                               ('sync',), ('collect', 'quad'), ('audit', 1)])
        trace = FakeQuadTrace.instances[0]
        (round_number, pairs), = trace.audits
        self.assertEqual([labels for labels, _ in pairs], [[0, 1], [2, 3]])
        self.assertEqual([type(pair) for _, pair in pairs], [self.Pair, self.Pair])
        self.assertTrue(all(pair is not trace for _, pair in pairs), 'never the quad itself')
        for bridge in bridges:
            self.assertIs(bridge.request.runtime.drafter.proposal_capture._trace, trace, 'the pairs install no view')
        del LOG[:]
        self.prepare(coordinator, bridges)
        self.assertEqual(len(self.Pair.instances), 2, 'the audit reuses the built pairs')

    def test_an_audit_whose_snapshot_fails_is_unequal_and_replays_no_pair(self):
        os.environ[quad_draft.AUDIT_FLAG] = 'all'
        coordinator, bridges = self.coordinator(), quad_bridges(self.operations, self.mesh)
        with patch.object(FakeQuadTrace, 'snapshot_audit', side_effect=RuntimeError('read failed')):
            self.prepare(coordinator, bridges)
        trace = FakeQuadTrace.instances[0]
        self.assertEqual(trace.audits, [(1, 'snapshot-unavailable:RuntimeError')])
        self.assertEqual(self.Pair.instances, [])

    def test_a_rebuild_after_a_failed_build_is_gated_by_the_headroom_and_a_replay_is_not(self):
        coordinator, bridges = self.coordinator(), quad_bridges(self.operations, self.mesh)
        FakeQuadTrace.failures = [True]
        self.prepare(coordinator, bridges)
        trace = FakeQuadTrace.instances[0]
        self.assertEqual(trace.buckets, {}, 'the failed build kept nothing')
        with patch('dflash_packed_proposal_coordinator.dram_headroom', return_value=100 * 2 ** 20):
            _, lines = self.prepare(coordinator, bridges)
        self.assertEqual(lines[0], '[QUAD-DRAFT] fallback round=2 reason=dram_reserve:headroom=104857600')
        self.assertEqual(trace.prepared, [], 'no capture without the headroom')
        self.assertEqual(coordinator.quad_failures, 1, 'a DRAM refusal is not a failure')
        self.prepare(coordinator, bridges)
        self.assertEqual(trace.prepared, [(100, 101, 102, 103)])
        self.assertEqual(len(FakeQuadTrace.instances), 1, 'the same four devices keep their quad')
        with patch('dflash_packed_proposal_coordinator.dram_headroom', return_value=100 * 2 ** 20) as headroom:
            _, lines = self.prepare(coordinator, bridges)
        headroom.assert_not_called()
        self.assertIn('[QUAD-DRAFT] round=4 built=0 ms=', lines[0])

    def test_an_audit_of_n_rounds_stops_after_n(self):
        os.environ[quad_draft.AUDIT_FLAG] = '1'
        coordinator, bridges = self.coordinator(), quad_bridges(self.operations, self.mesh)
        self.prepare(coordinator, bridges)
        self.prepare(coordinator, bridges)
        trace = FakeQuadTrace.instances[0]
        self.assertEqual([pairs is not None for _, pairs in trace.audits], [True, False])

    def test_under_the_release_line_the_pairs_are_released_after_the_build(self):
        coordinator, bridges = self.coordinator(), quad_bridges(self.operations, self.mesh)
        self.prepare(coordinator, bridges[:2])
        pair = self.Pair.instances[0]
        with patch('dflash_packed_proposal_coordinator.dram_headroom', side_effect=[None, 500 * 10 ** 6]):
            _, lines = self.prepare(coordinator, bridges)
        self.assertTrue(pair.closed)
        self.assertNotIn((0, 1), coordinator.pairs)
        self.assertIn('[QUAD-DRAFT] released pairs=[[0, 1]] headroom=500000000', lines)

    def test_a_failure_after_the_quad_prepared_fences_and_discards_it(self):
        import dflash_packed_proposal_coordinator

        coordinator, bridges = self.coordinator(), quad_bridges(self.operations, self.mesh)
        with patch.object(dflash_packed_proposal_coordinator.PackedProposalCoordinator, '_release_single_user',
                          side_effect=RuntimeError('close failed')), self.assertRaises(RuntimeError):
            self.prepare(coordinator, bridges)
        trace = FakeQuadTrace.instances[0]
        self.assertEqual(trace.discards, 1)
        self.assertEqual(LOG, [('quad', (100, 101, 102, 103)), ('sync',)])

    def test_close_closes_the_quad(self):
        coordinator, bridges = self.coordinator(), quad_bridges(self.operations, self.mesh)
        self.prepare(coordinator, bridges)
        coordinator.close()
        self.assertTrue(FakeQuadTrace.instances[0].closed)
        self.assertIsNone(coordinator.quad)

    def test_the_fence_window_runs_once_the_quad_is_enqueued_and_a_failing_one_is_dropped(self):
        """Round-fence plan H1a: the window's callable runs after the quad's replay is enqueued, before the round's
        one fence, and its `fenced` right after that fence; one that raises goes to its `drop`, never the round."""
        coordinator, bridges = self.coordinator(), quad_bridges(self.operations, self.mesh)
        window = Mock(side_effect=lambda: LOG.append(('window',)))
        window.fenced = Mock(side_effect=lambda: LOG.append(('fenced',)))
        window.drop = Mock()
        self.prepare(coordinator, bridges, while_waiting=window)
        self.assertEqual(LOG, [('quad', (100, 101, 102, 103)), ('window',), ('sync',), ('fenced',), ('collect', 'quad'),
                               ('audit', 1)])
        window.drop.assert_not_called()
        del LOG[:]
        failure = RuntimeError('prestage failed')
        window.side_effect = failure
        self.prepare(coordinator, bridges, while_waiting=window)
        window.drop.assert_called_once_with(failure)
        self.assertEqual(LOG, [('quad', (100, 101, 102, 103)), ('sync',), ('fenced',), ('collect', 'quad'),
                               ('audit', 2)])
        self.assertEqual(FakeQuadTrace.instances[0].adopted, ((100,), (101,), (102,), (103,)))

    def test_the_pool_slots_not_the_bridge_order_decide_who_is_which_user(self):
        """M3NATIVE_STAGGER / START_ORDER change the order the bridges arrive in, never who is which user: the quad
        takes its devices, seeds and views in pool-slot order."""
        coordinator, bridges = self.coordinator(), quad_bridges(self.operations, self.mesh)
        prepared, _ = self.prepare(coordinator, [bridges[index] for index in (1, 0, 3, 2)])
        trace = FakeQuadTrace.instances[0]
        devices = tuple(bridge.request.runtime.drafter for bridge in bridges)
        self.assertEqual(trace.devices, devices)
        self.assertEqual(trace.prepared, [(100, 101, 102, 103)])
        self.assertEqual([id(device) for device in prepared], [id(device) for device in devices])
        for slot, device in enumerate(devices):
            self.assertIs(device.proposal_capture._trace, trace)
            self.assertEqual(device.proposal_capture._which, slot)

    def test_a_user_gone_mid_round_retires_the_quad_and_the_survivors_run_todays_round(self):
        """A user that finished (EOS, or its max_tokens tail) after a quad round - its device closed, its slot empty
        the next round - retires the quad before anything else; the three survivors then run exactly what the flag
        off runs at that point (pair (0, 1) and slot 2's rebuilt single-user capture), call for call."""
        import dflash_packed_proposal_coordinator

        def run(flags):
            del LOG[:]
            self.Pair.instances, FakeQuadTrace.instances = [], []
            with clean_environment(**flags):
                coordinator = dflash_packed_proposal_coordinator.PackedProposalCoordinator()
                bridges = quad_bridges(self.operations, self.mesh)
                self.prepare(coordinator, bridges)
                bridges[3].request.runtime.drafter.closed = True
                del LOG[:]
                _, lines = self.prepare(coordinator, bridges[:3])
            return list(LOG), [strip_ms(line) for line in lines], bridges

        base = dict(REQUIRED, QWEN_FAST_PACKED_AUDIT='1')
        off = run(base)
        on = run(dict(base, **{FLAG: '1'}))
        (trace,) = FakeQuadTrace.instances
        self.assertTrue(trace.closed, 'retired: its device closed')
        self.assertEqual(on[:2], off[:2])
        self.assertIn(('pair', 100, 101), on[0])
        self.assertIn('[PACKED-PROPOSE] recapture slot=2', on[1])
        self.assertNotIn('quad', [entry[0] for entry in on[0]])
        survivor = on[2][2].request.runtime.drafter
        self.assertIs(survivor.proposal_capture._trace, trace, 'the survivor keeps the retired view, pending nothing')
        self.assertIsNotNone(survivor.proposal_capture._original, 'its single-user capture rebuilt')


# ---------------------------------------------------------------------------------------------
# The flag off: every touched module is its parent's, call for call.
# ---------------------------------------------------------------------------------------------

class FlagOffIsTheParentTests(unittest.TestCase):
    def parent(self, relative):
        module = parent_module(relative, relative[:-3] + '_quad_parent')
        if module is None:
            self.skipTest('no git history for %s' % PARENT)
        return module

    def test_the_attention_branch(self):
        import draft_attention_branch

        parent = self.parent('draft_attention_branch.py')
        for row_exact in (True, False):
            with self.subTest(row_exact=row_exact):
                before = attention_run(parent, row_exact=row_exact)
                today = attention_run(draft_attention_branch, row_exact=row_exact)
                self.assertGreater(len(before), 40)
                self.assertEqual(today, before)
        from test_pair_row_exact import Recorder, run_branch

        for extra in (dict(), dict(mask_rows=2080, row_exact=True)):
            runs = []
            for module in (parent, draft_attention_branch):
                operations = Recorder()
                calls, _, _, _ = run_branch(module, operations, **extra)
                runs.append((operations.events, calls))
            self.assertEqual(runs[1], runs[0])

    def test_the_mlp_branch(self):
        import draft_mlp_branch

        parent = self.parent('draft_mlp_branch.py')
        for boundaries in (True, False):
            with self.subTest(boundaries=boundaries):
                before = mlp_run(parent, boundaries=boundaries)
                self.assertGreater(len(before), 25)
                self.assertEqual(mlp_run(draft_mlp_branch, boundaries=boundaries), before)

    def test_execute_proposal(self):
        import dflash_device
        from test_dflash_proposal_trace import Normalizer

        parent = self.parent('dflash_device.py')

        def run(module, pack, fused=False, **extra):
            from test_dflash_device_audit import ExecuteProposalTests as Audit

            audit = Audit()
            device = audit.device()
            device.kv_history, device.native_proposal_attention = object(), True
            # The served drafter runs the fused conv (v220): its branch, not only the composed one, must be the
            # parent's - the branches are handed the same convolution_operation and it calls checked_convolution
            # with the same arguments.
            device.fused_convolution, device.convolution_checks = fused, []
            normalize, log = Normalizer(), []

            def recording(name, value):
                def side_effect(*args, **kwargs):
                    result = value()
                    log.append((name, normalize(args), normalize(kwargs), normalize(result)))
                    return result
                return Mock(side_effect=side_effect)

            def branch(name, value):
                recorded = recording(name, value)

                def side_effect(*args, **kwargs):
                    operation = kwargs.get('convolution_operation')
                    if operation is not None:
                        operation('operations', 'mesh', 'hidden', ['dynamic'], ['base'], fp32_intermediates=True,
                                  retain_temporaries='retain', boundaries=kwargs.get('boundaries'))
                    return recorded(*args, **kwargs)
                return Mock(side_effect=side_effect)

            for name in ('reshape', 'pad', 'rms_norm', 'matmul', 'typecast', 'slice',
                         'MatmulMultiCoreReuseMultiCast1DProgramConfig'):
                setattr(device.operations, name, recording(name, object))
            device.model.embd = recording('embd', object)
            device.operations.experimental.all_gather_async = recording('all_gather', object)
            cached = [[object()] * 5] * len(pack) if pack else [object()] * 5
            with patch.object(module, 'projection_links', return_value=1), \
                    patch.object(module, 'shared_head_candidates', recording('head', lambda: ['chunks'])), \
                    patch.object(module, 'execute_attention_branch', branch('attention', object)), \
                    patch.object(module, 'execute_mlp_branch', branch('mlp', lambda: dict(output=object()))), \
                    patch('draft_convolution_fused.checked_convolution', recording('checked_convolution', object)), \
                    patch.object(module, 'addresses', return_value=('a', 'b')):
                device.validated_native_proposal_masks.add(('a', 'b'))
                module.DFlashDevice.execute_proposal(device, 'ids', None, 'mask', {'q': 1, 'k': 2, 'live_k': 3},
                    context=None if pack else 2048, pack=pack, cached_history=cached, owned=[],
                    retain=lambda value: value, stage=lambda name, **values: log.append(('stage', name)), **extra)
            return log

        pair = [dict(position=4096, history_rows=2048), dict(position=8192, history_rows=2048)]
        for fused in (False, True):
            for pack, extra in ((pair, {}), (pair, dict(row_exact=True)), (None, {})):
                with self.subTest(fused=fused, pack=bool(pack), extra=extra):
                    before = run(parent, pack, fused, **extra)
                    self.assertGreater(len(before), 30)
                    convs = len([entry for entry in before if entry[0] == 'checked_convolution'])
                    self.assertTrue(convs > 0 if fused else convs == 0, convs)
                    self.assertEqual(run(dflash_device, pack, fused, **extra), before)
                    self.assertEqual(run(dflash_device, pack, fused, quad=None, **extra), before)

    def coordinator_run(self, module, flags, rounds=((0, 1, 2, 3), (0, 1, 2), (0, 1, 2, 3))):
        del LOG[:]
        Pair = pair_trace_class()
        Pair.instances = []
        from test_dflash_packed_proposal_coordinator import FakeSingleUserCapture

        # A fresh once-per-process B1 marker for each run: the parent and today's then log the same lines whatever
        # ran before them (in isolation the first run logged it and the second did not).
        with clean_environment(**flags), patch('dflash_proposal_trace.PreparedPackedDFlashProposal', Pair), \
                patch('dflash_proposal_trace.PreparedDFlashProposal', FakeSingleUserCapture), \
                patch('dflash_packed_proposal.select_packed_batched', Mock(side_effect=selected_tokens)), \
                patch('dflash_packed_proposal._ROUND_B1_NOTED', []):
            operations = SimpleNamespace(synchronize_device=Mock(side_effect=lambda mesh: LOG.append(('sync',))))
            mesh = object()
            bridges = quad_bridges(operations, mesh)
            coordinator = module.PackedProposalCoordinator()
            lines, loguru = logged()
            with loguru:
                for slots in rounds:
                    coordinator.prepare([bridge for bridge in bridges
                                         if bridge.request.runtime.drafter.pool_slot.index in slots])
                coordinator.close()
        return list(LOG), [strip_ms(line) for line in lines]

    def test_the_coordinator(self):
        import dflash_packed_proposal_coordinator

        parent = self.parent('dflash_packed_proposal_coordinator.py')
        for flags in ({}, {FLAG: '0'}, {'QWEN_FAST_ROUND_B1': '1', 'QWEN_FAST_PACKED_AUDIT': '1'},
                      dict(REQUIRED, QWEN_FAST_PACKED_AUDIT='1', **{FLAG: '0'})):
            with self.subTest(flags=flags):
                before = self.coordinator_run(parent, flags)
                self.assertGreater(len(before[0]), 8)
                self.assertEqual(self.coordinator_run(dflash_packed_proposal_coordinator, flags), before)

    def test_reads_covered_for_pair_traces(self):
        import dflash_packed_proposal_coordinator

        parent = self.parent('dflash_packed_proposal_coordinator.py')
        a, b, c = SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
        pair = SimpleNamespace(device_a=a, device_b=b)
        for prepared in ([a, b], [a, b, c], [a]):
            self.assertEqual(dflash_packed_proposal_coordinator.reads_covered([([0, 1], pair)], prepared),
                             parent.reads_covered([([0, 1], pair)], prepared))

    def test_the_off_path_never_imports_quad_draft(self):
        """Unset or '0', nothing on the serving path imports quad_draft."""
        script = ('import sys, os; sys.path.insert(0, %r); os.environ.pop(%r, None); '
                  'import dflash_packed_proposal_coordinator, dflash_device, draft_attention_branch, draft_mlp_branch; '
                  'print("quad_draft" in sys.modules)' % (str(HERE), FLAG))
        result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True, timeout=120,
                                cwd=str(HERE))
        if result.returncode != 0:
            self.skipTest('the serving modules do not import here: %s' % result.stderr[-200:])
        self.assertEqual(result.stdout.strip(), 'False')
        for name in ('dflash_packed_proposal_coordinator.py', 'dflash_device.py', 'draft_attention_branch.py',
                     'draft_mlp_branch.py'):
            source = (HERE / name).read_text(encoding='utf-8')
            self.assertNotRegex(source, r'(?m)^(import quad_draft|from quad_draft import)', name)

    def test_the_gate(self):
        """No quad flag (unset, '' or '0', and no sub-flag): the gate promises and reports exactly what its parent
        did - quad lines in the log or not."""
        from types import ModuleType

        import lever_n_m3native_gate as gate

        # parent_module's loader, with the __file__ the gate reads at import (its checkout-relative paths).
        try:
            result = subprocess.run(['git', 'show', '%s:scripts/ci/lever_n_m3native_gate.py' % PARENT],
                                    capture_output=True, cwd=str(HERE), timeout=60)
        except (OSError, subprocess.SubprocessError):
            self.skipTest('no git')
        if result.returncode != 0:
            self.skipTest('no git history for %s' % PARENT)
        parent = ModuleType('lever_n_m3native_gate_quad_parent')
        parent.__file__ = gate.__file__
        exec(compile(result.stdout.decode('utf-8'), 'lever_n_m3native_gate.py@%s' % PARENT, 'exec'), parent.__dict__)
        v220 = dict(REQUIRED, QWEN_FAST_PACKED_AUDIT='1', QWEN_FAST_EARLY_DRAFT='1', QWEN_FAST_GDN_AFTER_PAIRS='1',
                    QWEN_FAST_FUSED_COMMIT='1', QWEN_FAST_FUSED_COMMIT_INPLACE='1')
        logs = ('', gate_log(), gate_log(fallback=1, disabled=1, unequal=2) + chr(10) + '[PINDIAG] round b1 engaged '
                'site=quad-update cuts=C1,C2,C7,C8,M0a')
        for environ in ({}, v220, dict(v220, **{FLAG: '0'}), dict(v220, **{FLAG: ''})):
            for users in (1, 2, 4):
                for number, log in enumerate(logs):
                    with self.subTest(environ=sorted(environ.items()), users=users, log=number):
                        self.assertEqual(gate.required_flag_markers(environ, users),
                                         parent.required_flag_markers(environ, users))
                        self.assertEqual(gate.flag_marker_report(environ, users, log),
                                         parent.flag_marker_report(environ, users, log))


# ---------------------------------------------------------------------------------------------
# The arm and the gate.
# ---------------------------------------------------------------------------------------------

class ArmTests(unittest.TestCase):
    ARM = HERE / 'lever_n_m3native_run_arm.sh'
    START = '# Q4, the four-user 64-row draft pass (quad_draft.py; default off).'
    END = ('  echo "quad draft 1 sdpa ${M3NATIVE_QUAD_SDPA:-fold} conv ${M3NATIVE_QUAD_CONV:-110} audit '
           '${quad_audit:-none}"' + chr(10) + 'fi' + chr(10))
    NEEDS = dict(M3NATIVE_PACKED_PROPOSAL='1', M3NATIVE_PAIR_ROW_EXACT='1', M3NATIVE_ROUND_B1='1',
                 M3NATIVE_FUSED_COMMIT_LIVE_BANKS='1')

    def validate(self, **environ):
        import shutil

        bash = shutil.which('bash')
        if bash is None:
            self.skipTest('no bash')
        text = self.ARM.read_text(encoding='utf-8')
        start = text.index(self.START)
        end = text.index(self.END, start) + len(self.END)
        script = ('set -euo pipefail' + chr(10) + 'users="${USERS_UNDER_TEST}"' + chr(10) + text[start:end]
                  + 'echo VALID' + chr(10))
        try:
            return subprocess.run([bash, '-c', script], capture_output=True, text=True, timeout=60,
                                  env=dict(PATH=os.environ.get('PATH', ''), USERS_UNDER_TEST=environ.pop('users', '4'),
                                           **environ))
        except OSError as error:
            self.skipTest('bash unusable: %s' % error)

    def test_the_arm_refuses_what_the_gate_or_the_server_would(self):
        on = dict(self.NEEDS, M3NATIVE_QUAD_DRAFT='1')
        for environ in ({}, on, dict(on, M3NATIVE_QUAD_SDPA='pairs', M3NATIVE_QUAD_CONV='halves'),
                        dict(on, M3NATIVE_QUAD_CONV='80', M3NATIVE_QUAD_DRAFT_AUDIT='all'),
                        dict(on, M3NATIVE_QUAD_DRAFT_AUDIT='20'), self.NEEDS):
            with self.subTest(accepted=environ):
                result = self.validate(**dict(environ))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('VALID', result.stdout)
        refused = [(dict(on, M3NATIVE_QUAD_DRAFT='yes'), 'must be 1 or unset'),
                   (dict(on, M3NATIVE_QUAD_DRAFT='0'), 'must be 1 or unset'),
                   (dict(M3NATIVE_QUAD_SDPA='fold'), 'without M3NATIVE_QUAD_DRAFT=1 do nothing'),
                   (dict(M3NATIVE_QUAD_DRAFT_AUDIT='all'), 'without M3NATIVE_QUAD_DRAFT=1 do nothing'),
                   (dict(on, users='2'), 'four concurrent users only'),
                   (dict(on, users='1'), 'four concurrent users only'),
                   (dict(on, M3NATIVE_SEQUENTIAL_USERS='4'), 'four concurrent users only'),
                   (dict(on, M3NATIVE_QUAD_SDPA='dense'), 'fold or pairs'),
                   (dict(on, M3NATIVE_QUAD_CONV='64'), '110, 80 or halves'),
                   (dict(on, M3NATIVE_QUAD_DRAFT_AUDIT='0'), 'all or a positive count'),
                   (dict(on, M3NATIVE_QUAD_DRAFT_AUDIT='x'), 'all or a positive count')]
        for name in self.NEEDS:
            refused.append(({key: value for key, value in on.items() if key != name}, 'needs %s=1' % name))
        for environ, message in refused:
            with self.subTest(refused=environ):
                result = self.validate(**dict(environ))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_the_flags_cross_after_the_pair_fold_before_the_entrypoint(self):
        text = self.ARM.read_text(encoding='utf-8')
        lines = text.split(chr(10))
        expected = ['${M3NATIVE_PAIR_ROW_EXACT:+-e QWEN_FAST_PAIR_ROW_EXACT=1}',
                    '${M3NATIVE_QUAD_DRAFT:+-e QWEN_FAST_QUAD_DRAFT=1}',
                    '${M3NATIVE_QUAD_DRAFT_AUDIT:+-e QWEN_FAST_QUAD_DRAFT_AUDIT=$M3NATIVE_QUAD_DRAFT_AUDIT}',
                    '${M3NATIVE_QUAD_SDPA:+-e QWEN_FAST_QUAD_SDPA=$M3NATIVE_QUAD_SDPA}',
                    '${M3NATIVE_QUAD_CONV:+-e QWEN_FAST_QUAD_CONV=$M3NATIVE_QUAD_CONV}']
        start = next(number for number, line in enumerate(lines) if expected[0] in line)
        for offset, line in enumerate(expected):
            with self.subTest(line=line):
                self.assertEqual(text.count(line), 1)
                self.assertEqual(lines[start + offset].strip(), line + ' ' + chr(92))
        self.assertLess(text.index(expected[-1]), text.index('--entrypoint python3'))
        self.assertLess(text.index(self.START), text.index('docker run --rm --name "$name"'))
        self.assertNotIn(b'\r\n', self.ARM.read_bytes())

    def test_unset_nothing_crosses(self):
        import shutil

        bash = shutil.which('bash')
        if bash is None:
            self.skipTest('no bash')
        script = ('printf "%s|" ${M3NATIVE_QUAD_DRAFT:+-e QWEN_FAST_QUAD_DRAFT=1} '
                  '${M3NATIVE_QUAD_DRAFT_AUDIT:+-e QWEN_FAST_QUAD_DRAFT_AUDIT=$M3NATIVE_QUAD_DRAFT_AUDIT} '
                  '${M3NATIVE_QUAD_SDPA:+-e QWEN_FAST_QUAD_SDPA=$M3NATIVE_QUAD_SDPA} '
                  '${M3NATIVE_QUAD_CONV:+-e QWEN_FAST_QUAD_CONV=$M3NATIVE_QUAD_CONV}' + chr(10))
        try:
            unset = subprocess.run([bash, '-c', script], env=dict(PATH=os.environ.get('PATH', '')),
                                   capture_output=True, text=True, timeout=60)
            every = subprocess.run([bash, '-c', script], env=dict(
                PATH=os.environ.get('PATH', ''), M3NATIVE_QUAD_DRAFT='1', M3NATIVE_QUAD_DRAFT_AUDIT='all',
                M3NATIVE_QUAD_SDPA='fold', M3NATIVE_QUAD_CONV='110'), capture_output=True, text=True, timeout=60)
        except OSError as error:
            self.skipTest('bash unusable: %s' % error)
        self.assertEqual(unset.stdout.strip('|'), '')
        self.assertEqual(every.stdout, '-e|QWEN_FAST_QUAD_DRAFT=1|-e|QWEN_FAST_QUAD_DRAFT_AUDIT=all|-e|'
                                       'QWEN_FAST_QUAD_SDPA=fold|-e|QWEN_FAST_QUAD_CONV=110|')

    def test_unset_the_block_is_inert(self):
        """The flag off, the arm is its parent's: with no M3NATIVE_QUAD_* set the Q4 block prints nothing and refuses
        nothing at any user count, it sets only its own quad_* names and nothing else in the arm reads them, and its
        four -e lines expand to nothing (test_unset_nothing_crosses)."""
        text = self.ARM.read_text(encoding='utf-8')
        start = text.index(self.START)
        end = text.index(self.END, start) + len(self.END)
        block = text[start:end]
        assigned = set(re.findall(r'(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)=', block))
        assigned |= set(re.findall(r'(?m)^\s*for ([A-Za-z_][A-Za-z0-9_]*) in ', block))
        self.assertEqual(sorted(assigned), ['quad_audit', 'quad_need', 'quad_value'])
        self.assertEqual(re.findall(r'\bquad_(?:audit|need|value)\b', text[:start] + text[end:]), [])
        for users in ('4', '2', '1'):
            for environ in ({}, self.NEEDS):
                with self.subTest(users=users, environ=sorted(environ)):
                    result = self.validate(users=users, **dict(environ))
                    self.assertEqual((result.returncode, result.stdout, result.stderr), (0, 'VALID' + chr(10), ''))


def gate_log(*, rounds=12, four=12, markers=1, audits=None, fallback=0, disabled=0, unequal=0):
    lines = ['2026-09-25 | INFO | [PINDIAG] quad draft engaged slots=[0,1,2,3] heads=64/16 rows=64 sdpa=fold conv=110'
             ] * markers
    for number in range(1, four + 1):
        lines.append('[PACKED-SELECT] round=%d pairs=[[0, 1, 2, 3]] users=4 calls=1 collect_ms=1.30 select_ms=2.00'
                     % number)
    for number in range(1, rounds + 1):
        lines.append('[QUAD-DRAFT] round=%d built=%d ms=%.1f' % (number, int(number == 1), 3.0))
    for number in range(1, (rounds if audits is None else audits) + 1):
        lines.append('[QUAD-AUDIT] round=%d equal=%d stage=%s users=4 checks=52'
                     % (number, int(number > unequal), 'all' if number > unequal else 'features:u2'))
    lines.extend(['[QUAD-DRAFT] fallback round=9 reason=RuntimeError:x'] * fallback)
    lines.extend(['[PINDIAG] quad draft disabled round=9 failures=2 reason=consecutive_failures=2'] * disabled)
    return chr(10).join(lines)


class GateTests(unittest.TestCase):
    ON = dict(REQUIRED, QWEN_FAST_PACKED_AUDIT='1', **{FLAG: '1'})

    def problems(self, environ, log, users=4):
        import lever_n_m3native_gate as gate

        report = gate.flag_marker_report(environ, users, log)
        return [entry for entry in report['missing'] if 'QUAD' in entry], report

    def test_the_names_match_the_module(self):
        import lever_n_m3native_gate as gate

        self.assertEqual((gate.QUAD_DRAFT_FLAG, gate.QUAD_SDPA_FLAG, gate.QUAD_CONV_FLAG, gate.QUAD_AUDIT_FLAG),
                         (quad_draft.FLAG, quad_draft.SDPA_FLAG, quad_draft.CONV_FLAG, quad_draft.AUDIT_FLAG))
        self.assertEqual((gate.QUAD_MARKER, gate.QUAD_DISABLED_MARKER), (quad_draft.MARKER, quad_draft.DISABLED_MARKER))
        self.assertTrue(quad_draft.FALLBACK_LINE.startswith(gate.QUAD_FALLBACK_MARKER))
        self.assertEqual(gate.required_flag_markers(self.ON, 4)[FLAG], [quad_draft.MARKER])
        self.assertNotIn(FLAG, gate.required_flag_markers(self.ON, 2))
        self.assertNotIn(FLAG, gate.required_flag_markers({FLAG: '0'}, 4))

    def test_the_lines_the_module_logs_parse(self):
        import lever_n_m3native_gate as gate

        round_line = quad_draft.ROUND_LINE.format(round=4, built=0, ms='3.2')
        self.assertIsNotNone(gate.QUAD_ROUND_LINE.search(round_line))
        audit = quad_draft.AUDIT_LINE % (4, 1, 'all', 4, 52)
        self.assertIsNotNone(gate.QUAD_AUDIT_LINE.search(audit))
        mask = quad_draft.MASK_AUDIT_LINE % (4, 1, 0, 1)
        self.assertIsNotNone(gate.QUAD_MASK_LINE.search(mask))
        self.assertIsNone(gate.QUAD_ROUND_LINE.search(mask), 'a mask line is not a round')
        from dflash_packed_proposal_coordinator import SELECT_LINE

        select = SELECT_LINE.format(round=4, pairs=[[0, 1, 2, 3]], users=4, calls=1, collect_ms='1.3', select_ms='2.0')
        self.assertIsNotNone(gate.FOUR_USER_SELECT.search(select))
        self.assertIsNone(gate.FOUR_USER_SELECT.search(select.replace('users=4', 'users=3')))

    def test_a_clean_quad_arm_passes(self):
        problems, report = self.problems(dict(self.ON, **{quad_draft.AUDIT_FLAG: 'all'}), gate_log(rounds=24, four=24))
        self.assertEqual(problems, [])
        self.assertEqual(report['quad_draft']['rounds'], 24)
        self.assertEqual(report['quad_draft']['builds'], 1)
        self.assertEqual(report['quad_draft']['audits_equal'], 24)
        problems, _ = self.problems(self.ON, gate_log(audits=0))
        self.assertEqual(problems, [], 'a timed arm has no audit')

    def test_each_failure(self):
        cases = ((dict(markers=0), 'QWEN_FAST_QUAD_DRAFT: [PINDIAG] quad draft engaged'),
                 (dict(markers=2), 'logged 2 times, not once'),
                 (dict(fallback=1), 'fell back to the pairs'),
                 (dict(disabled=1), 'the quad gave up'),
                 (dict(rounds=0, audits=0), 'no quad round in 12 four-user round(s)'),
                 (dict(rounds=10, four=12, audits=10), '10 quad rounds in 12 four-user rounds'),
                 (dict(audits=11), '11 audit lines for 12 audited quad rounds'),
                 (dict(unequal=1), '1 audit line(s) not equal'))
        for log, message in cases:
            with self.subTest(message=message):
                problems, _ = self.problems(dict(self.ON, **{quad_draft.AUDIT_FLAG: 'all'}), gate_log(**log))
                self.assertTrue(any(message in problem for problem in problems), problems)

    def test_the_audit_of_n_rounds_and_small_runs(self):
        problems, _ = self.problems(dict(self.ON, **{quad_draft.AUDIT_FLAG: '5'}), gate_log(audits=5))
        self.assertEqual(problems, [])
        problems, _ = self.problems(dict(self.ON, **{quad_draft.AUDIT_FLAG: '30'}), gate_log(rounds=24, four=24))
        self.assertEqual(problems, [], 'N above the quad rounds: every quad round audited, at least the floor')
        problems, _ = self.problems(self.ON, gate_log(rounds=4, four=5, audits=0))
        self.assertEqual(problems, [], 'under QUAD_FLOOR_ROUNDS four-user rounds only engagement is required')

    def test_an_audit_cannot_pass_on_too_few_or_uncounted_rounds(self):
        import lever_n_m3native_gate as gate

        audited = dict(self.ON, **{quad_draft.AUDIT_FLAG: 'all'})
        problems, _ = self.problems(audited, gate_log(rounds=3, four=3))
        self.assertEqual(problems, ['QWEN_FAST_QUAD_DRAFT_AUDIT: 3 audited quad round(s), under the %d an audit arm '
                                    'needs' % gate.QUAD_AUDIT_FLOOR])
        problems, _ = self.problems(dict(self.ON, **{quad_draft.AUDIT_FLAG: '30'}), gate_log(rounds=12, four=12))
        self.assertTrue(any('12 audited quad round(s), under the 20' in problem for problem in problems), problems)
        uncounted = dict(audited)
        del uncounted['QWEN_FAST_PACKED_AUDIT']
        log = chr(10).join(line for line in gate_log(rounds=24, four=24, audits=0).splitlines()
                           if not line.startswith(('[QUAD-DRAFT] round=', '[PACKED-SELECT]')))
        problems, _ = self.problems(uncounted, log)
        self.assertEqual(problems, ['QWEN_FAST_QUAD_DRAFT needs QWEN_FAST_PACKED_AUDIT=1: its [QUAD-DRAFT] round lines '
                                    'and the four-user [PACKED-SELECT] rounds they are counted against are logged only '
                                    'under it',
                                    'QWEN_FAST_QUAD_DRAFT_AUDIT needs QWEN_FAST_PACKED_AUDIT=1: its [QUAD-DRAFT] round '
                                    'lines count the audited rounds'],
                         'no round lines and no audit lines must not read as zero of zero')

    def test_the_rounds_it_served_cannot_pass_uncounted(self):
        """Without QWEN_FAST_PACKED_AUDIT neither the round lines nor the four-user rounds are logged, so the share
        check would pass on nothing - a quad that engaged once and then quietly served pairs included."""
        uncounted = dict(self.ON)
        del uncounted['QWEN_FAST_PACKED_AUDIT']
        log = chr(10).join(line for line in gate_log(audits=0).splitlines()
                           if not line.startswith(('[QUAD-DRAFT] round=', '[PACKED-SELECT]')))
        problems, _ = self.problems(uncounted, log)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn('QWEN_FAST_QUAD_DRAFT needs QWEN_FAST_PACKED_AUDIT=1', problems[0])

    def test_the_marker_names_the_modes_the_arm_asked_for(self):
        """The marker's sdpa=/conv= (and slots, heads, rows) must be what the arm asked for, defaults fold and 110:
        an image with other defaults, or a sub-flag that never reached the server, is not the arm it claims."""
        marker = '[PINDIAG] quad draft engaged slots=[0,1,2,3] heads=64/16 rows=64 sdpa=fold conv=110'

        def log(line):
            return gate_log(audits=0).replace(marker, line)

        pairs = '[PINDIAG] quad draft engaged slots=[0,1,2,3] heads=32/8x2 rows=64 sdpa=pairs conv=80'
        for environ, text, wrong in (
                (self.ON, log(marker), None),
                (dict(self.ON, **{quad_draft.SDPA_FLAG: 'fold', quad_draft.CONV_FLAG: '110'}), log(marker), None),
                (dict(self.ON, **{quad_draft.SDPA_FLAG: 'pairs', quad_draft.CONV_FLAG: '80'}), log(pairs), None),
                (dict(self.ON, **{quad_draft.CONV_FLAG: '80'}), log(marker), 'the marker reads conv=110, not the conv=80'),
                (dict(self.ON, **{quad_draft.SDPA_FLAG: 'pairs'}), log(marker),
                 'the marker reads heads=64/16 sdpa=fold, not the heads=32/8x2 sdpa=pairs'),
                (self.ON, log(pairs), 'the marker reads heads=32/8x2 sdpa=pairs conv=80, not the heads=64/16 '
                                      'sdpa=fold conv=110'),
                (self.ON, log(marker.replace('slots=[0,1,2,3]', 'slots=[0,1,3,2]')),
                 'the marker reads slots=0,1,3,2, not the slots=0,1,2,3'),
                (self.ON, log('[PINDIAG] quad draft engaged'), 'does not name its slots')):
            with self.subTest(environ=sorted(environ.items()), wrong=wrong):
                problems, report = self.problems(environ, text)
                if wrong is None:
                    self.assertEqual(problems, [])
                    self.assertEqual(report['quad_draft']['engaged']['sdpa'], environ.get(quad_draft.SDPA_FLAG, 'fold'))
                else:
                    self.assertEqual(len(problems), 1, problems)
                    self.assertIn(wrong, problems[0])
        # quad_draft.note's own line parses as the modes it was given.
        lines = []
        with patch.object(quad_draft, '_NOTED', []):
            quad_draft.note([0, 1, 2, 3], 'pairs', 'halves', log=lines.append)
        problems, _ = self.problems(dict(self.ON, **{quad_draft.SDPA_FLAG: 'pairs', quad_draft.CONV_FLAG: 'halves'}),
                                    log(lines[0]))
        self.assertEqual(problems, [])

    def test_requirements_and_orphan_sub_flags(self):
        problems, _ = self.problems({FLAG: '1', 'QWEN_FAST_PACKED_AUDIT': '1'}, gate_log(audits=0))
        self.assertEqual(sorted(problem for problem in problems if 'needs' in problem),
                         sorted('QWEN_FAST_QUAD_DRAFT needs %s=1' % name for name in quad_draft.REQUIRED_FLAGS))
        problems, _ = self.problems({quad_draft.CONV_FLAG: '80'}, '')
        self.assertEqual(len(problems), 1)
        self.assertIn('without QWEN_FAST_QUAD_DRAFT=1 do nothing', problems[0])
        problems, report = self.problems(self.ON, gate_log(), users=2)
        self.assertTrue(any('four users only' in problem for problem in problems))
        import lever_n_m3native_gate as gate

        self.assertNotIn('quad_draft', gate.flag_marker_report({}, 4, gate_log()))


# ---------------------------------------------------------------------------------------------
# Shipping and the pins.
# ---------------------------------------------------------------------------------------------

class ShippingTests(unittest.TestCase):
    def test_the_module_and_its_kernel_reach_the_image_through_both_copy_lists(self):
        from test_serving_image_copy_closure import context_modules, dockerfile_modules, dockerfile_text

        for name in ('quad_draft.py', 'quad_conv_io.cpp', 'dflash_packed_proposal_coordinator.py', 'dflash_device.py',
                     'draft_attention_branch.py', 'draft_mlp_branch.py', 'pair_row_exact.py', 'fused_commit.py',
                     'draft_convolution_fused.py', 'memory_ledger.py'):
            with self.subTest(module=name):
                self.assertIn(name, dockerfile_modules(dockerfile_text()))
                self.assertIn(name, context_modules())

    def test_the_cpu_suite_runs_this_file(self):
        self.assertRegex(CPU_WORKFLOW.read_text(encoding='utf-8'), r'python -B -m unittest [^\n]*\btest_quad_draft\b')

    def git_changed(self, names):
        try:
            result = subprocess.run(['git', 'diff', '--name-only', PARENT, '--', *names], capture_output=True,
                                    cwd=str(HERE), timeout=60, text=True)
        except (OSError, subprocess.SubprocessError):
            self.skipTest('no git')
        if result.returncode != 0:
            self.skipTest('no git history for %s' % PARENT)
        return result.stdout.strip()

    def test_the_t16_admissions_hashed_sources_are_untouched(self):
        from dflash_t16_native_attention_gate import SOURCES

        self.assertEqual(self.git_changed(SOURCES), '')

    def test_the_bundle_only_helpers_and_the_served_kernels_are_untouched(self):
        """Plan section 4.3: in neither copy list, so never edited; the served conv kernels stay as served."""
        self.assertEqual(self.git_changed(['draft_shared_head.py', 'feature_collective.py', 'draft_head_layout.py',
                                           'draft_kv_projection.py', 'draft_selector.py', 'draft_mlp.py',
                                           'draft_convolution_fused_compute.cpp', 'dflash_proposal_inputs.py',
                                           'draft_head_preparation.py', 'draft_convolution_fused_io.cpp',
                                           'draft_convolution_fused.py', 'draft_convolution.py', 'pair_row_exact.py',
                                           'dflash_proposal_trace.py', 'fused_commit.py', 'serving_bundle.py']), '')

    def test_the_serving_bundle_inventorys_eight_files_are_untouched(self):
        """Plan section 4.1: serving_bundle.package's critical staged-source inventory."""
        self.assertEqual(self.git_changed(['model_batch.py', 'verifier_engine.py', 'dflash_combined_request.py',
                                           'draft_kv_slide.cpp', 'draft_kv_slide_gate.py', 'frozen_combined_runtime.py',
                                           'target_t16_attention_gate.py', 'dflash_t16_native_scope.py']), '')

    def test_every_touched_file_is_lf(self):
        for name in ('quad_draft.py', 'quad_conv_io.cpp', 'test_quad_draft.py', 'dflash_device.py',
                     'draft_attention_branch.py', 'draft_mlp_branch.py', 'dflash_packed_proposal_coordinator.py',
                     'lever_n_m3native_gate.py', 'lever_n_m3native_run_arm.sh', 'test_dflash_round_b1.py',
                     'test_m3native_arm_env.py'):
            with self.subTest(name=name):
                self.assertNotIn(b'\r', (HERE / name).read_bytes())
        for path in (ROOT / 'docker' / 'qwen-fast-serving.Dockerfile', CPU_WORKFLOW,
                     ROOT / '.github' / 'workflows' / 'qwen-fast-serving-image.yml'):
            with self.subTest(name=path.name):
                self.assertNotIn(b'\r', path.read_bytes())


if __name__ == '__main__':
    unittest.main()
