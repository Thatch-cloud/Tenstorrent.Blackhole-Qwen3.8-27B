"""QWEN_FAST_PAIR_ROW_EXACT: the packed pair's draft SDPA folded (pair_row_exact.py).

  - the flag: '0' or '1' only, and it engages only at the (2048, 2048) T16 pair;
  - the fold's index mapping. Q head 8h+4u+j is head 4h+j with user u's rows at 0-15. KV head 2h+u is user
    u's segment of head h. The unfold puts user a's rows at 0-15 and user b's at 16-31 of each original head;
  - on a CPU stand-in for the ttnn surface (a GQA SDPA computed head by head in float32), the folded pair's
    rows are BIT-identical to each user's own single-user SDPA, built as the single-user trace builds it: its
    own live block with pad rows, pad query rows and the served mask. So every folded work unit is the
    single-user unit. Today's packed call equals them in exact arithmetic only (the residue M1 names is a
    device effect the card-B probe measures);
  - the mask is the served single-user mask, which is also each user's own block of the packed mask, and it
    passes the single-user validation (row 1's RoPE is the served single-user tables:
    test_dflash_batched_mask.PackedRopeTests);
  - folded_sdpa's arguments are draft_sdpa's, argument for argument;
  - the wiring:
    - the attention branch folds only under row_exact and only on the pair geometry;
    - execute_proposal forwards row_exact only when it is set;
    - under the flag the pair trace folds a (2048, 2048) bucket: the single mask is uploaded, refreshed and
      audited, row_exact reaches the build, the capture and every replay, the marker is logged once, and
      nothing else differs from the flag-off run;
    - propose_packed(row_exact=True) behaves the same way;
  - the flag off (unset or '0'): each module this change touches is its PARENT (58415c6f) call for call;
  - the module reaches the image (both copy lists) and the CPU suite runs this file.

    py -3.11 -B -m unittest test_pair_row_exact      (from scripts/ci)
"""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import pair_row_exact  # noqa: E402
from pair_row_exact import (fold_attention, fold_keys, fold_mask, fold_query, folded_sdpa, unfold_output,  # noqa: E402
                            validate_fold)

# This change's parent: every module it touches is that commit's there.
PARENT = '58415c6f'
FLAG = 'QWEN_FAST_PAIR_ROW_EXACT'
CPU_WORKFLOW = ROOT / '.github' / 'workflows' / 'qwen-integration-cpu.yml'


def clean_environment(**flags):
    """No QWEN_FAST_* flag but the ones given."""
    environment = {name: value for name, value in os.environ.items() if not name.startswith('QWEN_FAST_')}
    environment.update(flags)
    return patch.dict(os.environ, environment, clear=True)


def parent_module(relative, name):
    from test_dflash_proposal_trace import pinned_module

    return pinned_module(relative, name, commit=PARENT)


def same_bits(left, right):
    return (tuple(left.shape) == tuple(right.shape) and left.dtype == right.dtype
            and torch.equal(left.contiguous().view(torch.int16), right.contiguous().view(torch.int16)))


# ---------------------------------------------------------------------------------------------
# A CPU stand-in for the ttnn surface the fold uses.
# ---------------------------------------------------------------------------------------------

class Device:
    """A device tensor: a torch tensor plus the attributes validate_fold reads."""

    def __init__(self, value, dtype='bf16', layout='tile', memory='dram'):
        self.value, self.dtype, self.layout, self.memory = value, dtype, layout, memory
        self.shape = tuple(value.shape)

    def memory_config(self):
        return self.memory


class TorchOps:
    """slice, concat and reshape as ttnn does them on logical shapes (reshape is the row-major logical reshape a
    tiled view keeps), and a GQA SDPA computed head by head in float32: every head is the same computation on a
    (32, 128) query, (S, 128) keys and values and a (32, S) mask, so a head whose inputs are bit-identical gives
    bit-identical rows. Every call is logged in `calls`."""

    bfloat16, float32 = 'bf16', 'fp32'
    TILE_LAYOUT, DRAM_MEMORY_CONFIG = 'tile', 'dram'
    MathFidelity = SimpleNamespace(HiFi4='hifi4')

    def __init__(self):
        self.calls = []
        self.transformer = SimpleNamespace(scaled_dot_product_attention=self.sdpa)

    def WormholeComputeKernelConfig(self, **options):
        return ('kernel', tuple(sorted(options.items())))

    def SDPAProgramConfig(self, **options):
        return ('program', tuple(sorted(options.items())))

    def slice(self, tensor, start, end):
        self.calls.append(('slice', tensor.shape, tuple(start), tuple(end)))
        index = tuple(slice(low, high) for low, high in zip(start, end))
        return Device(tensor.value[index].clone(), tensor.dtype)

    def concat(self, parts, dim, memory_config=None):
        self.calls.append(('concat', tuple(part.shape for part in parts), dim))
        return Device(torch.cat([part.value for part in parts], dim=dim), parts[0].dtype)

    def reshape(self, tensor, shape):
        self.calls.append(('reshape', tensor.shape, tuple(shape)))
        return Device(tensor.value.reshape(shape), tensor.dtype)

    def sdpa(self, query, key, value, *, attn_mask, is_causal, scale, program_config, compute_kernel_config,
             memory_config):
        self.calls.append(('sdpa', query.shape, key.shape, attn_mask.shape))
        q, k, v, mask = query.value, key.value, value.value, attn_mask.value
        heads, kv_heads = q.shape[1], k.shape[1]
        rows = []
        for head in range(heads):
            kv = head // (heads // kv_heads)
            head_mask = mask[0, 0 if mask.shape[1] == 1 else head].float()
            scores = q[0, head].float() @ k[0, kv].float().T * scale + head_mask
            rows.append(torch.softmax(scores, dim=-1) @ v[0, kv].float())
        return Device(torch.stack(rows)[None].bfloat16())


def keep(owned):
    def retain(value):
        owned.append(value)
        return value
    return retain


def coded(heads, rows, width=128, *, base=0):
    """(1, heads, rows, width) float32 whose every element names its (head, row, column): exact in float32."""
    head = torch.arange(heads).view(1, heads, 1, 1) * 100000
    row = torch.arange(rows).view(1, 1, rows, 1) * 1000
    column = torch.arange(width).view(1, 1, 1, width)
    return (base + head + row + column).float()


class PairFixture:
    """Two users' operands built twice: as the pair's branch assembles them (the shared 32-row live block,
    key_value_plan's pieces) and as each user's single-user trace does (its own live block, whose rows 16-31
    are the pad proposal rows, its own pad query rows)."""

    def __init__(self, seed=0, *, scale=1.0):
        from dflash_batched_mask import key_value_plan

        generator = torch.Generator().manual_seed(seed)

        def normal(*shape, factor=1.0):
            return (torch.randn(*shape, generator=generator) * factor).bfloat16()

        self.cache = {user: {name: normal(1, 4, 2048, 128) for name in 'kv'} for user in 'ab'}
        self.live = {user: {name: normal(1, 4, 16, 128) for name in 'kv'} for user in 'ab'}
        self.live_pad = {user: {name: normal(1, 4, 16, 128) for name in 'kv'} for user in 'ab'}
        self.query = {user: normal(1, 16, 16, 128, factor=scale) for user in 'ab'}
        self.query_pad = {user: normal(1, 16, 16, 128, factor=scale) for user in 'ab'}
        # The pair: one 32-row block, user a's rows then user b's.
        self.block = {name: torch.cat([self.live['a'][name], self.live['b'][name]], dim=2) for name in 'kv'}
        self.pair_query = torch.cat([self.query['a'], self.query['b']], dim=2)
        plan, spans, key_rows = key_value_plan([2048, 2048], 16)
        self.pair_keys = {}
        for name in 'kv':
            pieces = []
            for part in plan:
                if part['kind'] == 'cached':
                    pieces.append(self.cache['ab'[part['user']]][name])
                    continue
                start = part['source'].start if part['kind'] == 'live' else 0
                pieces.append(self.block[name][:, :, start:start + part['rows']])
            self.pair_keys[name] = torch.cat(pieces, dim=2)
        assert key_rows == 4160 and self.pair_keys['k'].shape == (1, 4, 4160, 128)

    def single(self, user):
        """(query, keys, values) of `user`'s single-user trace: K = [cache | own live block of 32]."""
        keys = {name: torch.cat([self.cache[user][name], self.live[user][name], self.live_pad[user][name]], dim=2)
                for name in 'kv'}
        return torch.cat([self.query[user], self.query_pad[user]], dim=2), keys['k'], keys['v']


# ---------------------------------------------------------------------------------------------
# The flag.
# ---------------------------------------------------------------------------------------------

class FlagTests(unittest.TestCase):
    def test_zero_or_one_only_and_off_by_default(self):
        self.assertFalse(pair_row_exact.enabled({}))
        self.assertFalse(pair_row_exact.enabled({FLAG: '0'}))
        self.assertTrue(pair_row_exact.enabled({FLAG: '1'}))
        for value in ('', 'yes', 'true', '2', ' 1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pair_row_exact.enabled({FLAG: value})
        with clean_environment():
            self.assertFalse(pair_row_exact.enabled())
        with clean_environment(**{FLAG: '1'}):
            self.assertTrue(pair_row_exact.enabled())

    def test_it_engages_at_the_2048_pair_of_t16_users_only(self):
        on = {FLAG: '1'}
        self.assertTrue(pair_row_exact.engages((2048, 2048), 16, on))
        self.assertTrue(pair_row_exact.engages([2048, 2048], 16, on))
        for contexts, rows in (((2048, 1024), 16), ((1024, 1024), 16), ((500, 500), 16), ((2048,), 16),
                               ((2048, 2048, 2048), 16), ((2048, 2048), 8), ((2048, 2048), 32), ((2048, 2048), '16')):
            with self.subTest(contexts=contexts, rows=rows):
                self.assertFalse(pair_row_exact.engages(contexts, rows, on))
                with self.assertRaises(ValueError):
                    pair_row_exact.require_fold(contexts, rows)
        self.assertFalse(pair_row_exact.engages((2048, 2048), 16, {}))
        pair_row_exact.require_fold((2048, 2048), 16)

    def test_the_marker_is_logged_once_per_process(self):
        lines = []
        with patch.object(pair_row_exact, '_NOTED', []):
            self.assertTrue(pair_row_exact.note([0, 1], (2048, 2048), log=lines.append))
            self.assertFalse(pair_row_exact.note([2, 3], (2048, 2048), log=lines.append))
        self.assertEqual(lines, ['[PINDIAG] pair row exact engaged pair=[0,1] context=2048,2048 heads=32/8 keys=2080'])
        self.assertTrue(lines[0].startswith(pair_row_exact.MARKER))


# ---------------------------------------------------------------------------------------------
# The fold's index mapping.
# ---------------------------------------------------------------------------------------------

class FoldMappingTests(unittest.TestCase):
    def test_folded_query_head_8h_4u_j_is_head_4h_j_with_user_u_at_rows_0_to_15(self):
        ops, owned = TorchOps(), []
        query = coded(16, 32)
        folded = fold_query(ops, Device(query), keep(owned)).value
        self.assertEqual(tuple(folded.shape), (1, 32, 32, 128))
        for h in range(4):
            for u in range(2):
                for j in range(4):
                    source = query[0, 4 * h + j]
                    expected = source if u == 0 else torch.cat([source[16:], source[:16]])
                    self.assertTrue(torch.equal(folded[0, 8 * h + 4 * u + j], expected), (h, u, j))
                    # and GQA group 4 sends it to KV head 2h+u
                    self.assertEqual((8 * h + 4 * u + j) // (32 // 8), 2 * h + u)

    def test_folded_kv_head_2h_u_is_user_us_segment_of_head_h(self):
        ops, owned = TorchOps(), []
        keys = coded(4, 4160)
        folded = fold_keys(ops, Device(keys), keep(owned)).value
        self.assertEqual(tuple(folded.shape), (1, 8, 2080, 128))
        for h in range(4):
            for u in range(2):
                self.assertTrue(torch.equal(folded[0, 2 * h + u], keys[0, h, 2080 * u:2080 * (u + 1)]))
        self.assertEqual(ops.calls, [('reshape', (1, 4, 4160, 128), (1, 8, 2080, 128))], 'a view, no copy')

    def test_the_unfold_takes_each_users_rows_0_to_15_back_to_its_packed_rows(self):
        ops, owned = TorchOps(), []
        output = coded(32, 32)
        unfolded = unfold_output(ops, Device(output), keep(owned)).value
        self.assertEqual(tuple(unfolded.shape), (1, 16, 32, 128))
        for h in range(4):
            for j in range(4):
                self.assertTrue(torch.equal(unfolded[0, 4 * h + j, :16], output[0, 8 * h + j, :16]))
                self.assertTrue(torch.equal(unfolded[0, 4 * h + j, 16:], output[0, 8 * h + 4 + j, :16]))

    def test_fold_then_unfold_of_an_identity_attention_is_the_packed_query(self):
        """An 'attention' that returns its query unchanged: fold and unfold must give back the packed query, row
        for row. That holds only because the u=1 heads carry user b at rows 0-15."""
        ops, owned = TorchOps(), []
        query = coded(16, 32)
        folded = fold_query(ops, Device(query), keep(owned))
        self.assertTrue(torch.equal(unfold_output(ops, folded, keep(owned)).value, query))

    def test_the_op_count_is_seven_copies_plus_views(self):
        ops, owned = TorchOps(), []
        fixture = PairFixture()
        mask = Device(fold_mask((2048, 2048), 16))
        fold_attention(ops, Device(fixture.pair_query), Device(fixture.pair_keys['k']), Device(fixture.pair_keys['v']),
                       mask, keep(owned), mask_validated=True)
        kinds = [call[0] for call in ops.calls]
        self.assertEqual(kinds.count('sdpa'), 1)
        self.assertEqual(kinds.count('slice') + kinds.count('concat'), 7)
        self.assertEqual(kinds.count('reshape'), 8)
        sdpa = next(call for call in ops.calls if call[0] == 'sdpa')
        self.assertEqual(sdpa[1:], ((1, 32, 32, 128), (1, 8, 2080, 128), (1, 1, 32, 2080)))
        self.assertEqual(len(owned), 16, 'every intermediate is retained for the caller to release')


# ---------------------------------------------------------------------------------------------
# Numerics on the CPU stand-in.
# ---------------------------------------------------------------------------------------------

class FoldEqualsAloneTests(unittest.TestCase):
    """Every folded work unit is the single-user unit, so its rows are bit-identical to drafting alone."""

    def run_fold(self, fixture):
        ops, owned = TorchOps(), []
        mask = fold_mask((2048, 2048), 16)
        out = fold_attention(ops, Device(fixture.pair_query), Device(fixture.pair_keys['k']),
                             Device(fixture.pair_keys['v']), Device(mask), keep(owned), mask_validated=True)
        return out.value, mask

    def alone(self, fixture, user, mask):
        query, keys, values = fixture.single(user)
        return TorchOps().sdpa(Device(query), Device(keys), Device(values), attn_mask=Device(mask), is_causal=False,
                               scale=128 ** -0.5, program_config=None, compute_kernel_config=None,
                               memory_config='dram').value

    def test_row_0_and_row_1_are_each_users_single_user_rows_bit_for_bit(self):
        for seed, scale in ((0, 1.0), (1, 4.0), (2, 0.25)):
            with self.subTest(seed=seed, scale=scale):
                fixture = PairFixture(seed, scale=scale)
                folded, mask = self.run_fold(fixture)
                self.assertTrue(same_bits(folded[:, :, :16], self.alone(fixture, 'a', mask)[:, :, :16]), 'row 0')
                self.assertTrue(same_bits(folded[:, :, 16:], self.alone(fixture, 'b', mask)[:, :, :16]), 'row 1')

    def test_a_hundredfold_partner_changes_nothing(self):
        fixture = PairFixture(3)
        folded, mask = self.run_fold(fixture)
        loud = PairFixture(3)
        for name in 'kv':
            # user a's keys and values reach segment b only as masked pad rows, and the other way round
            loud.block[name][:, :, :16] = (loud.block[name][:, :, :16].float() * 100).bfloat16()
            loud.pair_keys[name][:, :, 2064:2080] = loud.block[name][:, :, :16]
            loud.pair_keys[name][:, :, 4144:4160] = loud.block[name][:, :, :16]
        louder, _ = self.run_fold(loud)
        self.assertTrue(same_bits(louder[:, :, 16:], folded[:, :, 16:]), "row 1 never reads user a's keys")

    def test_todays_packed_call_is_equal_in_exact_arithmetic_only(self):
        from dflash_batched_mask import batched_attention_mask

        fixture = PairFixture(4)
        packed = batched_attention_mask([2048, 2048], 16)
        out = TorchOps().sdpa(Device(fixture.pair_query), Device(fixture.pair_keys['k']), Device(fixture.pair_keys['v']),
                              attn_mask=Device(packed), is_causal=False, scale=128 ** -0.5, program_config=None,
                              compute_kernel_config=None, memory_config='dram').value
        mask = fold_mask((2048, 2048), 16)
        for user, rows in (('a', slice(0, 16)), ('b', slice(16, 32))):
            alone = self.alone(fixture, user, mask)[:, :, :16].float()
            torch.testing.assert_close(out[:, :, rows].float(), alone, rtol=1e-2, atol=1e-2)

    def test_the_operands_are_validated(self):
        fixture, ops, owned = PairFixture(5), TorchOps(), []
        query, keys, values = (Device(fixture.pair_query), Device(fixture.pair_keys['k']),
                               Device(fixture.pair_keys['v']))
        mask = Device(fold_mask((2048, 2048), 16))
        from dflash_batched_mask import batched_attention_mask

        with self.assertRaises(ValueError):
            fold_attention(ops, query, keys, values, mask, keep(owned))
        cases = [
            (Device(fixture.pair_query[:, :, :16]), keys, values, mask),
            (query, Device(fixture.pair_keys['k'][:, :, :2080]), values, mask),
            (query, keys, Device(fixture.pair_keys['v'][:, :3]), mask),
            (query, keys, values, Device(batched_attention_mask([2048, 2048], 16))),
            (Device(fixture.pair_query, dtype='fp32'), keys, values, mask),
            (query, Device(fixture.pair_keys['k'], layout='row'), values, mask),
            (query, keys, values, Device(mask.value, memory='l1')),
        ]
        for index, operands in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                validate_fold(ops, *operands)
        self.assertEqual(ops.calls, [])


# ---------------------------------------------------------------------------------------------
# The mask and the wrapper.
# ---------------------------------------------------------------------------------------------

class MaskTests(unittest.TestCase):
    def test_the_fold_mask_is_the_served_single_user_mask_and_each_users_packed_block(self):
        from dflash_batched_mask import batched_attention_mask
        from dflash_proposal_inputs import proposal_inputs
        from dflash_t16_native_attention import validate_mask

        mask = fold_mask((2048, 2048), 16)
        self.assertEqual(tuple(mask.shape), (1, 1, 32, 2080))
        for position in (2048, 32768, 131072 + 17):
            self.assertTrue(same_bits(mask, proposal_inputs(0, position, 2048, 16, 2048)['mask']), position)
        packed = batched_attention_mask([2048, 2048], 16)
        self.assertTrue(same_bits(packed[:, :, :16, :2080], mask[:, :, :16]), "user a's block")
        self.assertTrue(same_bits(packed[:, :, 16:, 2080:], mask[:, :, :16]), "user b's block")
        self.assertTrue(bool(torch.isneginf(packed[:, :, 16:, :2080]).all()), "row 1's 65 leading chunks are masked")
        validate_mask(mask)
        with self.assertRaises(ValueError):
            fold_mask((2048, 1024), 16)


class WrapperTests(unittest.TestCase):
    def recording(self):
        operations = SimpleNamespace(bfloat16='bf16', DRAM_MEMORY_CONFIG='dram',
                                     MathFidelity=SimpleNamespace(HiFi4='hifi4'),
                                     WormholeComputeKernelConfig=Mock(return_value='kernel'),
                                     SDPAProgramConfig=Mock(return_value='program'),
                                     transformer=SimpleNamespace(scaled_dot_product_attention=Mock(return_value='out')))
        return operations

    def test_folded_sdpa_is_draft_sdpas_call_argument_for_argument(self):
        from draft_attention import draft_sdpa

        tensor = lambda *shape: SimpleNamespace(shape=shape, dtype='bf16')
        served, folded = self.recording(), self.recording()
        draft_sdpa(served, tensor(1, 16, 32, 128), tensor(1, 4, 2080, 128), tensor(1, 4, 2080, 128),
                   tensor(1, 1, 32, 2080))
        query, key, value, mask = object(), object(), object(), object()
        self.assertEqual(folded_sdpa(folded, query, key, value, mask), 'out')
        for name in ('WormholeComputeKernelConfig', 'SDPAProgramConfig'):
            left, right = getattr(served, name).call_args, getattr(folded, name).call_args
            self.assertEqual((left.args, left.kwargs), (right.args, right.kwargs), name)
        left = served.transformer.scaled_dot_product_attention.call_args
        right = folded.transformer.scaled_dot_product_attention.call_args
        without_mask = lambda kwargs: {name: value for name, value in kwargs.items() if name != 'attn_mask'}
        self.assertEqual(without_mask(left.kwargs), without_mask(right.kwargs))
        self.assertEqual(set(left.kwargs), set(right.kwargs))
        self.assertIs(right.kwargs['attn_mask'], mask)
        self.assertEqual(right.args, (query, key, value))
        self.assertEqual(right.kwargs['scale'], 128 ** -0.5)
        self.assertEqual(served.WormholeComputeKernelConfig.call_args.kwargs['fp32_dest_acc_en'], True)
        self.assertEqual(served.SDPAProgramConfig.call_args.kwargs['k_chunk_size'], 32)


# ---------------------------------------------------------------------------------------------
# The wiring.
# ---------------------------------------------------------------------------------------------

class Recorder:
    """An operations namespace that records every call, normalized, and returns a fresh token object."""

    bfloat16, float32, uint32 = 'bf16', 'fp32', 'u32'
    TILE_LAYOUT, ROW_MAJOR_LAYOUT, DRAM_MEMORY_CONFIG = 'tile', 'row', 'dram'
    MathFidelity = SimpleNamespace(HiFi4='hifi4')

    def __init__(self):
        from test_dflash_proposal_trace import Normalizer

        self.events, self.normalize = [], Normalizer()
        self.experimental = SimpleNamespace(rotary_embedding_hf=self.recorder('rotary_embedding_hf'))
        self.transformer = SimpleNamespace(scaled_dot_product_attention=self.recorder('sdpa'))

    def recorder(self, name):
        def call(*args, **kwargs):
            result = SimpleNamespace(token=name)
            self.events.append((name, self.normalize(args), self.normalize(kwargs), self.normalize(result)))
            return result
        return call

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        return self.recorder(name)


def branch_inputs(mask_rows, key_rows=4160):
    tensor = lambda *shape: SimpleNamespace(shape=shape, dtype='bf16')
    rope = {'q': tuple(tensor(1, 1, 32, 128) for _ in range(2)),
            'k': tuple(tensor(1, 1, key_rows, 128) for _ in range(2)),
            'live_k': tuple(tensor(1, 1, 32, 128) for _ in range(2))}
    caches = [{name: tensor(1, 4, 2048, 128) for name in ('k', 'v')} for _ in range(2)]
    return tensor(1, 1, 32, 5120), tensor(1, 1, 32, mask_rows), rope, caches


def run_branch(module, operations, *, mask_rows=4160, contexts=(2048, 2048), **extra):
    """execute_attention_branch of `module` (today's or the parent's) on the packed pair, every collaborator
    patched to a recorder. Returns (the calls into the collaborators, the fold and native attention mocks)."""
    calls = []

    def record(name, value=None):
        def side_effect(*args, **kwargs):
            result = value() if callable(value) else SimpleNamespace(token=name)
            calls.append((name, operations.normalize(args), operations.normalize(kwargs), operations.normalize(result)))
            return result
        return Mock(side_effect=side_effect)

    hidden, mask, rope, caches = branch_inputs(mask_rows)
    caches = [{name: SimpleNamespace(shape=(1, 4, context, 128), dtype='bf16') for name in ('k', 'v')}
              for context in contexts]
    users = [dict(position=4096, history_rows=contexts[0]), dict(position=9000, history_rows=contexts[1])]
    live = {name: SimpleNamespace(token='live-' + name) for name in ('q', 'k', 'v')}
    native, fold = record('native'), record('fold')
    with patch.object(module, 'grouped_causal_convolution', record('convolve')), \
            patch.object(module, 'gather_add_projection', record('gather')), \
            patch.object(module, 'concatenate_query_heads', record('concat_heads')), \
            patch.object(module, 'split_projected_heads', record('split_heads')), \
            patch('draft_kv_projection.project_key_value', record('project_kv', lambda: live)), \
            patch('dflash_t16_native_attention.attention', native), \
            patch('pair_row_exact.fold_attention', fold), \
            patch('dflash_t16_native_scope.require_active', return_value=None):
        parameters = dict(operations=operations, mesh='mesh', block_rows=16, native_head_layout=True,
                          native_proposal_attention=True, kernel='kernel', norm='norm', convolution='conv',
                          bases=['b0', 'b1', 'b2', 'b3'], projections=dict(q='wq', k='wk', v='wv'),
                          head_norms=dict(q='nq', k='nk'), output_projection='wo')
        output = module.execute_attention_branch(operations, 'mesh', 'collectives', hidden, None, mask, rope,
            lambda value: value, parameters=parameters, context=None, pack=users, cached_history=caches,
            native_proposal_mask_validated=True, **extra)
    return calls, native, fold, output


class BranchTests(unittest.TestCase):
    def test_row_exact_folds_the_same_heads_with_the_single_mask_and_skips_the_native_call(self):
        operations = Recorder()
        calls, native, fold, _ = run_branch(__import__('draft_attention_branch'), operations, mask_rows=2080,
                                            row_exact=True)
        native.assert_not_called()
        fold.assert_called_once()
        args, kwargs = fold.call_args.args, fold.call_args.kwargs
        self.assertIs(args[0], operations)
        self.assertEqual(tuple(args[4].shape), (1, 1, 32, 2080))
        self.assertEqual(kwargs, dict(mask_validated=True))
        # the keys and values are the assembled concat, exactly as the native call would have read them
        concats = [event for event in operations.events if event[0] == 'concat' and len(event[1][0]) == 6]
        self.assertEqual([operations.normalize(args[2]), operations.normalize(args[3])], [event[3] for event in concats])

    def test_the_folded_branch_is_todays_branch_but_for_the_attention_call(self):
        module = __import__('draft_attention_branch')
        on, off = Recorder(), Recorder()
        calls_on, _, _, _ = run_branch(module, on, mask_rows=2080, row_exact=True)
        calls_off, _, _, _ = run_branch(module, off)
        self.assertEqual(on.events, off.events, 'every ttnn call of the branch is unchanged')
        self.assertEqual([call[0] for call in calls_on], [call[0] if call[0] != 'native' else 'fold'
                                                         for call in calls_off])

    def test_row_exact_is_refused_off_the_pair_geometry_or_with_the_packed_mask(self):
        module = __import__('draft_attention_branch')
        for extra in (dict(mask_rows=4160, row_exact=True),
                      dict(mask_rows=2080, row_exact=True, contexts=(2048, 1024)),
                      dict(mask_rows=2080, row_exact=1),
                      dict(mask_rows=2080)):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                run_branch(module, Recorder(), **extra)

    def test_row_exact_without_a_pack_is_refused(self):
        from draft_attention_branch import execute_attention_branch

        operations = Recorder()
        hidden, mask, rope, caches = branch_inputs(2080)
        parameters = dict(operations=operations, mesh='mesh', block_rows=16, native_head_layout=True,
                          native_proposal_attention=True)
        with patch('dflash_t16_native_scope.require_active', return_value=None), self.assertRaises(ValueError):
            execute_attention_branch(operations, 'mesh', 'c', hidden, None, mask, rope, lambda value: value,
                                     parameters=parameters, context=2048, cached_history=caches[0],
                                     native_proposal_mask_validated=True, row_exact=True)
        self.assertEqual(operations.events, [])

    def test_flag_off_the_branch_is_the_parents_call_for_call(self):
        parent = parent_module('draft_attention_branch.py', 'draft_attention_branch_row_exact_parent')
        if parent is None:
            self.skipTest('no git history for %s' % PARENT)
        today = __import__('draft_attention_branch')
        runs = []
        for module in (parent, today):
            operations = Recorder()
            calls, _, _, _ = run_branch(module, operations)
            runs.append((operations.events, calls))
        self.assertGreater(len(runs[0][0]), 20)
        self.assertEqual(runs[1], runs[0])


class ExecuteProposalTests(unittest.TestCase):
    def fixture(self):
        from test_dflash_device_audit import ExecuteProposalTests as Audit

        return Audit()

    def pack(self):
        return [dict(position=4096, history_rows=2048), dict(position=8192, history_rows=2048)]

    def test_row_exact_reaches_every_layers_attention_branch_and_nothing_else(self):
        audit = self.fixture()
        device = audit.device()
        device.kv_history, device.native_proposal_attention = object(), True
        with patch('dflash_device.addresses', return_value=('a', 'b')):
            device.validated_native_proposal_masks.add(('a', 'b'))
            _, attention, mlp = audit.run_proposal(device, context=None, pack=self.pack(),
                                                   cached_history=[[object()] * 5] * 2, row_exact=True)
        self.assertEqual([call.kwargs.get('row_exact') for call in attention.call_args_list], [True] * 5)
        self.assertTrue(all('row_exact' not in call.kwargs for call in mlp.call_args_list))

    def test_without_it_no_branch_sees_the_keyword(self):
        audit = self.fixture()
        device = audit.device()
        device.kv_history, device.native_proposal_attention = object(), True
        with patch('dflash_device.addresses', return_value=('a', 'b')):
            device.validated_native_proposal_masks.add(('a', 'b'))
            _, attention, _ = audit.run_proposal(device, context=None, pack=self.pack(),
                                                 cached_history=[[object()] * 5] * 2)
        self.assertTrue(all('row_exact' not in call.kwargs for call in attention.call_args_list))

    def test_row_exact_off_a_native_pair_is_refused(self):
        audit = self.fixture()
        for native, pack, value in ((False, self.pack(), True), (True, None, True), (True, self.pack()[:1], True),
                                    (True, self.pack(), 1)):
            device = audit.device()
            device.kv_history, device.native_proposal_attention = object(), native
            cached = None if pack is None else [[object()] * 5] * len(pack)
            with self.subTest(native=native, pack=pack, value=value), \
                    patch('dflash_device.addresses', return_value=('a', 'b')), self.assertRaises(ValueError):
                device.validated_native_proposal_masks.add(('a', 'b'))
                audit.run_proposal(device, context=None if pack else 2048, pack=pack, cached_history=cached,
                                   row_exact=value)

    def test_flag_off_execute_proposal_is_the_parents_call_for_call(self):
        parent = parent_module('dflash_device.py', 'dflash_device_row_exact_parent')
        if parent is None:
            self.skipTest('no git history for %s' % PARENT)
        import dflash_device
        from test_dflash_proposal_trace import Normalizer

        def run(module, **extra):
            audit = self.fixture()
            device = audit.device()
            device.kv_history, device.native_proposal_attention = object(), True
            normalize, log = Normalizer(), []

            def recording(name, value):
                def side_effect(*args, **kwargs):
                    result = value()
                    log.append((name, normalize(args), normalize(kwargs), normalize(result)))
                    return result
                return Mock(side_effect=side_effect)

            for name in ('reshape', 'pad', 'rms_norm', 'matmul', 'typecast', 'slice',
                         'MatmulMultiCoreReuseMultiCast1DProgramConfig'):
                setattr(device.operations, name, recording(name, object))
            device.model.embd = recording('embd', object)
            device.operations.experimental.all_gather_async = recording('all_gather', object)
            with patch.object(module, 'projection_links', return_value=1), \
                    patch.object(module, 'shared_head_candidates', recording('head', lambda: ['chunks'])), \
                    patch.object(module, 'execute_attention_branch', recording('attention', object)), \
                    patch.object(module, 'execute_mlp_branch', recording('mlp', lambda: dict(output=object()))), \
                    patch.object(module, 'addresses', return_value=('a', 'b')):
                device.validated_native_proposal_masks.add(('a', 'b'))
                module.DFlashDevice.execute_proposal(device, 'ids', None, 'mask', {'q': 1, 'k': 2, 'live_k': 3},
                    context=None, pack=self.pack(), cached_history=[[object()] * 5] * 2, owned=[],
                    retain=lambda value: value, stage=lambda name, **values: log.append(('stage', name)), **extra)
            return log

        before, today = run(parent), run(dflash_device)
        self.assertGreater(len(before), 30)
        self.assertEqual(today, before)
        self.assertEqual(run(dflash_device, row_exact=False), before)


class PairTraceTests(unittest.TestCase):
    """Under the flag the pair trace folds its (2048, 2048) bucket and differs from the flag-off run in the mask and
    the row_exact keyword alone."""

    def setUp(self):
        import dflash_proposal_trace

        self.module = dflash_proposal_trace
        self.addCleanup(dflash_proposal_trace._PAIR_MASK_REFRESH_NOTED.clear)

    def scenario(self, module, context, **flags):
        from test_dflash_proposal_trace import RecordingOps, logged, pair_devices, pair_scenario

        def devices(ops, **options):
            made = pair_devices(ops, context=context)
            made[1].position = max(made[1].position, context + 100)
            return made

        ops = RecordingOps()
        lines, loguru = logged(module)
        import dflash_packed_proposal

        # every once-per-process marker the scenario can log, fresh for each run
        del module._PAIR_MASK_REFRESH_NOTED[:]
        with clean_environment(**flags), loguru, patch('test_dflash_proposal_trace.pair_devices', devices),                 patch.object(pair_row_exact, '_NOTED', []), patch.object(dflash_packed_proposal, '_ROUND_B1_NOTED', []):
            trace, events = pair_scenario(module, ops)
        return trace, events, lines, ops

    @staticmethod
    def masks_and_keyword_removed(events):
        """The events with every host mask replaced by 'MASK' and execute_proposal's row_exact dropped."""
        def scrub(value):
            if isinstance(value, tuple) and len(value) == 4 and value[0] == 'tensor' \
                    and tuple(value[1][:3]) == (1, 1, 32) and value[1][3] in (2080, 4160):
                return 'MASK'
            if isinstance(value, tuple):
                return tuple(scrub(item) for item in value if not (isinstance(item, tuple) and len(item) == 2
                                                                   and item[0] == repr('row_exact')))
            return value
        return [scrub(event) for event in events]

    def test_the_2048_bucket_folds_and_nothing_else_changes(self):
        for extra in ({}, {'QWEN_FAST_PAIR_MASK_REFRESH': '1'}, {'QWEN_FAST_ROUND_B1': '1'}):
            with self.subTest(flags=extra):
                _, off, off_lines, _ = self.scenario(self.module, 2048, **extra)
                trace, on, lines, ops = self.scenario(self.module, 2048, **{FLAG: '1'}, **extra)
                self.assertEqual(self.masks_and_keyword_removed(on), self.masks_and_keyword_removed(off))
                self.assertNotEqual(on, off)
                calls = [event for event in on if event[0] == 'execute_proposal']
                folded = [event for event in calls if (repr('row_exact'), True) in event[3]]
                # build (eager), capture, then the (500, 500) bucket after the geometry change: not folded
                self.assertEqual(len(folded), 2)
                self.assertEqual(len(calls), 4)
                widths = [event[1][1][3] for event in on if event[0] == 'from_torch' and event[1][0] == 'tensor'
                          and tuple(event[1][1][:3]) == (1, 1, 32) and event[1][1][3] in (2080, 4160, 1088)]
                # the folded (2048, 2048) bucket's single mask, then the (500, 500) bucket's packed one; the
                # refresh's per-round copies go through from_torch too, each the kept mask again
                self.assertEqual(sorted(set(widths)), [1088, 2080])
                self.assertEqual(widths[0], 2080)
                marked = [line for line in lines if line.startswith(pair_row_exact.MARKER)]
                self.assertEqual(marked, ['[PINDIAG] pair row exact engaged pair=[0,1] context=2048,2048 '
                                          'heads=32/8 keys=2080'])
                self.assertFalse([line for line in off_lines if pair_row_exact.MARKER in line])

    def test_the_kept_host_mask_is_the_single_mask_and_the_refresh_copies_it(self):
        trace, events, _, ops = self.scenario(self.module, 2048, **{FLAG: '1', 'QWEN_FAST_PAIR_MASK_REFRESH': '1'})
        single = ops.normalize(fold_mask((2048, 2048), 16))
        copies = [event for event in events if event[0] == 'copy_host_to_device_tensor' and event[3] == single]
        self.assertEqual(len(copies), 4, 'the build update and three rounds')

    def test_a_2048_bucket_without_the_flag_or_with_0_is_todays(self):
        _, off, _, _ = self.scenario(self.module, 2048)
        _, zero, _, _ = self.scenario(self.module, 2048, **{FLAG: '0'})
        self.assertEqual(zero, off)

    def test_other_contexts_never_fold(self):
        _, off, _, _ = self.scenario(self.module, 300)
        _, on, lines, _ = self.scenario(self.module, 300, **{FLAG: '1'})
        self.assertEqual(on, off)
        self.assertFalse([line for line in lines if pair_row_exact.MARKER in line])

    def test_a_bad_value_fails_the_bucket_build(self):
        from test_dflash_proposal_trace import RecordingOps, pair_devices

        ops = RecordingOps()
        device_a, device_b = pair_devices(ops, context=2048)
        with clean_environment(**{FLAG: 'yes'}):
            trace = self.module.PreparedPackedDFlashProposal(device_a, device_b)
            with self.assertRaises(ValueError):
                trace.prepare_device(1, 2)
        self.assertEqual(trace.buckets, {})

    def test_flag_off_the_pair_trace_is_the_parents_call_for_call(self):
        parent = parent_module('dflash_proposal_trace.py', 'dflash_proposal_trace_row_exact_parent')
        if parent is None:
            self.skipTest('no git history for %s' % PARENT)
        for context in (300, 2048):
            for flags in ({}, {FLAG: '0'}, {'QWEN_FAST_ROUND_B1': '1'}, {'QWEN_FAST_PAIR_MASK_REFRESH': '1'},
                          {'QWEN_FAST_PAIR_MASK_AUDIT': '1', 'QWEN_FAST_PAIR_MASK_REFRESH': '1'}):
                with self.subTest(context=context, flags=flags):
                    _, before, before_lines, _ = self.scenario(parent, context, **flags)
                    _, today, lines, _ = self.scenario(self.module, context, **flags)
                    self.assertGreater(len(before), 50)
                    self.assertEqual(today, before)
                    self.assertEqual(lines, before_lines)


class ProposePackedTests(unittest.TestCase):
    def fixture(self):
        from test_dflash_packed_proposal import ProposePackedTests as Packed

        return Packed()

    def run_packed(self, module, *, contexts=(2048, 2048), **extra):
        from test_dflash_proposal_trace import Normalizer

        packed = self.fixture()
        device, operations = packed.device()
        normalize, log = Normalizer(), []

        def upload(*args, **kwargs):
            result = object()
            log.append(('from_torch', normalize(args), normalize(kwargs)))
            return result

        operations.from_torch = Mock(side_effect=upload)
        slots = [dict(position=4096, history_rows=contexts[0], kv_history=[{'k': 0, 'v': 0}] * 5),
                 dict(position=9000, history_rows=contexts[1], kv_history=[{'k': 1, 'v': 1}] * 5)]
        with patch('gdn_multitoken_conv.addresses', return_value=('a', 'b')), \
                patch('gdn_multitoken_conv.release_owned'), \
                patch.object(module, 'select_device_outputs', return_value=(('x',), ('y',))):
            tokens = module.propose_packed(device, slots, [11, 22], [15, 15], **extra)
        call = device.execute_proposal.call_args
        log.append(('execute_proposal', normalize(call.args), normalize(call.kwargs)))
        return log, tokens, call

    def test_row_exact_uploads_the_single_mask_and_forwards_the_keyword(self):
        import dflash_packed_proposal

        log, tokens, call = self.run_packed(dflash_packed_proposal, row_exact=True)
        self.assertIs(call.kwargs.get('row_exact'), True)
        masks = [entry[1][0] for entry in log if entry[0] == 'from_torch' and entry[1][0][0] == 'tensor'
                 and tuple(entry[1][0][1][:3]) == (1, 1, 32) and entry[1][0][1][3] in (2080, 4160)]
        self.assertEqual([mask[1][3] for mask in masks], [2080])
        self.assertEqual(tokens, (('x',), ('y',)))

    def test_row_exact_off_the_pair_geometry_is_refused_before_any_upload(self):
        import dflash_packed_proposal

        for contexts, value in (((2048, 1024), True), ((2048, 2048), 1)):
            with self.subTest(contexts=contexts, value=value), self.assertRaises(ValueError):
                self.run_packed(dflash_packed_proposal, contexts=contexts, row_exact=value)

    def test_flag_off_propose_packed_is_the_parents_call_for_call(self):
        parent = parent_module('dflash_packed_proposal.py', 'dflash_packed_proposal_row_exact_parent')
        if parent is None:
            self.skipTest('no git history for %s' % PARENT)
        import dflash_packed_proposal

        for contexts in ((2048, 2048), (2048, 1024)):
            with self.subTest(contexts=contexts):
                before = self.run_packed(parent, contexts=contexts)[0]
                self.assertEqual(self.run_packed(dflash_packed_proposal, contexts=contexts)[0], before)
                self.assertEqual(self.run_packed(dflash_packed_proposal, contexts=contexts, row_exact=False)[0],
                                 before)


# ---------------------------------------------------------------------------------------------
# The arm and the gate.
# ---------------------------------------------------------------------------------------------

class ArmTests(unittest.TestCase):
    ARM = HERE / 'lever_n_m3native_run_arm.sh'
    START = "# The pair drafter's row-1 fix (pair_row_exact.py; default off)."
    END = '  echo "start order $start_order (stagger $stagger s)"' + chr(10) + 'fi' + chr(10)

    def text(self):
        return self.ARM.read_text(encoding='utf-8')

    def validate(self, **environ):
        import shutil
        import subprocess

        bash = shutil.which('bash')
        if bash is None:
            self.skipTest('no bash')
        text = self.text()
        start = text.index(self.START)
        end = text.index(self.END, start) + len(self.END)
        script = ('set -euo pipefail' + chr(10) + 'users="${USERS_UNDER_TEST}"' + chr(10)
                  + 'stagger="${STAGGER_UNDER_TEST}"' + chr(10) + text[start:end] + 'echo VALID' + chr(10))
        try:
            return subprocess.run([bash, '-c', script], capture_output=True, text=True, timeout=60,
                                  env=dict(PATH=os.environ.get('PATH', ''), USERS_UNDER_TEST=environ.pop('users', '4'),
                                           STAGGER_UNDER_TEST=environ.pop('stagger', '0.25'), **environ))
        except OSError as error:
            self.skipTest('bash unusable: %s' % error)

    def test_the_arm_refuses_what_the_gate_or_the_server_would(self):
        packed = dict(M3NATIVE_PACKED_PROPOSAL='1')
        for environ in ({}, dict(packed, M3NATIVE_PAIR_ROW_EXACT='1'),
                        dict(packed, M3NATIVE_PAIR_ROW_EXACT='1', M3NATIVE_START_ORDER='1,0,3,2'),
                        dict(M3NATIVE_START_ORDER='0,1,2,3', stagger='5')):
            with self.subTest(accepted=environ):
                result = self.validate(**dict(environ))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('VALID', result.stdout)
        for environ, message in (
                (dict(packed, M3NATIVE_PAIR_ROW_EXACT='yes'), 'must be 1 or unset'),
                (dict(packed, M3NATIVE_PAIR_ROW_EXACT='0'), 'must be 1 or unset'),
                (dict(M3NATIVE_PAIR_ROW_EXACT='1'), 'needs M3NATIVE_PACKED_PROPOSAL=1'),
                (dict(packed, M3NATIVE_PAIR_ROW_EXACT='1', users='1'), 'folds the packed pairs only'),
                (dict(packed, M3NATIVE_PAIR_ROW_EXACT='1', M3NATIVE_SEQUENTIAL_USERS='4'), 'folds the packed pairs only'),
                (dict(M3NATIVE_START_ORDER='1,0,3,2', stagger='0'), 'needs M3NATIVE_STAGGER > 0'),
                (dict(M3NATIVE_START_ORDER='1,0,3,2', stagger='0.0'), 'needs M3NATIVE_STAGGER > 0'),
                (dict(M3NATIVE_START_ORDER='1 0 3 2'), 'comma-separated user indices'),
                (dict(M3NATIVE_START_ORDER='1,0,3,2', M3NATIVE_SEQUENTIAL_USERS='4'), 'a sequential arm has none')):
            with self.subTest(refused=environ):
                result = self.validate(**dict(environ))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_the_flag_crosses_after_h2_and_the_order_reaches_the_gate(self):
        text = self.text()
        entry = text.index('--entrypoint python3')
        line = '${M3NATIVE_PAIR_ROW_EXACT:+-e QWEN_FAST_PAIR_ROW_EXACT=1}'
        self.assertEqual(text.count(line), 1)
        self.assertLess(text.index('${M3NATIVE_GDN_AFTER_PAIRS:+-e QWEN_FAST_GDN_AFTER_PAIRS=1}'), text.index(line))
        self.assertLess(text.index(line), entry)
        order = '${M3NATIVE_START_ORDER:+--start-order $M3NATIVE_START_ORDER}'
        self.assertEqual(text.count(order), 1)
        self.assertGreater(text.index(order), entry, 'a gate argument, after the image')
        self.assertLess(text.index(self.START), text.index('docker run --rm --name "$name"'))

    def test_the_arm_parses_with_lf_endings(self):
        import shutil
        import subprocess

        bash = shutil.which('bash')
        if bash is None:
            self.skipTest('no bash')
        result = subprocess.run([bash, '-n', str(self.ARM)], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(b'\r\n', self.ARM.read_bytes())

    def test_unset_nothing_crosses(self):
        import shutil
        import subprocess

        bash = shutil.which('bash')
        if bash is None:
            self.skipTest('no bash')
        script = ('printf "%s|" ${M3NATIVE_PAIR_ROW_EXACT:+-e QWEN_FAST_PAIR_ROW_EXACT=1} '
                  '${M3NATIVE_START_ORDER:+--start-order $M3NATIVE_START_ORDER}' + chr(10))
        try:
            unset = subprocess.run([bash, '-c', script], env=dict(PATH=os.environ.get('PATH', '')),
                                   capture_output=True, text=True, timeout=60)
            both = subprocess.run([bash, '-c', script], env=dict(PATH=os.environ.get('PATH', ''),
                                                                 M3NATIVE_PAIR_ROW_EXACT='1', M3NATIVE_START_ORDER='1,0,3,2'),
                                  capture_output=True, text=True, timeout=60)
        except OSError as error:
            self.skipTest('bash unusable: %s' % error)
        self.assertEqual(unset.stdout.strip('|'), '')
        self.assertEqual(both.stdout, '-e|QWEN_FAST_PAIR_ROW_EXACT=1|--start-order|1,0,3,2|')


class GateTests(unittest.TestCase):
    def test_the_marker_is_required_with_the_flag_on_a_pair_arm(self):
        import lever_n_m3native_gate as gate

        self.assertEqual(gate.PAIR_ROW_EXACT_MARKER, pair_row_exact.MARKER)
        self.assertEqual(gate.PAIR_ROW_EXACT_FLAG, pair_row_exact.FLAG)
        self.assertEqual(gate.required_flag_markers({FLAG: '1'}, 4)[FLAG], [pair_row_exact.MARKER])
        self.assertEqual(gate.required_flag_markers({FLAG: '1'}, 2)[FLAG], [pair_row_exact.MARKER])
        self.assertNotIn(FLAG, gate.required_flag_markers({FLAG: '1'}, 1), 'a single stream builds no pair')
        self.assertNotIn(FLAG, gate.required_flag_markers({FLAG: '0'}, 4))
        self.assertNotIn(FLAG, gate.required_flag_markers({}, 4))
        line = '2026-09-25 | INFO | [PINDIAG] pair row exact engaged pair=[0,1] context=2048,2048 heads=32/8 keys=2080'
        self.assertEqual([entry for entry in gate.flag_marker_report({FLAG: '1'}, 4, line)['missing']
                          if FLAG in entry], [])
        self.assertEqual([entry for entry in gate.flag_marker_report({FLAG: '1'}, 4, 'nothing')['missing']
                          if FLAG in entry], ['%s: %s' % (FLAG, pair_row_exact.MARKER)])

    def test_the_start_order_is_a_permutation_with_a_stagger(self):
        import lever_n_m3native_gate as gate

        base = ['--users', '4', '--stagger', '0.25']
        options = gate.parse_options(base + ['--start-order', '1,0,3,2'])
        self.assertEqual(options.start_order, [1, 0, 3, 2])
        self.assertEqual(gate.request_order(options), [1, 0, 3, 2])
        self.assertEqual(gate.request_order(gate.parse_options(base)), [0, 1, 2, 3])
        self.assertIsNone(gate.parse_options(base).start_order)
        for argv in (base + ['--start-order', '1,0,3'], base + ['--start-order', '1,1,2,3'],
                     base + ['--start-order', '0,1,2,4'], base + ['--start-order', 'one'],
                     ['--users', '4', '--start-order', '1,0,3,2'],
                     ['--users', '4', '--stagger', '0', '--start-order', '1,0,3,2']):
            with self.subTest(argv=argv), patch('sys.stderr'), self.assertRaises(SystemExit):
                gate.parse_options(argv)

    def test_the_report_names_the_order_only_when_one_was_given(self):
        import lever_n_m3native_gate as gate

        report = dict(packed_fingerprints=dict(admission_order=[1, 0, 3, 2]))
        with patch('builtins.print'):
            gate.start_order_report(report, gate.parse_options(['--users', '4', '--stagger', '0.25']))
            self.assertEqual(set(report), {'packed_fingerprints'})
            gate.start_order_report(report, gate.parse_options(['--users', '4', '--stagger', '0.25',
                                                                '--start-order', '1,0,3,2']))
        self.assertEqual(report['start_order'], dict(requested=[1, 0, 3, 2], admitted=[1, 0, 3, 2], matches=True))
        other = dict(packed_fingerprints=dict(admission_order=[0, 1, 2, 3]))
        with patch('builtins.print'):
            gate.start_order_report(other, gate.parse_options(['--users', '4', '--stagger', '0.25',
                                                               '--start-order', '1,0,3,2']))
        self.assertFalse(other['start_order']['matches'])

    def test_the_threads_start_in_the_request_order(self):
        source = (HERE / 'lever_n_m3native_gate.py').read_text(encoding='utf-8')
        self.assertIn('for position, index in enumerate(request_order(options) if threads else []):', source)
        self.assertIn('threads[index].start()', source)
        self.assertNotIn('for index, thread in enumerate(threads):', source)


# ---------------------------------------------------------------------------------------------
# Shipping.
# ---------------------------------------------------------------------------------------------

class ShippingTests(unittest.TestCase):
    def test_the_module_reaches_the_image_through_both_copy_lists(self):
        from test_serving_image_copy_closure import context_modules, dockerfile_modules, dockerfile_text

        for name in ('pair_row_exact.py', 'dflash_proposal_trace.py', 'dflash_device.py', 'draft_attention_branch.py',
                     'dflash_packed_proposal.py', 'dflash_batched_mask.py'):
            with self.subTest(module=name):
                self.assertIn(name, dockerfile_modules(dockerfile_text()))
                self.assertIn(name, context_modules())

    def test_the_cpu_suite_runs_this_file(self):
        self.assertRegex(CPU_WORKFLOW.read_text(encoding='utf-8'), r'python -B -m unittest [^\n]*\btest_pair_row_exact\b')

    def test_the_t16_admissions_hashed_sources_are_untouched(self):
        """draft_attention.py, dflash_t16_native_attention.py and dflash_attention_mask.py are hashed by the T16
        admission (dflash_t16_native_attention_gate.SOURCES): the fold must not change them."""
        import subprocess

        from dflash_t16_native_attention_gate import SOURCES

        try:
            result = subprocess.run(['git', 'diff', '--name-only', PARENT, '--', *SOURCES], capture_output=True,
                                    cwd=str(HERE), timeout=60, text=True)
        except (OSError, subprocess.SubprocessError):
            self.skipTest('no git')
        if result.returncode != 0:
            self.skipTest('no git history for %s' % PARENT)
        self.assertEqual(result.stdout.strip(), '')


if __name__ == '__main__':
    unittest.main()
